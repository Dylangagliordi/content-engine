import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from qualume.models.contracts import ExternalVideoAnnotation
from qualume.annotate import gemini_watcher
from qualume.annotate.gemini_watcher import (
    ANALYSIS_PROMPT_VERSION,
    GEMINI_MODEL,
    annotate_video,
    annotate_and_store,
)
from qualume.db.watchlist import add_watched_account
from qualume.db.connection import get_connection

# The genai client is always mocked here: real Gemini calls would be slow,
# cost money, and be flaky from the test suite (same reasoning as mocking
# yt_dlp in test_tiktok_reader.py).


def _good_annotation_dict(**overrides):
    data = {
        "hook_type": "question",
        "hook_text": "Why does nobody talk about this?",
        "visual_hook_description": "Close-up of hands opening a jar",
        "format_tag": "talking_head",
        "script_structure_tags": ["problem", "demo", "cta"],
        "cta_type": "follow",
        "caption_style": "bold_white_center",
        "social_proof_used": False,
        "energy_level": 3,
        "tone_category": "sincere",
        "voice_delivery_style": "conversational, direct to camera",
    }
    data.update(overrides)
    return data


# --- Contract ----------------------------------------------------------------


def test_annotation_contract_accepts_good_input():
    ann = ExternalVideoAnnotation(**_good_annotation_dict())
    assert ann.energy_level == 3
    assert ann.tone_category == "sincere"
    assert ann.script_structure_tags == ["problem", "demo", "cta"]


@pytest.mark.parametrize("bad_energy", [0, 6, 10])
def test_annotation_contract_rejects_out_of_range_energy_level(bad_energy):
    with pytest.raises(ValidationError):
        ExternalVideoAnnotation(**_good_annotation_dict(energy_level=bad_energy))


def test_annotation_contract_rejects_tone_category_outside_enum():
    with pytest.raises(ValidationError):
        ExternalVideoAnnotation(**_good_annotation_dict(tone_category="whimsical"))


def test_annotation_contract_has_no_computed_signal_fields():
    """pacing_cuts_per_min / music_mood are computed by a separate (future)
    signal-processing job — Gemini must never be asked to guess them."""
    fields = set(ExternalVideoAnnotation.model_fields)
    assert "pacing_cuts_per_min" not in fields
    assert "music_mood" not in fields


# --- annotate_video ----------------------------------------------------------


def _mock_client(response_text: str, states=("PROCESSING", "ACTIVE")):
    client = MagicMock()
    uploaded = SimpleNamespace(name="files/abc123", state=SimpleNamespace(name=states[0]))
    client.files.upload.return_value = uploaded
    client.files.get.side_effect = [
        SimpleNamespace(name="files/abc123", state=SimpleNamespace(name=s)) for s in states[1:]
    ]
    client.models.generate_content.return_value = SimpleNamespace(text=response_text)
    return client


@pytest.fixture
def fake_video(tmp_path):
    path = tmp_path / "7123.mp4"
    path.write_bytes(b"fake video bytes")
    return str(path)


@pytest.fixture
def gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")


def test_annotate_video_raises_clear_error_when_api_key_missing(monkeypatch, fake_video):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch("qualume.annotate.gemini_watcher.genai.Client") as client_cls:
        with pytest.raises(RuntimeError, match=r"GEMINI_API_KEY.*\.env"):
            annotate_video(fake_video)
    client_cls.assert_not_called()


def test_annotate_video_uploads_waits_for_active_and_constrains_output(gemini_key, fake_video):
    client = _mock_client(json.dumps(_good_annotation_dict()))
    with patch("qualume.annotate.gemini_watcher.genai.Client", return_value=client) as client_cls, \
         patch("qualume.annotate.gemini_watcher.time.sleep"):
        result = annotate_video(fake_video)

    client_cls.assert_called_once_with(api_key="test-key-not-real")
    client.files.upload.assert_called_once_with(file=fake_video)
    client.files.get.assert_called_with(name="files/abc123")

    call = client.models.generate_content.call_args
    assert call.kwargs["model"] == GEMINI_MODEL
    contents = call.kwargs["contents"]
    assert contents[0].name == "files/abc123"  # the ACTIVE file, not a path string
    assert isinstance(contents[1], str) and contents[1].strip()
    config = call.kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is ExternalVideoAnnotation

    assert isinstance(result, ExternalVideoAnnotation)
    assert result == ExternalVideoAnnotation(**_good_annotation_dict())
    client.files.delete.assert_called_once_with(name="files/abc123")


def test_annotate_video_fails_loud_when_gemini_returns_invalid_annotation(gemini_key, fake_video):
    """The SDK's own response.parsed silently becomes None on a validation
    failure — we must raise instead of returning nothing."""
    client = _mock_client(json.dumps(_good_annotation_dict(energy_level=9)))
    with patch("qualume.annotate.gemini_watcher.genai.Client", return_value=client), \
         patch("qualume.annotate.gemini_watcher.time.sleep"):
        with pytest.raises(ValidationError):
            annotate_video(fake_video)
    client.files.delete.assert_called_once_with(name="files/abc123")


def test_annotate_video_raises_when_file_processing_fails(gemini_key, fake_video):
    client = _mock_client("{}", states=("PROCESSING", "FAILED"))
    with patch("qualume.annotate.gemini_watcher.genai.Client", return_value=client), \
         patch("qualume.annotate.gemini_watcher.time.sleep"):
        with pytest.raises(RuntimeError, match="FAILED"):
            annotate_video(fake_video)
    client.models.generate_content.assert_not_called()


def test_annotate_video_raises_when_local_file_missing(gemini_key, tmp_path):
    with patch("qualume.annotate.gemini_watcher.genai.Client") as client_cls:
        with pytest.raises(FileNotFoundError):
            annotate_video(str(tmp_path / "does_not_exist.mp4"))
    client_cls.return_value.files.upload.assert_not_called()


# --- annotate_and_store ------------------------------------------------------


def _cleanup_account(handle: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """DELETE FROM external_creative_dna
                   WHERE external_content_item_id IN (
                       SELECT eci.id FROM external_content_items eci
                       JOIN watched_accounts wa ON wa.id = eci.watched_account_id
                       WHERE wa.handle = %s)""",
                (handle,),
            )
            cur.execute(
                """DELETE FROM external_content_items
                   WHERE watched_account_id IN (
                       SELECT id FROM watched_accounts WHERE handle = %s)""",
                (handle,),
            )
            cur.execute("DELETE FROM watched_accounts WHERE handle = %s", (handle,))
        conn.commit()
    finally:
        conn.close()


def _insert_item(account_id: str, local_video_path):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            vid = uuid.uuid4().hex[:12]
            cur.execute(
                """INSERT INTO external_content_items
                       (watched_account_id, platform, platform_video_id, url, local_video_path)
                   VALUES (%s, 'tiktok', %s, %s, %s) RETURNING id""",
                (account_id, f"test_{vid}", f"https://www.tiktok.com/@test/video/{vid}", local_video_path),
            )
            item_id = str(cur.fetchone()[0])
        conn.commit()
        return item_id
    finally:
        conn.close()


def test_annotate_and_store_raises_when_local_video_path_is_null():
    handle = f"test_annot_null_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    try:
        account_id = add_watched_account("tiktok", handle)
        item_id = _insert_item(account_id, None)
        with patch("qualume.annotate.gemini_watcher.annotate_video") as mock_annotate:
            with pytest.raises(ValueError, match="local_video_path"):
                annotate_and_store(item_id)
        mock_annotate.assert_not_called()
    finally:
        _cleanup_account(handle)


def test_annotate_and_store_raises_for_unknown_item():
    with patch("qualume.annotate.gemini_watcher.annotate_video") as mock_annotate:
        with pytest.raises(ValueError, match="not found"):
            annotate_and_store(str(uuid.uuid4()))
    mock_annotate.assert_not_called()


def test_annotate_and_store_inserts_row_with_provenance():
    handle = f"test_annot_store_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    try:
        account_id = add_watched_account("tiktok", handle)
        item_id = _insert_item(account_id, "/fake/path/7123.mp4")
        annotation = ExternalVideoAnnotation(**_good_annotation_dict(social_proof_used=True))

        with patch("qualume.annotate.gemini_watcher.annotate_video", return_value=annotation) as mock_annotate:
            dna_id = annotate_and_store(item_id)
        mock_annotate.assert_called_once_with("/fake/path/7123.mp4")

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT external_content_item_id, hook_type, hook_text, visual_hook_description,
                              format_tag, script_structure_tags, cta_type, caption_style,
                              social_proof_used, energy_level, tone_category, voice_delivery_style,
                              pacing_cuts_per_min, music_mood, extended_features,
                              analysis_model, analysis_prompt_version, analyzed_at
                       FROM external_creative_dna WHERE id = %s""",
                    (dna_id,),
                )
                row = cur.fetchone()
        finally:
            conn.close()

        assert row is not None
        (eci_id, hook_type, hook_text, visual, fmt, tags, cta, caption, proof, energy, tone,
         voice, pacing, music, extended, model, prompt_version, analyzed_at) = row
        assert str(eci_id) == item_id
        assert hook_type == "question"
        assert hook_text == "Why does nobody talk about this?"
        assert visual == "Close-up of hands opening a jar"
        assert fmt == "talking_head"
        assert tags == ["problem", "demo", "cta"]
        assert cta == "follow"
        assert caption == "bold_white_center"
        assert proof is True
        assert energy == 3
        assert tone == "sincere"
        assert voice == "conversational, direct to camera"
        assert pacing is None  # computed-signals job, not Gemini
        assert music is None   # computed-signals job, not Gemini
        assert extended == {}
        assert model == GEMINI_MODEL
        assert prompt_version == ANALYSIS_PROMPT_VERSION == "v1"
        assert analyzed_at is not None
    finally:
        _cleanup_account(handle)
