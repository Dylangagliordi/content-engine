import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from qualume.models.contracts import ExternalVideoAnnotation, ExternalVideoAnnotationV2
from qualume.annotate import gemini_watcher
from qualume.annotate.gemini_watcher import (
    ANALYSIS_PROMPT_VERSION,
    ANNOTATION_PROMPT,
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


def _good_v2_dict(**overrides):
    data = _good_annotation_dict(
        script_structure_timeline=[
            {"beat": "hook", "start_sec": 0.0, "end_sec": 2.5},
            {"beat": "demo", "start_sec": 2.5, "end_sec": 14.0},
            {"beat": "cta", "start_sec": 14.0, "end_sec": 17.0},
        ],
        on_screen_text=[
            {"text": "Why does nobody talk about this?", "timestamp_sec": 0.0},
            {"text": "follow for part 2", "timestamp_sec": 14.5},
        ],
        emotion_timeline=[
            {"timestamp_sec": 0.0, "emotion": "curious"},
            {"timestamp_sec": 9.0, "emotion": "delighted"},
        ],
        pacing_qualitative="medium",
        pacing_rationale="Steady shots of a few seconds each, no rapid montage",
        music_mood=["upbeat", "lofi"],
    )
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


def test_annotation_contract_v1_is_unchanged():
    """v1 stays usable as-is for code that still references it."""
    fields = set(ExternalVideoAnnotation.model_fields)
    assert "pacing_cuts_per_min" not in fields
    assert "music_mood" not in fields
    assert "script_structure_timeline" not in fields


# --- Contract v2 -------------------------------------------------------------


def test_v2_contract_extends_v1_and_accepts_good_input():
    assert issubclass(ExternalVideoAnnotationV2, ExternalVideoAnnotation)
    ann = ExternalVideoAnnotationV2(**_good_v2_dict())
    assert ann.hook_type == "question"  # v1 fields still present
    assert [b.beat for b in ann.script_structure_timeline] == ["hook", "demo", "cta"]
    assert ann.script_structure_timeline[1].end_sec == 14.0
    assert ann.on_screen_text[1].text == "follow for part 2"
    assert ann.on_screen_text[1].timestamp_sec == 14.5
    assert ann.emotion_timeline[1].emotion == "delighted"
    assert ann.pacing_qualitative == "medium"
    assert ann.music_mood == ["upbeat", "lofi"]


def test_v2_contract_never_asks_for_a_precise_cut_count():
    """Gemini samples video at 1 FPS, so an exact cuts-per-minute count is
    structurally unreliable — only a qualitative pacing read is asked for."""
    assert "pacing_cuts_per_min" not in ExternalVideoAnnotationV2.model_fields


def test_v2_contract_allows_empty_lists():
    ann = ExternalVideoAnnotationV2(**_good_v2_dict(on_screen_text=[], music_mood=[]))
    assert ann.on_screen_text == []
    assert ann.music_mood == []


def test_v2_contract_rejects_pacing_outside_enum():
    with pytest.raises(ValidationError):
        ExternalVideoAnnotationV2(**_good_v2_dict(pacing_qualitative="12 cuts/min"))


def test_v2_contract_rejects_beat_ending_before_it_starts():
    with pytest.raises(ValidationError):
        ExternalVideoAnnotationV2(**_good_v2_dict(
            script_structure_timeline=[{"beat": "hook", "start_sec": 3.0, "end_sec": 1.0}]))


@pytest.mark.parametrize("field,entry", [
    ("script_structure_timeline", {"beat": "hook", "start_sec": -1.0, "end_sec": 2.0}),
    ("on_screen_text", {"text": "hi", "timestamp_sec": -0.5}),
    ("emotion_timeline", {"timestamp_sec": -2.0, "emotion": "calm"}),
])
def test_v2_contract_rejects_negative_timestamps(field, entry):
    with pytest.raises(ValidationError):
        ExternalVideoAnnotationV2(**_good_v2_dict(**{field: [entry]}))


@pytest.mark.parametrize("field,entries", [
    ("script_structure_timeline", [{"beat": "demo", "start_sec": 5.0, "end_sec": 9.0},
                                   {"beat": "hook", "start_sec": 0.0, "end_sec": 5.0}]),
    ("on_screen_text", [{"text": "b", "timestamp_sec": 8.0}, {"text": "a", "timestamp_sec": 1.0}]),
    ("emotion_timeline", [{"timestamp_sec": 8.0, "emotion": "calm"},
                          {"timestamp_sec": 1.0, "emotion": "tense"}]),
])
def test_v2_contract_rejects_out_of_order_timelines(field, entries):
    with pytest.raises(ValidationError):
        ExternalVideoAnnotationV2(**_good_v2_dict(**{field: entries}))


def test_v2_contract_allows_simultaneous_on_screen_text():
    ann = ExternalVideoAnnotationV2(**_good_v2_dict(on_screen_text=[
        {"text": "top line", "timestamp_sec": 1.0}, {"text": "bottom line", "timestamp_sec": 1.0}]))
    assert len(ann.on_screen_text) == 2


def test_prompt_asks_for_every_v2_field():
    for field in ExternalVideoAnnotationV2.model_fields:
        assert field in ANNOTATION_PROMPT, f"ANNOTATION_PROMPT never mentions {field}"


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
    client = _mock_client(json.dumps(_good_v2_dict()))
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
    assert config.response_schema is ExternalVideoAnnotationV2

    assert isinstance(result.annotation, ExternalVideoAnnotationV2)
    assert result.annotation == ExternalVideoAnnotationV2(**_good_v2_dict())
    assert result.raw_response == _good_v2_dict()
    client.files.delete.assert_called_once_with(name="files/abc123")


def test_annotate_video_keeps_full_raw_response_including_unmapped_keys(gemini_key, fake_video):
    """Nothing Gemini returns is discarded: keys with no typed field still
    land in raw_response."""
    raw = _good_v2_dict(unmapped_extra={"anything": [1, 2, 3]})
    client = _mock_client(json.dumps(raw))
    with patch("qualume.annotate.gemini_watcher.genai.Client", return_value=client), \
         patch("qualume.annotate.gemini_watcher.time.sleep"):
        result = annotate_video(fake_video)
    assert result.raw_response == raw
    assert result.raw_response["unmapped_extra"] == {"anything": [1, 2, 3]}
    assert "unmapped_extra" not in result.annotation.model_dump()


def test_annotate_video_raises_when_raw_response_is_not_a_json_object(gemini_key, fake_video):
    client = _mock_client(json.dumps([_good_v2_dict()]))
    with patch("qualume.annotate.gemini_watcher.genai.Client", return_value=client), \
         patch("qualume.annotate.gemini_watcher.time.sleep"):
        with pytest.raises(RuntimeError, match="JSON object"):
            annotate_video(fake_video)
    client.files.delete.assert_called_once_with(name="files/abc123")


def test_annotate_video_returns_raw_response_when_validation_fails(gemini_key, fake_video):
    """A field that fails our strict schema (e.g. energy_level out of 1-5)
    must NOT discard the whole video — raw_response is preserved so nothing
    Gemini actually said is silently lost, and annotation is None with
    validation_error explaining why."""
    bad_dict = _good_v2_dict(energy_level=9)
    client = _mock_client(json.dumps(bad_dict))
    with patch("qualume.annotate.gemini_watcher.genai.Client", return_value=client), \
         patch("qualume.annotate.gemini_watcher.time.sleep"):
        result = annotate_video(fake_video)
    assert result.annotation is None
    assert result.validation_error is not None
    assert result.raw_response == bad_dict  # nothing lost, even though validation failed
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
        raw = _good_v2_dict(social_proof_used=True, unmapped_extra="kept")
        result = gemini_watcher.AnnotationResult(
            annotation=ExternalVideoAnnotationV2(**raw), raw_response=raw)

        with patch("qualume.annotate.gemini_watcher.annotate_video", return_value=result) as mock_annotate:
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
                              analysis_model, analysis_prompt_version, analyzed_at,
                              script_structure_timeline, on_screen_text, emotion_timeline,
                              pacing_qualitative, music_mood_v2, raw_gemini_response
                       FROM external_creative_dna WHERE id = %s""",
                    (dna_id,),
                )
                row = cur.fetchone()
        finally:
            conn.close()

        assert row is not None
        (eci_id, hook_type, hook_text, visual, fmt, tags, cta, caption, proof, energy, tone,
         voice, pacing, music, extended, model, prompt_version, analyzed_at,
         timeline, on_screen, emotions, pacing_q, music_v2, raw_response) = row
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
        assert pacing is None  # never filled: no precise cut count at 1 FPS
        assert music is None   # legacy column stays NULL; real data is music_mood_v2
        assert extended == {}
        assert model == GEMINI_MODEL
        assert prompt_version == ANALYSIS_PROMPT_VERSION == "v2"
        assert analyzed_at is not None
        assert timeline == raw["script_structure_timeline"]
        assert on_screen == raw["on_screen_text"]
        assert emotions == raw["emotion_timeline"]
        assert pacing_q == "medium"
        assert music_v2 == ["upbeat", "lofi"]
        assert raw_response == raw  # complete, including unmapped_extra
    finally:
        _cleanup_account(handle)


def test_annotate_and_store_keeps_raw_response_when_validation_fails():
    """The actual fix: a validation failure must not lose the video's data.
    raw_gemini_response and validation_error are populated; typed fields we
    can read straight off the raw dict (hook_type etc) are still filled in;
    the nested timeline fields that are NULL on a failed annotation stay
    NULL rather than being guessed."""
    handle = f"test_annot_degraded_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    try:
        account_id = add_watched_account("tiktok", handle)
        item_id = _insert_item(account_id, "/fake/path/9999.mp4")
        bad_raw = _good_v2_dict(energy_level=9)  # out of 1-5, fails strict validation
        result = gemini_watcher.AnnotationResult(
            annotation=None, raw_response=bad_raw, validation_error="energy_level out of range"
        )

        with patch("qualume.annotate.gemini_watcher.annotate_video", return_value=result):
            dna_id = annotate_and_store(item_id)

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT hook_type, hook_text, script_structure_timeline,
                              raw_gemini_response, validation_error
                       FROM external_creative_dna WHERE id = %s""",
                    (dna_id,),
                )
                hook_type, hook_text, timeline, raw_response, val_error = cur.fetchone()
        finally:
            conn.close()

        assert hook_type == "question"  # best-effort scalar still filled
        assert hook_text == "Why does nobody talk about this?"
        assert timeline is None  # nested/typed field NOT guessed when validation failed
        assert raw_response == bad_raw  # nothing lost
        assert val_error == "energy_level out of range"
    finally:
        _cleanup_account(handle)
