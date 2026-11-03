"""Tests for the Gemini annotation fallback-model feature — real fix per
2026-11-03 user decision ("ensure annotations get sped up"). Root cause
confirmed live: GEMINI_MODEL (gemini-3.1-flash-lite) is capped at the
real free-tier limit of 500 requests/day (confirmed via Google's own
rate-limits docs), and the real annotation backlog (~10,400 videos) would
take 20+ days to clear at that rate alone.

Confirmed live against the real API (2026-11-03): gemini-3.5-flash-lite
is a SEPARATE model with its OWN separate 500 RPD free-tier quota
(Google's docs: "Limits vary depending on the specific model being
used" — not pooled across models in a project) — genuinely watches
video successfully, but returns a persistent 503 ServerError when a
STRICT response_schema is attached alongside video input (same known
503-on-video-input pattern already documented in this file's module
docstring for other newer models). Dropping the strict schema (keep
response_mime_type='application/json' only) avoids the 503 and
produces real, correctly-shaped JSON — this project's own existing
degraded-path validation (best-effort field extraction + raw_response/
validation_error storage) already handles exactly this shape, so no
new fallback-storage logic is needed, only a model/schema fallback at
the API-call layer.
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from google.genai import errors

from qualume.annotate.gemini_watcher import (
    GEMINI_MODEL,
    GEMINI_MODEL_FALLBACK,
    annotate_video,
)


def _good_v6_dict(**overrides):
    # A real, fully-validating ExternalVideoAnnotationV6 payload — every
    # field value checked against the actual schema (not guessed), so
    # this test exercises real validation behavior, same discipline as
    # test_gemini_watcher.py's own _good_v6_dict builder.
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
        "script_structure_timeline": [
            {"beat": "hook", "start_sec": 0.0, "end_sec": 2.5},
            {"beat": "demo", "start_sec": 2.5, "end_sec": 14.0},
            {"beat": "cta", "start_sec": 14.0, "end_sec": 17.0},
        ],
        "on_screen_text": [
            {"text": "Why does nobody talk about this?", "timestamp_sec": 0.0},
            {"text": "follow for part 2", "timestamp_sec": 14.5},
        ],
        "emotion_timeline": [
            {"timestamp_sec": 0.0, "emotion": "curious"},
            {"timestamp_sec": 9.0, "emotion": "delighted"},
        ],
        "pacing_qualitative": "medium",
        "pacing_rationale": "Steady shots of a few seconds each, no rapid montage",
        "music_mood": ["upbeat", "lofi"],
        "shot_breakdown": [
            {
                "start_sec": 0.0, "end_sec": 2.5, "shot_type": "close_up",
                "camera_movement": "static", "setting": "kitchen",
                "setting_category": "home_indoor", "color_temperature": "warm",
                "dominant_colors": ["pink", "white"], "lighting": "bright_natural",
                "subject_action": "opening a jar",
            },
            {
                "start_sec": 2.5, "end_sec": 14.0, "shot_type": "close_up",
                "camera_movement": "handheld", "setting": "kitchen",
                "setting_category": "home_indoor", "color_temperature": "warm",
                "dominant_colors": ["pink", "white"], "lighting": "bright_natural",
                "subject_action": "demonstrating the product",
            },
        ],
        "full_transcript": [
            {"start_sec": 0.0, "text": "Why does nobody talk about this?"},
            {"start_sec": 3.0, "text": "Let me show you exactly how it works."},
        ],
        "presenters": [
            {
                "apparent_age_range": "30s",
                "apparent_gender_presentation": "female",
                "relatable_archetype": "mom",
                "wardrobe_style": "casual loungewear",
                "delivery_affect": "passionately talking",
            }
        ],
        "delivery_mode": "direct_to_camera",
        "video_topic_summary": "an herbal immune supplement for kids",
    }
    data.update(overrides)
    return data



def _server_error(code=503):
    return errors.ServerError(code, {"error": {"code": code, "message": "high demand", "status": "UNAVAILABLE"}})


def _mock_client_with_fallback(primary_side_effect, fallback_response_text, states=("PROCESSING", "ACTIVE")):
    """Like test_gemini_watcher.py's _mock_client, but generate_content's
    FIRST call raises primary_side_effect (simulating the real 503 on the
    strict-schema call) and the SECOND call succeeds with
    fallback_response_text (simulating the schema-less fallback call)."""
    client = MagicMock()
    uploaded = SimpleNamespace(name="files/abc123", state=SimpleNamespace(name=states[0]))
    client.files.upload.return_value = uploaded
    client.files.get.side_effect = [
        SimpleNamespace(name="files/abc123", state=SimpleNamespace(name=s)) for s in states[1:]
    ]
    client.models.generate_content.side_effect = [
        primary_side_effect,
        SimpleNamespace(text=fallback_response_text),
    ]
    return client


@patch("qualume.annotate.gemini_watcher.genai.Client")
@patch("qualume.annotate.gemini_watcher.time.sleep")
def test_annotate_video_falls_back_to_second_model_on_503(mock_sleep, client_cls, tmp_path, monkeypatch):
    """The real fix: a 503 on the primary model (GEMINI_MODEL) + strict
    schema must trigger a retry against GEMINI_MODEL_FALLBACK WITHOUT a
    strict response_schema (confirmed live: schema is what triggers the
    503 on the fallback model specifically) — not propagate the error and
    fail the whole item, which is what would happen without this fix and
    is the real throughput problem (primary model's 500 RPD cap starves
    the ~10,400-video backlog)."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    video_path = tmp_path / "7123.mp4"
    video_path.write_bytes(b"fake video bytes")

    good_dict = _good_v6_dict()
    client = _mock_client_with_fallback(
        primary_side_effect=_server_error(503),
        fallback_response_text=json.dumps(good_dict),
    )
    client_cls.return_value = client

    result = annotate_video(str(video_path), 10.0)

    assert client.models.generate_content.call_count == 2
    first_call, second_call = client.models.generate_content.call_args_list

    # First (failed) call used the primary model + strict schema.
    assert first_call.kwargs["model"] == GEMINI_MODEL
    assert first_call.kwargs["config"].response_schema is not None

    # Second (fallback) call used the SECOND model, JSON mime type but
    # NO strict response_schema object — confirmed live this is what
    # avoids the 503 on this model for video input.
    assert second_call.kwargs["model"] == GEMINI_MODEL_FALLBACK
    assert second_call.kwargs["config"].response_mime_type == "application/json"
    assert second_call.kwargs["config"].response_schema is None

    # Real annotation data still comes back usable, same as the
    # primary-model success path.
    assert result.annotation is not None
    assert result.annotation.hook_type == "question"

    # Real provenance fix: result must report WHICH model actually
    # produced this annotation (the fallback), not silently claim the
    # primary model did it — annotate_and_store persists this into
    # external_creative_dna.analysis_model, and a wrong value there
    # would corrupt provenance on every fallback-served row.
    assert result.model_used == GEMINI_MODEL_FALLBACK


@patch("qualume.annotate.gemini_watcher.genai.Client")
@patch("qualume.annotate.gemini_watcher.time.sleep")
def test_annotate_video_fallback_handles_schema_mismatch_via_existing_degraded_path(
    mock_sleep, client_cls, tmp_path, monkeypatch
):
    """Real case observed live 2026-11-03: the fallback model's JSON
    sometimes has a field value outside this project's strict enum
    (e.g. tone_category='excited', not one of the 10 allowed values).
    This must NOT crash the item — it should land on the SAME existing
    degraded path (raw_response + validation_error stored, best-effort
    fields extracted) that a strict-schema validation failure already
    uses, not a new/different failure mode."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    video_path = tmp_path / "7123.mp4"
    video_path.write_bytes(b"fake video bytes")

    bad_enum_dict = _good_v6_dict(tone_category="excited")  # not a real V6 enum value
    client = _mock_client_with_fallback(
        primary_side_effect=_server_error(503),
        fallback_response_text=json.dumps(bad_enum_dict),
    )
    client_cls.return_value = client

    result = annotate_video(str(video_path), 10.0)

    assert result.annotation is None  # failed strict validation, as expected
    assert result.validation_error is not None
    assert "tone_category" in result.validation_error
    assert result.raw_response == bad_enum_dict  # nothing silently discarded
    # Provenance must still be correct even on the degraded path — the
    # fallback model produced this data regardless of whether it fully
    # validated.
    assert result.model_used == GEMINI_MODEL_FALLBACK


@patch("qualume.annotate.gemini_watcher.genai.Client")
@patch("qualume.annotate.gemini_watcher.time.sleep")
def test_annotate_video_reports_primary_model_when_no_fallback_needed(mock_sleep, client_cls, tmp_path, monkeypatch):
    """Real provenance check for the common case: when the primary model
    succeeds on the first try (no 503, no fallback), model_used must be
    GEMINI_MODEL, not GEMINI_MODEL_FALLBACK or anything stale."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    video_path = tmp_path / "7123.mp4"
    video_path.write_bytes(b"fake video bytes")

    client = MagicMock()
    uploaded = SimpleNamespace(name="files/abc123", state=SimpleNamespace(name="ACTIVE"))
    client.files.upload.return_value = uploaded
    client.models.generate_content.return_value = SimpleNamespace(text=json.dumps(_good_v6_dict()))
    client_cls.return_value = client

    result = annotate_video(str(video_path), 10.0)

    assert client.models.generate_content.call_count == 1  # no fallback attempted
    assert result.model_used == GEMINI_MODEL


@patch("qualume.annotate.gemini_watcher.genai.Client")
@patch("qualume.annotate.gemini_watcher.time.sleep")
def test_annotate_video_propagates_a_non_503_error_without_falling_back(mock_sleep, client_cls, tmp_path, monkeypatch):
    """Real discipline: only a 503 (the specific, confirmed-live
    high-demand/video-input error) should trigger the fallback-model
    retry. A different error (e.g. a genuine 400 bad-request) must NOT
    silently retry against a second model and mask a real bug — it
    should propagate, same as before this fix existed."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    video_path = tmp_path / "7123.mp4"
    video_path.write_bytes(b"fake video bytes")

    client = MagicMock()
    uploaded = SimpleNamespace(name="files/abc123", state=SimpleNamespace(name="ACTIVE"))
    client.files.upload.return_value = uploaded
    client.models.generate_content.side_effect = errors.ClientError(
        400, {"error": {"code": 400, "message": "bad request", "status": "INVALID_ARGUMENT"}}
    )
    client_cls.return_value = client

    try:
        annotate_video(str(video_path), 10.0)
        assert False, "expected the 400 ClientError to propagate"
    except errors.ClientError:
        pass

    # No fallback attempt — only the one (failed) call was made.
    assert client.models.generate_content.call_count == 1
