import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from qualume.models.contracts import (
    ExternalVideoAnnotation,
    ExternalVideoAnnotationV2,
    ExternalVideoAnnotationV3,
    ExternalVideoAnnotationV4,
    ExternalVideoAnnotationV5,
    MAX_SHOT_SECONDS,
    MAX_TRANSCRIPT_SEGMENT_SECONDS,
    ScriptBeat,
    ShotBreakdown,
    ShotBreakdownV4,
    TranscriptSegment,
)
from qualume.annotate import gemini_watcher
from qualume.annotate.gemini_watcher import (
    ANALYSIS_PROMPT_VERSION,
    ANNOTATION_PROMPT,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    MAX_COVERAGE_GAP_SECONDS,
    _check_coverage,
    annotate_video,
    annotate_and_store,
    derive_hook_fields,
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


def _good_v3_dict(**overrides):
    # v3's REAL shape (color_palette, free prose) — must stay this way so the
    # real stored v3 row and ExternalVideoAnnotationV3 tests reflect what was
    # actually asked of Gemini at the time, not v4's later shot fields.
    data = _good_v2_dict(
        shot_breakdown=[
            {
                "start_sec": 0.0, "end_sec": 2.5, "shot_type": "close_up",
                "camera_movement": "static", "setting": "kitchen",
                "color_palette": "warm pinks and whites", "lighting": "bright_natural",
                "subject_action": "opening a jar",
            },
            {
                "start_sec": 2.5, "end_sec": 14.0, "shot_type": "medium_shot",
                "camera_movement": "handheld", "setting": "kitchen",
                "color_palette": "warm pinks and whites", "lighting": "bright_natural",
                "subject_action": "demonstrating the product",
            },
        ],
        full_transcript=[
            {"start_sec": 0.0, "text": "Why does nobody talk about this?"},
            {"start_sec": 3.0, "text": "Let me show you exactly how it works."},
        ],
    )
    data.update(overrides)
    return data


def _good_v4_dict(**overrides):
    # v4's shape: same as v2's base fields, but shot_breakdown uses the NEW
    # ShotBreakdownV4 fields (setting_category/color_temperature/
    # dominant_colors instead of v3's free-prose color_palette).
    data = _good_v2_dict(
        shot_breakdown=[
            _shot(0.0, 2.5, subject_action="opening a jar"),
            _shot(2.5, 14.0, camera_movement="handheld",
                  subject_action="demonstrating the product"),
        ],
        full_transcript=[
            {"start_sec": 0.0, "text": "Why does nobody talk about this?"},
            {"start_sec": 3.0, "text": "Let me show you exactly how it works."},
        ],
    )
    data.update(overrides)
    return data


V5_VIDEO_DURATION = 17  # matches _good_v5_dict's timeline end


def _good_v5_dict(**overrides):
    # v5's shape is v4's, but finely segmented enough to pass v5's
    # granularity ceilings (v4's fixture has an 11.5s shot and 14s of
    # trailing transcript, both of which v5 deliberately rejects).
    data = _good_v2_dict(
        shot_breakdown=[
            _shot(0.0, 2.5, subject_action="opening a jar"),
            _shot(2.5, 9.0, camera_movement="handheld", subject_action="demonstrating the product"),
            _shot(9.0, 14.0, shot_type="medium_shot", subject_action="showing the result"),
            _shot(14.0, 17.0, subject_action="pointing at follow button"),
        ],
        full_transcript=[
            {"start_sec": 0.0, "text": "Why does nobody talk about this?"},
            {"start_sec": 3.0, "text": "Let me show you exactly how it works."},
            {"start_sec": 7.5, "text": "You just twist and press."},
            {"start_sec": 12.0, "text": "Follow for part two."},
        ],
    )
    data.update(overrides)
    return data


def _shot(start, end, **overrides):
    data = {
        "start_sec": start, "end_sec": end, "shot_type": "close_up",
        "camera_movement": "static", "setting": "kitchen",
        "setting_category": "home_indoor", "color_temperature": "warm",
        "dominant_colors": ["pink", "white"], "lighting": "bright_natural",
        "subject_action": "opening a jar",
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


def test_v3_contract_extends_v2_and_accepts_good_input():
    assert issubclass(ExternalVideoAnnotationV3, ExternalVideoAnnotationV2)
    ann = ExternalVideoAnnotationV3(**_good_v3_dict())
    assert ann.hook_type == "question"  # v1/v2 fields still present
    assert len(ann.shot_breakdown) == 2
    assert ann.shot_breakdown[0].shot_type == "close_up"
    assert ann.shot_breakdown[1].setting == "kitchen"
    assert len(ann.full_transcript) == 2
    assert ann.full_transcript[0].text == "Why does nobody talk about this?"


def test_v3_contract_rejects_shot_ending_before_it_starts():
    with pytest.raises(ValidationError):
        ExternalVideoAnnotationV3(**_good_v3_dict(
            shot_breakdown=[{
                "start_sec": 5.0, "end_sec": 1.0, "shot_type": "close_up",
                "camera_movement": "static", "setting": "kitchen",
                "color_palette": "warm", "lighting": "bright_natural",
                "subject_action": "talking",
            }]))


def test_v3_contract_rejects_out_of_order_shots_or_transcript():
    with pytest.raises(ValidationError):
        ExternalVideoAnnotationV3(**_good_v3_dict(
            full_transcript=[
                {"start_sec": 5.0, "text": "second"},
                {"start_sec": 1.0, "text": "first, but out of order"},
            ]))


def test_v3_contract_allows_empty_shot_and_transcript_lists():
    """A silent video with one static shot still has valid (not empty) data
    in practice, but the schema shouldn't force a lie if either is genuinely
    absent."""
    ann = ExternalVideoAnnotationV3(**_good_v3_dict(full_transcript=[]))
    assert ann.full_transcript == []


# --- Contract v4 -------------------------------------------------------------


def test_v4_contract_extends_v3_and_accepts_good_input():
    assert issubclass(ExternalVideoAnnotationV4, ExternalVideoAnnotationV3)
    ann = ExternalVideoAnnotationV4(**_good_v4_dict())
    shot = ann.shot_breakdown[0]
    assert shot.color_temperature == "warm"
    assert shot.dominant_colors == ["pink", "white"]
    assert shot.setting_category == "home_indoor"
    assert shot.setting == "kitchen"  # free-text detail kept alongside the category


def test_color_palette_no_longer_exists_in_v4():
    """Replaced by color_temperature + dominant_colors; keeping it alongside
    would be the same redundancy problem in a new form. v3's own
    ShotBreakdown is untouched (it has color_palette, correctly — that's
    literally what was asked of Gemini for the real stored v3 row)."""
    assert "color_palette" not in ShotBreakdownV4.model_fields
    assert "color_palette" in ShotBreakdown.model_fields  # v3's real shape, untouched
    schema = ExternalVideoAnnotationV4.model_json_schema()
    for model_schema in [schema, *schema["$defs"].values()]:
        assert "color_palette" not in model_schema.get("properties", {})


@pytest.mark.parametrize("temp", ["warm", "cool", "neutral", "mixed"])
def test_shot_accepts_every_color_temperature(temp):
    assert ShotBreakdownV4(**_shot(0, 1, color_temperature=temp)).color_temperature == temp


@pytest.mark.parametrize("category", [
    "home_indoor", "home_outdoor", "studio", "public_indoor", "public_outdoor", "vehicle", "other",
])
def test_shot_accepts_every_setting_category(category):
    assert ShotBreakdownV4(**_shot(0, 1, setting_category=category)).setting_category == category


def test_shot_accepts_the_full_basic_color_vocabulary():
    colors = ["red", "orange", "yellow", "green", "blue", "purple",
              "pink", "brown", "black", "white", "gray", "beige"]
    assert ShotBreakdownV4(**_shot(0, 1, dominant_colors=colors)).dominant_colors == colors


@pytest.mark.parametrize("overrides", [
    {"color_temperature": "hot"},
    {"color_temperature": "bright warm tones"},  # the old free-prose style
    {"dominant_colors": ["pink", "fuchsia"]},  # outside the fixed vocabulary
    {"dominant_colors": ["deep reds and bright whites"]},
    {"dominant_colors": ["Pink"]},  # vocabulary is exact lowercase
    {"dominant_colors": []},  # every visible shot has at least one color
    {"setting_category": "kitchen"},  # that's `setting`, not the coarse category
    {"setting_category": "outdoor"},
])
def test_shot_rejects_values_outside_the_fixed_vocabularies(overrides):
    with pytest.raises(ValidationError):
        ShotBreakdownV4(**_shot(0, 1, **overrides))


@pytest.mark.parametrize("missing", ["color_temperature", "dominant_colors", "setting_category"])
def test_shot_requires_the_new_queryable_fields(missing):
    data = _shot(0, 1)
    del data[missing]
    with pytest.raises(ValidationError):
        ShotBreakdownV4(**data)


def test_hook_type_and_other_learned_labels_stay_free_strings():
    """Deliberately NOT enums: features are learned over time (see
    creative_dna_feature_registry), so an unanticipated-but-valid label must
    not be rejected."""
    ann = ExternalVideoAnnotationV4(**_good_v4_dict(
        hook_type="never_seen_before_hook", format_tag="new_format", cta_type="new_cta",
        shot_breakdown=[_shot(0, 2.5, shot_type="dutch_angle", camera_movement="whip_pan",
                              lighting="neon_rgb")],
    ))
    assert ann.hook_type == "never_seen_before_hook"
    assert ann.shot_breakdown[0].shot_type == "dutch_angle"


def test_prompt_describes_v4_shot_fields_and_their_vocabularies():
    for word in ["color_temperature", "dominant_colors", "setting_category",
                 "warm", "cool", "neutral", "mixed", "beige", "gray",
                 "home_indoor", "public_outdoor", "vehicle"]:
        assert word in ANNOTATION_PROMPT, f"ANNOTATION_PROMPT never mentions {word}"


# --- Contract v5 -------------------------------------------------------------


def _real_bug_32s_dict():
    """The real v4 run: shots/beats correctly cover a 32s video, but
    full_transcript collapsed into ONE segment at 0.0 covering everything —
    so derive_hook_fields attributed the whole script to the hook."""
    return _good_v5_dict(
        script_structure_timeline=[
            {"beat": "hook", "start_sec": 0.0, "end_sec": 3.0},
            {"beat": "demo", "start_sec": 3.0, "end_sec": 26.0},
            {"beat": "cta", "start_sec": 26.0, "end_sec": 32.0},
        ],
        shot_breakdown=[_shot(float(s), float(s + 4)) for s in range(0, 32, 4)],
        full_transcript=[{"start_sec": 0.0, "text": "Sentence one. Sentence two. ... Sentence eight."}],
    )


def test_v5_rejects_one_transcript_segment_spanning_the_whole_video():
    """Exact reproduction of the real v4 bug."""
    raw = _real_bug_32s_dict()
    ExternalVideoAnnotationV4(**raw)  # v4 accepted it — that's the bug
    with pytest.raises(ValidationError, match="full_transcript"):
        ExternalVideoAnnotationV5(**raw)


def test_v5_rejects_an_overlong_transcript_segment_in_the_middle():
    raw = _good_v5_dict(full_transcript=[
        {"start_sec": 0.0, "text": "a"},
        {"start_sec": 2.0, "text": "b — then nothing until 11.5"},  # spans 9.5s
        {"start_sec": 11.5, "text": "c"},
    ])
    with pytest.raises(ValidationError, match="full_transcript"):
        ExternalVideoAnnotationV5(**raw)


def test_v5_accepts_transcript_segments_all_under_the_ceiling():
    ann = ExternalVideoAnnotationV5(**_good_v5_dict())
    assert len(ann.full_transcript) == 4


def test_v5_transcript_ceiling_is_inclusive():
    # last segment at 9.0, annotation ends at 17.0 -> exactly the ceiling
    assert MAX_TRANSCRIPT_SEGMENT_SECONDS == 8.0
    raw = _good_v5_dict(full_transcript=[
        {"start_sec": 0.0, "text": "a"}, {"start_sec": 5.0, "text": "b"},
        {"start_sec": 9.0, "text": "c"},
    ])
    ExternalVideoAnnotationV5(**raw)


def test_v5_allows_empty_transcript():
    """A silent video legitimately has no speech."""
    assert ExternalVideoAnnotationV5(**_good_v5_dict(full_transcript=[])).full_transcript == []


def test_v5_rejects_a_single_shot_that_is_too_long():
    raw = _good_v5_dict(shot_breakdown=[_shot(0.0, 2.5), _shot(2.5, 17.0)])  # 14.5s shot
    with pytest.raises(ValidationError, match="shot_breakdown"):
        ExternalVideoAnnotationV5(**raw)


def test_v5_shot_ceiling_is_inclusive():
    assert MAX_SHOT_SECONDS == 10.0
    ExternalVideoAnnotationV5(**_good_v5_dict(shot_breakdown=[_shot(0.0, 10.0), _shot(10.0, 17.0)]))


def test_v5_extends_v4_without_changing_it():
    """v4 must still validate the already-stored real v4 row's shape —
    v5's stricter rules live only on the new class."""
    assert issubclass(ExternalVideoAnnotationV5, ExternalVideoAnnotationV4)
    ExternalVideoAnnotationV4(**_good_v4_dict())
    with pytest.raises(ValidationError):
        ExternalVideoAnnotationV5(**_good_v4_dict())  # 11.5s shot


def test_prompt_demands_fine_segmentation():
    for phrase in ["sentence", "camera cut", "do not merge", "8 seconds", "10 seconds"]:
        assert phrase in ANNOTATION_PROMPT.lower(), f"ANNOTATION_PROMPT never mentions {phrase!r}"


# --- coverage check ------------------------------------------------------------


def test_coverage_ok_for_a_fully_covered_video():
    assert _check_coverage(ExternalVideoAnnotationV5(**_good_v5_dict()), V5_VIDEO_DURATION) == []


def test_coverage_flags_shots_that_stop_early():
    """Shots covering only 0-10s of a 32s video validate fine (each shot is
    short) but are clearly incomplete."""
    ann = ExternalVideoAnnotationV5(**_good_v5_dict(
        script_structure_timeline=[{"beat": "hook", "start_sec": 0.0, "end_sec": 10.0}],
        shot_breakdown=[_shot(0.0, 5.0), _shot(5.0, 10.0)],
        full_transcript=[{"start_sec": 0.0, "text": "a"}, {"start_sec": 4.0, "text": "b"}],
    ))
    warnings = _check_coverage(ann, 32)
    assert any("shot_breakdown" in w and "10.0" in w and "32" in w for w in warnings), warnings


def test_coverage_flags_a_large_gap_between_shots():
    ann = ExternalVideoAnnotationV5(**_good_v5_dict(
        shot_breakdown=[_shot(0.0, 5.0), _shot(5.0 + MAX_COVERAGE_GAP_SECONDS + 1, 17.0)]))
    warnings = _check_coverage(ann, V5_VIDEO_DURATION)
    assert any("gap" in w for w in warnings), warnings


def test_coverage_tolerates_small_gaps():
    ann = ExternalVideoAnnotationV5(**_good_v5_dict(
        shot_breakdown=[_shot(0.5, 8.0), _shot(9.0, 16.0)]))
    assert _check_coverage(ann, V5_VIDEO_DURATION) == []


def test_coverage_flags_empty_shot_breakdown():
    ann = ExternalVideoAnnotationV5(**_good_v5_dict(shot_breakdown=[]))
    assert any("shot_breakdown" in w for w in _check_coverage(ann, V5_VIDEO_DURATION))


def test_coverage_flags_shots_running_past_the_real_video_end():
    ann = ExternalVideoAnnotationV5(**_good_v5_dict())
    assert any("past" in w for w in _check_coverage(ann, 10))


def test_coverage_flags_transcript_ending_far_before_the_video_does():
    """The contract can only measure the last segment against the
    annotation's OWN end (here 12s, so 4.0 -> 12.0 passes at exactly 8s);
    only the real duration (13s) reveals the 9s of unaccounted tail."""
    ann = ExternalVideoAnnotationV5(**_good_v5_dict(
        script_structure_timeline=[{"beat": "hook", "start_sec": 0.0, "end_sec": 12.0}],
        shot_breakdown=[_shot(0.0, 8.0), _shot(8.0, 12.0)],
        full_transcript=[{"start_sec": 0.0, "text": "a"}, {"start_sec": 4.0, "text": "b"}],
    ))
    warnings = _check_coverage(ann, 13)
    assert len(warnings) == 1 and "full_transcript" in warnings[0], warnings


def test_coverage_does_not_flag_an_empty_transcript():
    ann = ExternalVideoAnnotationV5(**_good_v5_dict(full_transcript=[]))
    assert _check_coverage(ann, V5_VIDEO_DURATION) == []


def test_coverage_flags_unknown_duration_instead_of_silently_passing():
    warnings = _check_coverage(ExternalVideoAnnotationV5(**_good_v5_dict()), None)
    assert len(warnings) == 1 and "duration" in warnings[0]


# --- derive_hook_fields --------------------------------------------------------


def _beats(*triples):
    return [ScriptBeat(beat=b, start_sec=s, end_sec=e) for b, s, e in triples]


def _transcript(*pairs):
    return [TranscriptSegment(start_sec=s, text=t) for s, t in pairs]


def _shots(*shots):
    return [ShotBreakdownV4(**s) for s in shots]


def test_derive_hook_text_joins_transcript_segments_inside_the_hook_beat():
    """The motivating real case: Gemini's own hook_text dropped 'I found the
    perfect one' while full_transcript captured it. Deriving from the
    transcript makes them agree by construction."""
    hook_text, _ = derive_hook_fields(
        _beats(("hook", 0.0, 4.0), ("demo", 4.0, 12.0)),
        _transcript((0.0, "I found the perfect one."), (2.1, "You need this."),
                    (4.0, "Here's how it works."), (8.0, "So easy.")),
        _shots(_shot(0.0, 12.0)),
    )
    assert hook_text == "I found the perfect one. You need this."


def test_derive_hook_window_end_is_exclusive_and_start_is_inclusive():
    hook_text, _ = derive_hook_fields(
        _beats(("intro", 0.0, 1.0), ("hook", 1.0, 3.0), ("cta", 3.0, 5.0)),
        _transcript((0.5, "before"), (1.0, "at start"), (2.9, "inside"), (3.0, "at end")),
        _shots(_shot(0.0, 5.0)),
    )
    assert hook_text == "at start inside"


def test_derive_hook_uses_the_beat_named_hook_even_when_not_first():
    hook_text, _ = derive_hook_fields(
        _beats(("cold_open", 0.0, 2.0), ("hook", 2.0, 5.0)),
        _transcript((0.0, "ambient"), (2.5, "the actual hook")),
        _shots(_shot(0.0, 5.0)),
    )
    assert hook_text == "the actual hook"


def test_derive_hook_falls_back_to_first_three_seconds_without_a_hook_beat():
    hook_text, visual = derive_hook_fields(
        _beats(("problem", 0.0, 6.0), ("cta", 6.0, 9.0)),
        _transcript((0.0, "one"), (2.9, "two"), (3.0, "three")),
        _shots(_shot(0.0, 3.0, shot_type="wide_shot", setting="bedroom",
                     subject_action="unboxing a package"),
               _shot(3.0, 9.0, shot_type="close_up")),
    )
    assert hook_text == "one two"
    assert visual == "A wide shot in bedroom: unboxing a package."


def test_derive_hook_falls_back_when_timeline_is_empty():
    hook_text, _ = derive_hook_fields([], _transcript((1.0, "hi")), _shots(_shot(0.0, 3.0)))
    assert hook_text == "hi"


def test_derive_hook_text_empty_when_transcript_empty():
    hook_text, visual = derive_hook_fields(
        _beats(("hook", 0.0, 3.0)), [], _shots(_shot(0.0, 3.0)))
    assert hook_text == ""
    assert visual  # visuals still derived for a silent video


def test_derive_hook_text_empty_when_no_segment_in_window():
    hook_text, _ = derive_hook_fields(
        _beats(("hook", 0.0, 2.0)), _transcript((5.0, "later")), _shots(_shot(0.0, 6.0)))
    assert hook_text == ""


def test_derive_visual_describes_the_single_overlapping_shot():
    _, visual = derive_hook_fields(
        _beats(("hook", 0.0, 2.5)), [],
        _shots(_shot(0.0, 2.5, shot_type="extreme_close_up", setting="bathroom",
                     subject_action="applying serum"),
               _shot(2.5, 8.0, shot_type="wide_shot")),
    )
    assert visual == "An extreme close up shot in bathroom: applying serum."


def test_derive_visual_notes_the_transition_when_multiple_shots_overlap():
    _, visual = derive_hook_fields(
        _beats(("hook", 1.0, 4.0)), [],
        _shots(_shot(0.0, 2.0, shot_type="close_up", setting="kitchen",
                     subject_action="opening a jar"),
               _shot(2.0, 6.0, shot_type="medium_shot", setting="kitchen",
                     subject_action="holding the jar up to camera"),
               _shot(6.0, 9.0, shot_type="wide_shot")),
    )
    assert visual == ("A close up shot in kitchen: opening a jar. "
                      "Then cuts to a medium shot: holding the jar up to camera.")


def test_derive_visual_counts_extra_shots_when_more_than_two_overlap():
    _, visual = derive_hook_fields(
        _beats(("hook", 0.0, 3.0)), [],
        _shots(_shot(0.0, 1.0, subject_action="a"), _shot(1.0, 2.0, shot_type="pov",
                                                          subject_action="b"),
               _shot(2.0, 3.0, subject_action="c")),
    )
    assert visual == "A close up shot in kitchen: a. Then cuts through 2 more shots, starting with a pov shot: b."


def test_derive_visual_empty_when_no_shot_overlaps():
    _, visual = derive_hook_fields(_beats(("hook", 0.0, 3.0)), [], [])
    assert visual == ""


def test_prompt_mentions_shot_breakdown_and_full_transcript():
    assert "shot_breakdown" in ANNOTATION_PROMPT
    assert "full_transcript" in ANNOTATION_PROMPT
    assert "COMPLETE transcript" in ANNOTATION_PROMPT  # not just the hook


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
            annotate_video(fake_video, V5_VIDEO_DURATION)
    client_cls.assert_not_called()


def test_annotate_video_uploads_waits_for_active_and_constrains_output(gemini_key, fake_video):
    client = _mock_client(json.dumps(_good_v5_dict()))
    with patch("qualume.annotate.gemini_watcher.genai.Client", return_value=client) as client_cls, \
         patch("qualume.annotate.gemini_watcher.time.sleep"):
        result = annotate_video(fake_video, V5_VIDEO_DURATION)

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
    assert config.response_schema is ExternalVideoAnnotationV5
    assert config.temperature == GEMINI_TEMPERATURE == None  # noqa: E711 — real sentinel, not falsy 0

    assert isinstance(result.annotation, ExternalVideoAnnotationV5)
    assert result.annotation == ExternalVideoAnnotationV5(**_good_v5_dict())
    assert result.raw_response == _good_v5_dict()
    assert result.coverage_warning is None
    client.files.delete.assert_called_once_with(name="files/abc123")


def test_annotate_video_reports_incomplete_coverage_without_failing(gemini_key, fake_video):
    """Shots that stop at 10s of a 32s video: schema-valid, so annotation is
    kept, but the coverage gap is surfaced rather than silently accepted."""
    raw = _good_v5_dict(
        script_structure_timeline=[{"beat": "hook", "start_sec": 0.0, "end_sec": 10.0}],
        shot_breakdown=[_shot(0.0, 5.0), _shot(5.0, 10.0)],
        full_transcript=[{"start_sec": 0.0, "text": "a"}, {"start_sec": 4.0, "text": "b"}],
    )
    client = _mock_client(json.dumps(raw))
    with patch("qualume.annotate.gemini_watcher.genai.Client", return_value=client), \
         patch("qualume.annotate.gemini_watcher.time.sleep"):
        result = annotate_video(fake_video, 32)
    assert isinstance(result.annotation, ExternalVideoAnnotationV5)
    assert result.validation_error is None
    assert "shot_breakdown" in result.coverage_warning


def test_annotate_video_routes_the_real_collapsed_transcript_bug_to_the_degraded_path(
        gemini_key, fake_video):
    """The granularity check is a real ValidationError, so it goes through
    the existing graceful path: annotation None, raw response kept."""
    raw = _real_bug_32s_dict()
    client = _mock_client(json.dumps(raw))
    with patch("qualume.annotate.gemini_watcher.genai.Client", return_value=client), \
         patch("qualume.annotate.gemini_watcher.time.sleep"):
        result = annotate_video(fake_video, 32)
    assert result.annotation is None
    assert "full_transcript" in result.validation_error
    assert result.raw_response == raw
    assert result.coverage_warning is None  # nothing validated to check coverage of


def test_annotate_video_keeps_full_raw_response_including_unmapped_keys(gemini_key, fake_video):
    """Nothing Gemini returns is discarded: keys with no typed field still
    land in raw_response."""
    raw = _good_v5_dict(unmapped_extra={"anything": [1, 2, 3]})
    client = _mock_client(json.dumps(raw))
    with patch("qualume.annotate.gemini_watcher.genai.Client", return_value=client), \
         patch("qualume.annotate.gemini_watcher.time.sleep"):
        result = annotate_video(fake_video, V5_VIDEO_DURATION)
    assert result.raw_response == raw
    assert result.raw_response["unmapped_extra"] == {"anything": [1, 2, 3]}
    assert "unmapped_extra" not in result.annotation.model_dump()


def test_annotate_video_raises_when_raw_response_is_not_a_json_object(gemini_key, fake_video):
    client = _mock_client(json.dumps([_good_v2_dict()]))
    with patch("qualume.annotate.gemini_watcher.genai.Client", return_value=client), \
         patch("qualume.annotate.gemini_watcher.time.sleep"):
        with pytest.raises(RuntimeError, match="JSON object"):
            annotate_video(fake_video, V5_VIDEO_DURATION)
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
        result = annotate_video(fake_video, V5_VIDEO_DURATION)
    assert result.annotation is None
    assert result.validation_error is not None
    assert result.raw_response == bad_dict  # nothing lost, even though validation failed
    client.files.delete.assert_called_once_with(name="files/abc123")


def test_annotate_video_raises_when_file_processing_fails(gemini_key, fake_video):
    client = _mock_client("{}", states=("PROCESSING", "FAILED"))
    with patch("qualume.annotate.gemini_watcher.genai.Client", return_value=client), \
         patch("qualume.annotate.gemini_watcher.time.sleep"):
        with pytest.raises(RuntimeError, match="FAILED"):
            annotate_video(fake_video, V5_VIDEO_DURATION)
    client.models.generate_content.assert_not_called()


def test_annotate_video_raises_when_local_file_missing(gemini_key, tmp_path):
    with patch("qualume.annotate.gemini_watcher.genai.Client") as client_cls:
        with pytest.raises(FileNotFoundError):
            annotate_video(str(tmp_path / "does_not_exist.mp4"), V5_VIDEO_DURATION)
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


def _insert_item(account_id: str, local_video_path, duration_seconds=None):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            vid = uuid.uuid4().hex[:12]
            cur.execute(
                """INSERT INTO external_content_items
                       (watched_account_id, platform, platform_video_id, url, local_video_path,
                        duration_seconds)
                   VALUES (%s, 'tiktok', %s, %s, %s, %s) RETURNING id""",
                (account_id, f"test_{vid}", f"https://www.tiktok.com/@test/video/{vid}",
                 local_video_path, duration_seconds),
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
        item_id = _insert_item(account_id, "/fake/path/7123.mp4", duration_seconds=17)
        # Gemini's own hook answers deliberately disagree with its transcript/
        # shots — the stored values must be the derived ones, not these.
        raw = _good_v5_dict(social_proof_used=True, unmapped_extra="kept",
                            hook_text="Gemini's independently drifted hook",
                            visual_hook_description="Gemini's independent visual read")
        result = gemini_watcher.AnnotationResult(
            annotation=ExternalVideoAnnotationV5(**raw), raw_response=raw,
            coverage_warning="shot_breakdown stops at 17.0s of a 99s video")

        with patch("qualume.annotate.gemini_watcher.annotate_video", return_value=result) as mock_annotate:
            dna_id = annotate_and_store(item_id)
        # the real duration from external_content_items is threaded through
        mock_annotate.assert_called_once_with("/fake/path/7123.mp4", 17)

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
                              pacing_qualitative, music_mood_v2, shot_breakdown, full_transcript,
                              raw_gemini_response, coverage_warning
                       FROM external_creative_dna WHERE id = %s""",
                    (dna_id,),
                )
                row = cur.fetchone()
        finally:
            conn.close()

        assert row is not None
        (eci_id, hook_type, hook_text, visual, fmt, tags, cta, caption, proof, energy, tone,
         voice, pacing, music, extended, model, prompt_version, analyzed_at,
         timeline, on_screen, emotions, pacing_q, music_v2, shots, transcript, raw_response,
         coverage_warning) = row
        assert str(eci_id) == item_id
        assert hook_type == "question"
        # Derived from full_transcript / shot_breakdown inside the 'hook' beat (0-2.5s)
        assert hook_text == "Why does nobody talk about this?"
        assert visual == "A close up shot in kitchen: opening a jar."
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
        assert prompt_version == ANALYSIS_PROMPT_VERSION == "v5"
        assert analyzed_at is not None
        assert timeline == raw["script_structure_timeline"]
        assert on_screen == raw["on_screen_text"]
        assert emotions == raw["emotion_timeline"]
        assert pacing_q == "medium"
        assert music_v2 == ["upbeat", "lofi"]
        assert shots == raw["shot_breakdown"]
        assert transcript == raw["full_transcript"]
        assert raw_response == raw  # complete, including unmapped_extra and Gemini's own hook answers
        assert coverage_warning == "shot_breakdown stops at 17.0s of a 99s video"  # queryable, not just logged
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
        bad_raw = _good_v5_dict(energy_level=9,  # out of 1-5, fails strict validation
                                hook_text="Gemini's independently drifted hook")
        result = gemini_watcher.AnnotationResult(
            annotation=None, raw_response=bad_raw, validation_error="energy_level out of range"
        )

        with patch("qualume.annotate.gemini_watcher.annotate_video", return_value=result):
            dna_id = annotate_and_store(item_id)

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT hook_type, hook_text, visual_hook_description,
                              script_structure_timeline, raw_gemini_response, validation_error,
                              coverage_warning
                       FROM external_creative_dna WHERE id = %s""",
                    (dna_id,),
                )
                (hook_type, hook_text, visual, timeline, raw_response, val_error,
                 coverage_warning) = cur.fetchone()
        finally:
            conn.close()

        assert hook_type == "question"  # best-effort scalar still filled
        # Still derived (best-effort) from the raw transcript/shots, never
        # Gemini's independent hook_text answer.
        assert hook_text == "Why does nobody talk about this?"
        assert visual == "A close up shot in kitchen: opening a jar."
        assert timeline is None  # nested/typed field NOT guessed when validation failed
        assert raw_response == bad_raw  # nothing lost
        assert val_error == "energy_level out of range"
        assert coverage_warning is None  # nothing validated to check; validation_error says why
    finally:
        _cleanup_account(handle)


def test_annotate_and_store_degraded_path_leaves_hook_fields_null_when_sources_unusable():
    """If the raw transcript/shots are missing or malformed we can't derive,
    and falling back to Gemini's independent hook answers would reintroduce
    the second source of truth — so the derived columns stay NULL (the raw
    answers are still in raw_gemini_response)."""
    handle = f"test_annot_degraded_null_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    try:
        account_id = add_watched_account("tiktok", handle)
        item_id = _insert_item(account_id, "/fake/path/8888.mp4")
        bad_raw = _good_v5_dict(shot_breakdown=[{"start_sec": 0.0, "color_palette": "warm"}])
        del bad_raw["full_transcript"]
        result = gemini_watcher.AnnotationResult(
            annotation=None, raw_response=bad_raw, validation_error="bad shots")

        with patch("qualume.annotate.gemini_watcher.annotate_video", return_value=result):
            dna_id = annotate_and_store(item_id)

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT hook_type, hook_text, visual_hook_description, raw_gemini_response
                       FROM external_creative_dna WHERE id = %s""",
                    (dna_id,),
                )
                hook_type, hook_text, visual, raw_response = cur.fetchone()
        finally:
            conn.close()

        assert hook_type == "question"
        assert hook_text is None
        assert visual is None
        assert raw_response == bad_raw
    finally:
        _cleanup_account(handle)


def test_annotate_and_store_degraded_path_rejects_the_real_bug_transcript():
    """Regression test for the actual bug found running v4 live twice on the
    same video: Gemini collapsed the whole video's speech into ONE
    TranscriptSegment (individually valid — nothing wrong with a single
    segment on its own) instead of 8 properly-segmented ones. Each item
    parses fine, so the raw list passes _parse_raw_list, but it violates the
    SAME granularity ceiling ExternalVideoAnnotationV5 enforces at the
    list level. Before the fix, the degraded path still derived hook_text
    from this oversized segment, since it only checked 'did it parse',
    not 'is it granular enough' — silently reintroducing the exact bug
    v5's validator exists to catch, just via the degraded path instead of
    the happy path."""
    handle = f"test_annot_degraded_bug_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    try:
        account_id = add_watched_account("tiktok", handle)
        item_id = _insert_item(account_id, "/fake/path/7670137194773826847.mp4")
        bad_raw = _good_v5_dict(
            full_transcript=[
                # One segment covering the ENTIRE video — this is exactly
                # what Gemini actually returned on the real re-run.
                {"start_sec": 0.0, "text": (
                    "Okay, if y'all have been looking for a daily immune "
                    "support for your children, I found the perfect one. "
                    "This is from Wish Gardens."
                )},
            ],
            energy_level=9,  # also fails validation, for an unrelated reason —
                              # the point is the transcript check specifically
        )
        result = gemini_watcher.AnnotationResult(
            annotation=None, raw_response=bad_raw, validation_error="too many problems"
        )

        with patch("qualume.annotate.gemini_watcher.annotate_video", return_value=result):
            dna_id = annotate_and_store(item_id)

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT hook_text, raw_gemini_response FROM external_creative_dna WHERE id = %s",
                    (dna_id,),
                )
                hook_text, raw_response = cur.fetchone()
        finally:
            conn.close()

        # The real bug: this used to be the ENTIRE 3-sentence transcript,
        # attributed to the hook. Now it must be NULL — not derived from an
        # under-segmented source, even though every individual item parsed.
        assert hook_text is None
        assert raw_response == bad_raw  # still nothing lost
    finally:
        _cleanup_account(handle)
