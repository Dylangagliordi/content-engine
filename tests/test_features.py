"""Tests for engineered features derived from the rich per-shot/transcript
JSONB data — written FIRST per this project's TDD discipline.

Per the architecture doc's "pacing should be computed, not LLM-guessed"
principle (originally meant for PySceneDetect, which was abandoned in
favor of Gemini providing real per-shot timestamps directly): now that
v5 annotations give exact shot start_sec/end_sec and transcript
start_sec, EXACT cuts-per-minute and other continuous signals can be
computed deterministically from that data, not estimated. This is what
makes the rich shot_breakdown/full_transcript JSONB (currently ignored by
learn.mining's categorical-only approach) actually usable as INPUT
FEATURES for a future prediction model, per the plan's regression/ML
upgrade path.

Pure functions over already-fetched dicts/lists — no DB dependency, fast,
exercises edge cases (empty data, one shot, etc.) directly.
"""
from qualume.learn.features import engineered_features


def _shot(start, end, shot_type="medium_shot", color_temperature="warm", setting="kitchen"):
    return {
        "start_sec": start, "end_sec": end, "shot_type": shot_type,
        "camera_movement": "static", "setting": setting,
        "setting_category": "home_indoor", "color_temperature": color_temperature,
        "dominant_colors": ["white", "beige"], "lighting": "bright_natural",
        "subject_action": "doing something",
    }


def _segment(start, text):
    return {"start_sec": start, "text": text}


def test_cuts_per_min_computed_exactly_from_real_shot_timestamps():
    """Real cuts-per-minute, not an estimate — this is the exact signal
    the plan originally wanted from PySceneDetect, now derivable directly
    from Gemini's own per-shot timestamps instead of a separate tool."""
    shots = [_shot(0, 10), _shot(10, 20), _shot(20, 30)]  # 3 shots, 30s video
    result = engineered_features(shots, [], video_duration_seconds=30)
    assert result["cuts_per_min"] == 6.0  # 3 cuts / 0.5 min


def test_avg_shot_length_sec():
    shots = [_shot(0, 10), _shot(10, 15), _shot(15, 30)]  # lengths: 10, 5, 15
    result = engineered_features(shots, [], video_duration_seconds=30)
    assert result["avg_shot_length_sec"] == 10.0


def test_shot_length_stddev_zero_for_perfectly_even_shots():
    shots = [_shot(0, 10), _shot(10, 20), _shot(20, 30)]  # all 10s
    result = engineered_features(shots, [], video_duration_seconds=30)
    assert result["shot_length_stddev"] == 0.0


def test_shot_length_stddev_nonzero_for_uneven_shots():
    shots = [_shot(0, 1), _shot(1, 2), _shot(2, 30)]  # very uneven
    result = engineered_features(shots, [], video_duration_seconds=30)
    assert result["shot_length_stddev"] > 0


def test_handles_empty_shot_breakdown_without_crashing():
    """A video with no shots annotated (e.g. validation_error row) must
    return a result with null/zero features, not raise."""
    result = engineered_features([], [], video_duration_seconds=30)
    assert result["cuts_per_min"] is None
    assert result["avg_shot_length_sec"] is None
    assert result["shot_length_stddev"] is None


def test_handles_single_shot():
    """One shot (e.g. a real static 60s talking-head take, per v5's real
    corrected understanding that this is legitimate) should not crash on
    stddev (needs >=2 points) and should report 1 cut/min correctly."""
    shots = [_shot(0, 60)]
    result = engineered_features(shots, [], video_duration_seconds=60)
    assert result["cuts_per_min"] == 1.0
    assert result["avg_shot_length_sec"] == 60.0
    assert result["shot_length_stddev"] == 0.0  # one point, no variance


def test_color_temperature_consistency_full_agreement():
    shots = [_shot(0, 10, color_temperature="warm"),
             _shot(10, 20, color_temperature="warm"),
             _shot(20, 30, color_temperature="warm")]
    result = engineered_features(shots, [], video_duration_seconds=30)
    assert result["color_temperature_consistency"] == 1.0


def test_color_temperature_consistency_partial_agreement():
    shots = [_shot(0, 10, color_temperature="warm"),
             _shot(10, 20, color_temperature="warm"),
             _shot(20, 30, color_temperature="cool")]
    result = engineered_features(shots, [], video_duration_seconds=30)
    assert result["color_temperature_consistency"] == 2 / 3


def test_setting_change_count():
    shots = [_shot(0, 10, setting="kitchen"), _shot(10, 20, setting="kitchen"),
             _shot(20, 30, setting="living room"), _shot(30, 40, setting="kitchen")]
    result = engineered_features(shots, [], video_duration_seconds=40)
    assert result["setting_change_count"] == 2  # kitchen->living room, living room->kitchen


def test_shot_type_variety_count():
    shots = [_shot(0, 10, shot_type="close_up"), _shot(10, 20, shot_type="medium_shot"),
             _shot(20, 30, shot_type="close_up")]
    result = engineered_features(shots, [], video_duration_seconds=30)
    assert result["shot_type_variety"] == 2  # close_up, medium_shot


def test_speech_coverage_ratio_full_coverage():
    """Speech spans the whole video with no real gaps. Each segment's
    duration is known exactly from the gap to the NEXT segment's start;
    the LAST segment's duration is estimated from its own word count at
    a typical speaking rate (2.5 words/sec) since there's no segment
    after it to bound it precisely — so the last segment here needs
    enough words to plausibly fill the remaining 10s (long-sentence filler
    text, not a 1-word stub)."""
    long_sentence = " ".join(["word"] * 25)  # 25 words / 2.5 wps = 10s, matches 20->30
    segments = [_segment(0, "a"), _segment(10, "b"), _segment(20, long_sentence)]
    result = engineered_features([], segments, video_duration_seconds=30)
    assert result["speech_coverage_ratio"] == 1.0  # 0-10, 10-20, 20-30: full


def test_speech_coverage_ratio_partial_coverage():
    """A real silent gap (e.g. a lingering b-roll shot with no speech)
    should reduce the ratio below 1.0, not be silently assumed covered."""
    segments = [_segment(0, "a"), _segment(5, "b")]  # speech only 0-10 of a 30s video
    result = engineered_features([], segments, video_duration_seconds=30)
    assert result["speech_coverage_ratio"] < 0.5


def test_handles_empty_transcript_without_crashing():
    """A silent video (no speech) is legitimate — zero coverage, not a
    crash or a missing key."""
    result = engineered_features([], [], video_duration_seconds=30)
    assert result["speech_coverage_ratio"] == 0.0
    assert result["words_per_second"] is None


def test_words_per_second_computed_from_real_word_count_and_speech_duration():
    """First segment's duration is known exactly (gap to the next
    segment's start: 0->10 = 10s, 10 words = 1.0 wps for that part). The
    LAST segment's duration is estimated from ITS OWN word count at the
    typical speaking rate (5 words / 2.5 wps = 2s), not assumed to run to
    the video's end — so total: 15 words over a 12s estimated speech
    span = 1.25 wps."""
    segments = [_segment(0, "one two three four five six seven eight nine ten"),  # 10 words
                _segment(10, "six seven eight nine ten")]  # 5 words, LAST segment
    result = engineered_features([], segments, video_duration_seconds=15)
    assert result["words_per_second"] == 1.25


def test_requires_a_positive_video_duration():
    import pytest
    with pytest.raises(ValueError):
        engineered_features([], [], video_duration_seconds=0)
    with pytest.raises(ValueError):
        engineered_features([], [], video_duration_seconds=None)
