"""Engineered continuous features derived from v5's rich per-shot/
transcript JSONB data — pure functions, no DB dependency.

Per the architecture doc: pacing was always meant to be COMPUTED, not
LLM-guessed (originally a job for PySceneDetect, abandoned once Gemini's
v5 shot_breakdown started giving exact, real start_sec/end_sec per shot —
the same exact-timestamp data makes computing this directly possible,
without a second tool). This module is what turns the detailed-but-
statistically-inert JSONB (shot_breakdown, full_transcript — currently
ignored by learn.mining's categorical-only approach) into continuous
numeric features a regression/ML model can actually use, per the plan's
explicitly-deferred-but-intended later upgrade path ("needs enough
continuous features and enough volume to control for confounders").

All functions are pure and side-effect-free: caller already has the
shot_breakdown/full_transcript lists (from a DB row or elsewhere) and the
video's real duration; nothing here talks to Postgres or Gemini.
"""
import statistics


def engineered_features(
    shot_breakdown: list[dict],
    full_transcript: list[dict],
    video_duration_seconds: float | int | None,
) -> dict:
    """Derive continuous features from one video's shot_breakdown and
    full_transcript. Returns a flat dict of named features; any feature
    that cannot be computed from missing/empty source data is None (never
    0 as a silent stand-in for "unknown" — 0 cuts/min and "we don't know"
    are different claims and must stay distinguishable to anything
    consuming this).

    Features:
    - cuts_per_min: shots / (duration in minutes). Exact, from real
      per-shot timestamps — not PySceneDetect, not an LLM guess.
    - avg_shot_length_sec, shot_length_stddev: shape of the pacing, not
      just its rate (a video with one 2s shot and one 28s shot has the
      same cuts_per_min as one with two 15s shots, but very different
      pacing texture).
    - color_temperature_consistency: fraction of shots sharing the most
      common color_temperature value (1.0 = uniform palette throughout,
      lower = more visual variety/inconsistency).
    - setting_change_count: how many times the setting changes between
      consecutive shots (0 = single location throughout).
    - shot_type_variety: count of distinct shot_type values used.
    - speech_coverage_ratio: fraction of the video's duration actually
      covered by speech (each segment's inferred duration = gap to the
      NEXT segment's start, or video end for the last one) — distinct
      from gemini_watcher._check_coverage, which flags gaps for human
      review; this is a continuous feature for a model, not a warning.
    - words_per_second: a real speech-density/talking-speed signal,
      computed over the total speech span (first segment's start to the
      video's end), not just the sum of word counts.
    """
    if not video_duration_seconds or video_duration_seconds <= 0:
        raise ValueError(
            f"video_duration_seconds must be a positive number, got {video_duration_seconds!r}"
        )
    duration = float(video_duration_seconds)

    features: dict = {}

    if shot_breakdown:
        shots = sorted(shot_breakdown, key=lambda s: s["start_sec"])
        lengths = [s["end_sec"] - s["start_sec"] for s in shots]
        features["cuts_per_min"] = len(shots) / (duration / 60.0)
        features["avg_shot_length_sec"] = statistics.mean(lengths)
        features["shot_length_stddev"] = (
            statistics.stdev(lengths) if len(lengths) >= 2 else 0.0
        )

        temps = [s.get("color_temperature") for s in shots if s.get("color_temperature")]
        if temps:
            most_common_count = max(temps.count(t) for t in set(temps))
            features["color_temperature_consistency"] = most_common_count / len(temps)
        else:
            features["color_temperature_consistency"] = None

        settings = [s.get("setting") for s in shots]
        features["setting_change_count"] = sum(
            1 for a, b in zip(settings, settings[1:]) if a != b
        )

        shot_types = {s.get("shot_type") for s in shots if s.get("shot_type")}
        features["shot_type_variety"] = len(shot_types)
    else:
        features["cuts_per_min"] = None
        features["avg_shot_length_sec"] = None
        features["shot_length_stddev"] = None
        features["color_temperature_consistency"] = None
        features["setting_change_count"] = None
        features["shot_type_variety"] = None

    if full_transcript:
        segments = sorted(full_transcript, key=lambda t: t["start_sec"])
        starts = [t["start_sec"] for t in segments]
        word_counts = [len(t["text"].split()) for t in segments]

        # Real gap between consecutive segments is known exactly (next
        # segment's start minus this one's). The LAST segment has no next
        # segment to bound it — assuming it runs all the way to the
        # video's end would silently claim zero trailing silence, which
        # contradicts the real, already-established finding that trailing
        # silence (lingering shot, on-screen-only CTA) is normal (see
        # gemini_watcher.MAX_TRANSCRIPT_TRAILING_SILENCE_SECONDS). Instead,
        # estimate its spoken duration from its own word count at a
        # typical English speaking rate (~2.5 words/sec), capped at
        # whatever video time actually remains.
        TYPICAL_WORDS_PER_SECOND = 2.5
        last_estimated = min(
            word_counts[-1] / TYPICAL_WORDS_PER_SECOND,
            max(0.0, duration - starts[-1]),
        )
        bounds = starts[1:] + [starts[-1] + last_estimated]
        speech_seconds = sum(
            max(0.0, end - start) for start, end in zip(starts, bounds)
        )
        features["speech_coverage_ratio"] = min(1.0, speech_seconds / duration)

        total_words = sum(word_counts)
        speech_span = (starts[-1] + last_estimated) - starts[0]
        features["words_per_second"] = (
            total_words / speech_span if speech_span > 0 else None
        )
    else:
        features["speech_coverage_ratio"] = 0.0
        features["words_per_second"] = None

    return features
