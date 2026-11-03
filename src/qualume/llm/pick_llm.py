import os

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Verified against the live Anthropic API on 2026-10-10 — these are real, working model IDs,
# not placeholders. Re-verify periodically since Anthropic's lineup moves; a quick check:
#   uv run python -c "import anthropic; print([m.id for m in anthropic.Anthropic().models.list().data])"
_MODELS = {
    "low": "claude-haiku-4-5",      # cheap cleanup/formatting/classification tasks
    "medium": "claude-sonnet-4-5",  # default generation/annotation
    "high": "claude-opus-4-5",      # structurally-critical generation (rare, expensive)
}

# Real fix, 2026-11-03: the project's prior "Claude-only, one scoped Gemini
# exception for video-watching" rule (see gemini_watcher.py, transcript_
# embedding.py) broke in practice — the real ANTHROPIC_API_KEY ran out of
# prepaid credit mid-session, which silently kills every 'low'-tier call
# (classification, cluster labeling) that this project's HIGH-VOLUME
# pipeline stages fire hundreds/thousands of times. The user's Claude Pro
# SUBSCRIPTION (via Claude Code's OAuth token) cannot substitute here: that
# token only authenticates the Claude Code CLI/Agent SDK, not the plain
# Messages API ChatAnthropic calls, and per-call CLI-process spawn overhead
# (hundreds of ms-seconds each, vs. one reused HTTP connection) plus
# interactive-session rate limits make shelling out to `claude -p` a bad
# fit for thousands of small automated calls anyway.
#
# Real fix: 'low'-tier calls are pure text-in/text-out classification/
# labeling (CopyAnalysisV2, ContentTypeClassification, cluster labels) —
# none of them need Claude's native video understanding, so they move to
# Gemini's free tier on a GEMINI_MODEL_LOW model that is DIFFERENT from
# GEMINI_MODEL (gemini_watcher.py's video-watching model). Confirmed live
# against Google's own rate-limits docs (ai.google.dev/gemini-api/docs/
# rate-limits, checked 2026-11-03): "Limits vary depending on the specific
# model being used" — free-tier quota is tracked PER MODEL, not pooled
# across a project, so this does not compete with or shrink the existing
# video-annotation quota. 'medium'/'high' stay on Claude: those are the
# actual content-GENERATION steps (write_ig_static's angle/copy drafting),
# lower call volume per run, and quality-sensitive in a way the cheap
# classification stages aren't — not blocked by this same credit issue, so
# left alone rather than changed speculatively.
_GEMINI_MODELS = {
    "low": "gemini-3.5-flash-lite",  # verified live 2026-11-03: works with
    # .with_structured_output(PydanticModel) via langchain-google-genai,
    # ~3-6s per call, free tier, no credit card. gemini-2.5-flash-lite
    # returned a live 404 (no longer available to new users per Google's
    # own error message) — do not revert to it.
}


def pick_llm(level: str = "medium", provider: str | None = None, **kwargs):
    """Return a chat model tiered by task complexity.

    level: 'low' | 'medium' | 'high'
    provider: None (default routing below) | 'anthropic' | 'gemini' — force
    a specific provider regardless of level, e.g. for a one-off test or to
    temporarily route a generation call off Claude if its credit runs out
    too. Default routing: 'low' -> Gemini (free, separate per-model quota
    from video-watching, fast), 'medium'/'high' -> Claude (generation
    quality, lower volume, not currently blocked).
    """
    level = level.lower()
    if level not in _MODELS:
        raise ValueError(f"Unknown model level: {level!r}. Must be one of {list(_MODELS)}.")

    resolved_provider = provider or ("gemini" if level == "low" else "anthropic")

    if resolved_provider == "gemini":
        if level not in _GEMINI_MODELS:
            raise ValueError(
                f"No Gemini model mapped for level {level!r} — only 'low' is "
                "routed to Gemini by default; pass provider='anthropic' explicitly "
                "for medium/high, or add a _GEMINI_MODELS entry if you really want this."
            )
        # Passed explicitly, same discipline as gemini_watcher.get_gemini_client —
        # a stray GOOGLE_API_KEY env var would otherwise silently take priority.
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set — cannot route a 'low'-tier call to Gemini.")
        kwargs.pop("temperature", None)  # Gemini 3.x models use fixed sampling defaults;
        # passing temperature triggers a live UserWarning and is ignored anyway (verified
        # 2026-11-03) — dropped here rather than passed through to avoid noisy warnings.
        return ChatGoogleGenerativeAI(model=_GEMINI_MODELS[level], google_api_key=api_key, **kwargs)

    return ChatAnthropic(
        model=_MODELS[level],
        temperature=kwargs.pop("temperature", 0),
        **kwargs,
    )
