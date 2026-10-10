from langchain_anthropic import ChatAnthropic

# Verified against the live Anthropic API on 2026-10-10 — these are real, working model IDs,
# not placeholders. Re-verify periodically since Anthropic's lineup moves; a quick check:
#   uv run python -c "import anthropic; print([m.id for m in anthropic.Anthropic().models.list().data])"
_MODELS = {
    "low": "claude-haiku-4-5",      # cheap cleanup/formatting/classification tasks
    "medium": "claude-sonnet-4-5",  # default generation/annotation
    "high": "claude-opus-4-5",      # structurally-critical generation (rare, expensive)
}


def pick_llm(level: str = "medium", **kwargs) -> ChatAnthropic:
    """Return a Claude chat model tiered by task complexity.

    level: 'low' | 'medium' | 'high'
    """
    level = level.lower()
    if level not in _MODELS:
        raise ValueError(f"Unknown model level: {level!r}. Must be one of {list(_MODELS)}.")
    return ChatAnthropic(
        model=_MODELS[level],
        temperature=kwargs.pop("temperature", 0),
        **kwargs,
    )
