from app.config import ThinkingLevel


def thinking_kwargs(level: ThinkingLevel) -> dict:
    """Return Anthropic SDK kwargs that configure thinking for the given level.

    disabled → {"thinking": {"type": "disabled"}}
    low/medium/high/xhigh/max → {"output_config": {"effort": <level>}}

    The Sonnet 5 thinking API (anthropic SDK ≥0.103):
    - budget_tokens is removed; depth is controlled via output_config.effort.
    - Passing thinking={"type": "disabled"} is the only way to suppress thinking
      on Sonnet 5, which runs adaptive thinking by default when the param is omitted.
    """
    if level == ThinkingLevel.disabled:
        return {"thinking": {"type": "disabled"}}
    return {"output_config": {"effort": level.value}}
