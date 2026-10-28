from app.config import ThinkingLevel


def thinking_kwargs(level: ThinkingLevel) -> dict:
    """Return Anthropic SDK kwargs that configure thinking for the given level.

    disabled → {"thinking": {"type": "between_tools"}}
    low/medium/high/xhigh/max → {"output_config": {"effort": <level>}}

    The Sonnet 5.5 thinking API (anthropic SDK ≥0.103):
    - budget_tokens is removed; depth is controlled via output_config.effort.
    - thinking={"type": "disabled"} is rejected with a 400. `between_tools` is the lowest setting: no
      extended thinking, but short progress notes between tool calls come back as `thinking` blocks
      (pass them back unchanged). It takes no other field and is only valid at effort `high` or below,
      so it is sent without an explicit effort (the model default is `high`).
    - Omitting `thinking` runs adaptive thinking.
    """
    if level == ThinkingLevel.disabled:
        return {"thinking": {"type": "between_tools"}}
    return {"output_config": {"effort": level.value}}
