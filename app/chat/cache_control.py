from logging import getLogger

from app.compaction.service import estimate_message_tokens
from app.config import CacheTtl, settings

logger = getLogger(__name__)

# Sonnet 5's minimum cacheable prefix. Shorter prefixes are not stored at all.
MIN_CACHEABLE_PREFIX_TOKENS = 1024


def message_cache_control() -> dict:
    """Build the cache_control marker for the conversation breakpoint.

    Returns:
        An ephemeral cache_control block. `ttl` is set only for the 1-hour option; the
        5-minute default is expressed by omitting it.
    """
    block = {"type": "ephemeral"}
    if settings.message_cache_ttl == CacheTtl.one_hour:
        block["ttl"] = settings.message_cache_ttl.value
    return block


def apply_compaction_aware_cache_control(new_messages: list[dict], should_compact: bool) -> list[dict]:
    """Return a copy of new_messages with a cache breakpoint applied.

    Args:
        new_messages: Formatted messages ready for the LLM call, oldest first.
        should_compact: Whether a same-turn compaction call will reuse this cache. If set,
            the breakpoint is placed on the last assistant message, so the compaction call
            (which uses `new_messages[:-1]`) shares the same cached prefix. Otherwise it is
            placed on the final message.

    Returns:
        A copy of new_messages with the breakpoint applied, or new_messages unchanged if
        caching is disabled, the conversation is below the minimum cacheable prefix, or (when
        `should_compact` is set) there is no assistant message to mark.
    """
    if not new_messages or not settings.message_cache_control_enabled:
        return new_messages

    # Below the model's minimum cacheable prefix nothing is stored at all, so the breakpoint
    # would simply be wasted.
    estimated_tokens = sum(estimate_message_tokens(msg["content"]) for msg in new_messages)
    if estimated_tokens < MIN_CACHEABLE_PREFIX_TOKENS:
        logger.debug(
            f"Conversation is ~{estimated_tokens} tokens, below the {MIN_CACHEABLE_PREFIX_TOKENS}-token "
            "minimum for caching; skipping the cache breakpoint"
        )
        return new_messages

    if should_compact:
        index = next(
            (i for i in range(len(new_messages) - 1, -1, -1) if new_messages[i]["role"] == "assistant"),
            None,
        )
        if index is None:
            return new_messages
    else:
        index = len(new_messages) - 1

    marked_messages = list(new_messages)
    marked_messages[index] = {
        **marked_messages[index],
        "content": [
            {
                "type": "text",
                "text": marked_messages[index]["content"],
                "cache_control": message_cache_control(),
            }
        ],
    }
    return marked_messages
