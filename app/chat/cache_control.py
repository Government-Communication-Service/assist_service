from logging import getLogger

from app.compaction.service import estimate_message_tokens
from app.config import CacheTtl, settings

logger = getLogger(__name__)

# Sonnet 5's minimum cacheable prefix. Shorter prefixes are not stored at all.
MIN_CACHEABLE_PREFIX_TOKENS = 1024

# Block types the API won't accept a cache_control marker on.
_UNCACHEABLE_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})


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
            (which sends the messages before the current turn) shares the same cached prefix.
            Otherwise it is placed on the final message.

    Returns:
        A copy of new_messages with the breakpoint applied, or new_messages unchanged if
        caching is disabled (via `message_cache_control_enabled`, or via
        `compaction_cache_control_enabled` when `should_compact` is set), the conversation is
        below the minimum cacheable prefix, or (when `should_compact` is set) there is no
        assistant message to mark. String content is converted to a single text block;
        for block-list content the marker goes on the last block that accepts one.
    """
    cache_control_enabled = settings.message_cache_control_enabled or (
        should_compact and settings.compaction_cache_control_enabled
    )
    if not new_messages or not cache_control_enabled:
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

    marked_content = _with_cache_control(new_messages[index]["content"])
    if marked_content is None:
        return new_messages

    marked_messages = list(new_messages)
    marked_messages[index] = {**marked_messages[index], "content": marked_content}
    return marked_messages


def _with_cache_control(content: str | list[dict]) -> list[dict] | None:
    """Return content as a block list with a cache breakpoint on its last cacheable block.

    Args:
        content: A string, or a list of content blocks (not mutated).

    Returns:
        A new block list, or None if there is no block that can carry the marker.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content, "cache_control": message_cache_control()}]

    index = next(
        (i for i in range(len(content) - 1, -1, -1) if content[i].get("type") not in _UNCACHEABLE_BLOCK_TYPES),
        None,
    )
    if index is None:
        return None
    marked_content = list(content)
    marked_content[index] = {**content[index], "cache_control": message_cache_control()}
    return marked_content
