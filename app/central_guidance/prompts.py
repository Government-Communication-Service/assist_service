TOOL_RESULTS_TAG = "government-comms-central-guidance-search-results"

SYSTEM_PROMPT_CHUNK_RELEVANCE_EVALUATOR = (
    "You are a document relevance evaluator for the Government Communication Service (GCS). "
    "Your job is to determine if document chunks are relevant to a user's query. "
    "You will be given a user's original query and a numbered list of document chunks "
    "(each with a title and content). "
    "Evaluate each chunk independently: does it contain information that would help answer "
    "the user's question? Be judicious, and think carefully before giving a final answer. "
    "You must return exactly one evaluation per chunk provided, using the same chunk_index."
)

TOOL_RESULT_NO_QUERIES = "No search queries were given."

TOOL_RESULT_NO_RELEVANT_PASSAGES = "No new relevant passages found for these queries."

TOOL_RESULT_RELEVANCE_CHECK_FAILED = (
    "The relevance check failed, so these search results could not be assessed. This is a system error, "
    "not evidence that GCS guidance has nothing on this. You can try again once, or tell the user this "
    "source is unavailable right now."
)


def already_shown_note(count: int) -> str:
    """Note for passages skipped because they were already cited earlier in the chat."""
    return f"{count} matching passage(s) were already provided earlier in the conversation."


def unchecked_note(count: int) -> str:
    """Note for passages left out because the relevance check returned no decision for them."""
    return f"{count} result(s) could not be checked for relevance and were left out."


def wrap_tool_result(text: str) -> str:
    """Wrap a tool result message in the central guidance results tag."""
    return f"<{TOOL_RESULTS_TAG}>\n{text}\n</{TOOL_RESULTS_TAG}>"
