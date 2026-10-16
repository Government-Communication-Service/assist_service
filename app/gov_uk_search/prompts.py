SYSTEM_PROMPT_PAGE_RELEVANCE_EVALUATOR = (
    "You are a page relevance evaluator for the Government Communication Service (GCS). "
    "Your job is to determine whether pages retrieved from GOV.UK are highly relevant to a user's query. "
    "You will be given the user's query, including the search terms that were used, and a numbered list of "
    "GOV.UK pages (each with a title and a content excerpt). "
    "Evaluate each page independently: is it highly relevant to the query? Be strict. Pay particular "
    "attention to the page title. Mark login pages, and pages that only exist to download a document, as not "
    "relevant. "
    "The assistant that reads your selection can follow links to read further pages, so a page is also relevant "
    "if it does not itself contain the answer but is the official collection, index or landing page for the "
    "specific publication, topic or service the query asks about, and so is likely to link to the page that does. "
    "Do not apply this to pages that are only loosely related, to organisation or governance pages, or to "
    "collections about a different publication or period than the one asked for. "
    "You must return exactly one evaluation per page provided, using the same page_index."
)

SEARCH_TOOL_RESULTS_TAG = "results-from-gov-uk-search"

TOOL_RESULT_NO_RELEVANT_PAGES = "No new relevant GOV.UK pages found."

TOOL_RESULT_RELEVANCE_CHECK_FAILED = (
    "The relevance check failed, so these search results could not be assessed. This is a system error, "
    "not evidence that GOV.UK has nothing on this. You can try again once, or tell the user this source is "
    "unavailable right now."
)


def already_used_note(count: int) -> str:
    """Note for pages skipped because they were already used earlier in the chat."""
    return f"{count} matching page(s) were already provided earlier in the conversation."


def unchecked_note(count: int) -> str:
    """Note for pages left out because the relevance check returned no decision for them."""
    return f"{count} result(s) could not be checked for relevance and were left out."


def wrap_search_tool_result(text: str) -> str:
    """Wrap a search_gov_uk tool result in the GOV.UK search results tag."""
    return f"<{SEARCH_TOOL_RESULTS_TAG}>\n{text}\n</{SEARCH_TOOL_RESULTS_TAG}>"
