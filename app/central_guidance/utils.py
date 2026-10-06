import json
import logging
from typing import Any, List

from app.central_guidance.constants import MAX_REWRITTEN_QUERIES

logger = logging.getLogger(__name__)


def sanitise_keyword_queries(value: Any, fallback_query: str) -> List[str]:
    """Coerce the query-rewriter tool's `keyword_queries` into a bounded list of non-empty strings.

    The model sometimes returns the array as a single string (e.g. '["a", "b"]') rather than a JSON
    array. Iterating that string directly runs one search per character, so it is parsed back into
    a list, or treated as one query if it isn't a JSON array. Falls back to the user's own query if
    nothing usable is returned, or to no queries at all if that is blank too, so no search runs on an
    empty string.
    """
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = None
        value = parsed if isinstance(parsed, list) else [value]

    if not isinstance(value, list):
        logger.warning(f"Unexpected keyword_queries type from query rewriter: {type(value).__name__}")
        value = []

    queries = list(dict.fromkeys(q.strip() for q in value if isinstance(q, str) and q.strip()))
    if not queries:
        fallback_query = (fallback_query or "").strip()
        if not fallback_query:
            logger.warning("Query rewriter returned no usable queries and the user's query is blank")
            return []
        logger.warning("Query rewriter returned no usable queries, falling back to the user's query")
        return [fallback_query]

    if len(queries) > MAX_REWRITTEN_QUERIES:
        logger.warning(f"Query rewriter returned {len(queries)} queries, truncating to {MAX_REWRITTEN_QUERIES}")
    return queries[:MAX_REWRITTEN_QUERIES]
