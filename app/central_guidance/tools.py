from pydantic import Field

from app.bedrock.tool_schema import REMOVE_DUPLICATES_DESCRIPTION, LenientList, ToolArgs, ToolDef
from app.config import CENTRAL_GUIDANCE_MAX_TOOL_QUERIES


class SearchCentralGuidanceArgs(ToolArgs):
    """Search the GCS central guidance library: authoritative guidance on government communications,
    including GCS strategy, the Modern Communications Operating Model (MCOM), OASIS campaign planning,
    propriety and ethics, accessible communications, HM Government brand guidelines, crisis and
    strategic communication, behavioural science (COM-B) and other GCS professional resources.
    Use it when the user asks about GCS policy, best practice, comms frameworks or anything likely
    covered by GCS central documents. Returns relevant passages with their document titles. Search
    again with different keywords if the results don't cover what you need."""

    queries: LenientList[str] = Field(
        description=(
            f"1 to {CENTRAL_GUIDANCE_MAX_TOOL_QUERIES} keyword queries for a BM25 keyword search over "
            "document titles, section titles and "
            "text. Quote the most relevant words from the user's message in the first query; use "
            "synonyms in the others to widen the search rather than repeating the same words."
        )
    )
    remove_duplicates: bool = Field(default=False, description=REMOVE_DUPLICATES_DESCRIPTION)


SEARCH_CENTRAL_GUIDANCE = ToolDef("search_central_guidance", SearchCentralGuidanceArgs, strict=False)


class ChunkEvaluation(ToolArgs):
    """One chunk's evaluation."""

    chunk_index: int = Field(description="The index of the chunk being evaluated, matching its <chunk-N> tag.")
    reasoning: str = Field(description="Brief explanation of why the chunk is or isn't relevant to the user's query.")
    is_relevant: bool = Field(
        description=(
            "True if the document chunk contains information relevant to the user's query; false if the chunk "
            "is not relevant or useful for answering the user's question."
        )
    )


class ChunkRelevanceArgs(ToolArgs):
    """Evaluate whether each of a list of document chunks is relevant to the user's query"""

    evaluations: LenientList[ChunkEvaluation] = Field(description="One evaluation per chunk provided, in any order.")


CHUNK_RELEVANCE = ToolDef("evaluate_chunk_relevance", ChunkRelevanceArgs)
