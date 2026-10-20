"""Tools available to the chat agent loop: per-request context, executors and the registry.

Tool argument models and `ToolDef`s live in each feature module's `tools.py`; this module
binds them to executors and user-facing status labels.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic

from sqlalchemy.ext.asyncio import AsyncSession

from app.audience_segments.services import AudienceSegmentsService
from app.audience_segments.tools import GET_AUDIENCE_SEGMENTS, GetAudienceSegmentsArgs
from app.bedrock.call_log import LlmCallContext
from app.bedrock.tool_schema import ArgsT, ToolDef
from app.central_guidance.service_rag import search_central_guidance_for_tool
from app.central_guidance.tools import SEARCH_CENTRAL_GUIDANCE, SearchCentralGuidanceArgs
from app.chat.schemas import (
    AudienceSegmentsSource,
    CentralGuidanceSource,
    ChatCreateMessageInput,
    GovUkSearchSource,
    Sources,
    StyleGuideSource,
    UserDocumentSource,
)
from app.chat.utils import truncate_query_for_helper_call
from app.config import (
    AUDIENCE_SEGMENTS_MAX_TOOL_SEGMENTS,
    CAMPAIGN_BENCHMARKS_SERVICE_DISABLED,
    TOOL_STATUS_LABEL_MAX_QUERY_CHARS,
    USE_RAG,
)
from app.database.models import Chat, Message
from app.document_upload.service import read_uploaded_document_for_tool, search_uploaded_documents_for_tool
from app.document_upload.tools import (
    READ_UPLOADED_DOCUMENT,
    SEARCH_UPLOADED_DOCUMENTS,
    ReadUploadedDocumentArgs,
    SearchUploadedDocumentsArgs,
)
from app.gov_uk_search.service import get_gov_uk_page_for_tool, get_gov_uk_pdf_for_tool, search_gov_uk_for_tool
from app.gov_uk_search.tools import (
    GET_GOV_UK_PAGE,
    GET_GOV_UK_PDF,
    SEARCH_GOV_UK,
    GetGovUkPageArgs,
    GetGovUkPdfArgs,
    SearchGovUkArgs,
)
from app.smart_targets.service import SmartTargetsService
from app.smart_targets.tools import QUERY_CAMPAIGN_BENCHMARKS, QuerySmartTargetsArgs
from app.style_guide.service import check_style_guide_for_tool
from app.style_guide.tools import CHECK_STYLE_GUIDE, CheckStyleGuideArgs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolContext:
    """Per-request state for executors. Identity and scope come from here, never from tool args.

    Holds no DB session: `_run_tool_call` opens a short one for each executor call and passes it in, so
    no connection is held across model calls.

    Args:
        chat: The chat being answered.
        user_id: Id of the chat's owner.
        user_message: This turn's user message row.
        history: Earlier message rows in the chat, oldest first, excluding user_message.
        input_data: The request, for option flags.
        document_uuids: Uploaded documents attached to the chat.
    """

    chat: Chat
    user_id: int
    user_message: Message
    history: list[Message]
    input_data: ChatCreateMessageInput
    document_uuids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ToolCall:
    """The model call that requested a tool.

    Args:
        tool_use_id: The tool_use block id.
        iteration: 1-based loop iteration.
        llm_internal_response_id: Logged row for the model call that made the request.
    """

    tool_use_id: str
    iteration: int
    llm_internal_response_id: int | None


@dataclass
class ToolExecResult:
    """An executor's output.

    Args:
        content: Text for the tool_result block, bounded and wrapped in the tool's XML tags.
        citations: Citation dicts (`docname`, `docurl`) for the message's citations array.
        sources: Sources to show under the answer.
        is_error: Send the result with is_error set.
    """

    content: str
    citations: list[dict] = field(default_factory=list)
    sources: Sources = field(default_factory=Sources)
    is_error: bool = False


@dataclass(frozen=True)
class RegisteredTool(Generic[ArgsT]):
    """A tool the loop can offer.

    `enabled` reflects deploy-level env flags (such as `USE_RAG`), not anything about the current
    chat, so the tool list sent to the model never varies by chat: that keeps the tools block of
    the prompt cache stable. It's a callable rather than a plain bool so it always reads the
    current flag value (env flags don't change while a process runs, but tests need to patch
    them), while still taking no `ctx`, to keep it obviously chat-independent. Per-chat
    availability (checkboxes, documents attached) is `is_available` instead: a call to a tool
    that is enabled but not available for this chat gets `unavailable_message` as an is_error
    result, rather than being left out of the request.

    Args:
        tool_def: Name, spec and argument parsing.
        executor: Runs the tool with validated args, on a session opened for this call alone.
        enabled: Whether this deploy offers the tool at all. Not chat-dependent.
        is_available: Whether the tool can be used in this chat.
        unavailable_message: is_error text sent to the model when `is_available` is false,
            telling it how the user can enable the tool. Unused if `is_available` is always true.
        running_label: Status text while running (en-GB, user-facing).
        done_label: Status text once finished.
        error_label: Status text if it failed. Must not contain internal detail.
    """

    tool_def: ToolDef[ArgsT]
    executor: Callable[[ToolContext, ArgsT, ToolCall, AsyncSession], Awaitable[ToolExecResult]]
    running_label: Callable[[ArgsT], str]
    done_label: Callable[[ArgsT], str]
    error_label: str
    enabled: Callable[[], bool] = lambda: True  # noqa: E731
    is_available: Callable[[ToolContext], bool] = lambda ctx: True  # noqa: E731
    unavailable_message: str = "This tool isn't available in this chat."

    @property
    def name(self) -> str:
        return self.tool_def.name


def quote_for_label(text: str) -> str:
    """Quote model-written text for a status label, shortened to TOOL_STATUS_LABEL_MAX_QUERY_CHARS."""
    collapsed = " ".join(text.split())
    if len(collapsed) > TOOL_STATUS_LABEL_MAX_QUERY_CHARS:
        collapsed = collapsed[: TOOL_STATUS_LABEL_MAX_QUERY_CHARS - 1].rstrip() + "…"
    return f"“{collapsed}”"


# --- executors ---


async def _exec_search_central_guidance(
    ctx: ToolContext, args: SearchCentralGuidanceArgs, call: ToolCall, db_session: AsyncSession
) -> ToolExecResult:
    content, citations, failed = await search_central_guidance_for_tool(
        queries=args.queries,
        remove_duplicates=args.remove_duplicates,
        message_id=ctx.user_message.id,
        user_query=truncate_query_for_helper_call(ctx.user_message.content),
        llm_internal_response_id=call.llm_internal_response_id,
        db_session=db_session,
        recent_messages=ctx.history,
    )
    return ToolExecResult(
        content=content,
        citations=citations,
        sources=Sources(
            central_guidance_sources=[
                CentralGuidanceSource(pretty_name=c["docname"], url=c["docurl"]) for c in citations
            ]
        ),
        is_error=failed,
    )


async def _exec_search_uploaded_documents(
    ctx: ToolContext, args: SearchUploadedDocumentsArgs, call: ToolCall, db_session: AsyncSession
) -> ToolExecResult:
    content, citations, failed = await search_uploaded_documents_for_tool(
        queries=args.queries,
        remove_duplicates=args.remove_duplicates,
        document_ids=args.document_ids,
        document_uuids=ctx.document_uuids,
        message_id=ctx.user_message.id,
        llm_internal_response_id=call.llm_internal_response_id,
        db_session=db_session,
    )
    return ToolExecResult(
        content=content,
        citations=citations,
        sources=Sources(
            user_document_sources=[UserDocumentSource(pretty_name=c["docname"], url=c["docurl"]) for c in citations]
        ),
        is_error=failed,
    )


async def _exec_read_uploaded_document(
    ctx: ToolContext, args: ReadUploadedDocumentArgs, call: ToolCall, db_session: AsyncSession
) -> ToolExecResult:
    content, citations, failed = await read_uploaded_document_for_tool(
        document_id=args.document_id,
        start_chunk=args.start_chunk,
        end_chunk=args.end_chunk,
        document_uuids=ctx.document_uuids,
        message_id=ctx.user_message.id,
        db_session=db_session,
    )
    return ToolExecResult(
        content=content,
        citations=citations,
        sources=Sources(
            user_document_sources=[UserDocumentSource(pretty_name=c["docname"], url=c["docurl"]) for c in citations]
        ),
        is_error=failed,
    )


async def _exec_search_gov_uk(
    ctx: ToolContext, args: SearchGovUkArgs, call: ToolCall, db_session: AsyncSession
) -> ToolExecResult:
    content, citations, failed = await search_gov_uk_for_tool(
        args=args,
        user_query=truncate_query_for_helper_call(ctx.user_message.content),
        message_id=ctx.user_message.id,
        llm_internal_response_id=call.llm_internal_response_id,
        db_session=db_session,
    )
    return ToolExecResult(
        content=content,
        citations=citations,
        sources=Sources(
            gov_uk_search_sources=[GovUkSearchSource(pretty_name=c["docname"], url=c["docurl"]) for c in citations]
        ),
        is_error=failed,
    )


def _gov_uk_label(args: SearchGovUkArgs, running: bool) -> str:
    return f"{'Searching' if running else 'Searched'} GOV.UK for {quote_for_label('; '.join(args.queries))}"


async def _exec_get_gov_uk_page(
    ctx: ToolContext, args: GetGovUkPageArgs, call: ToolCall, db_session: AsyncSession
) -> ToolExecResult:
    content, citations, failed = await get_gov_uk_page_for_tool(args.url)
    return ToolExecResult(
        content=content,
        citations=citations,
        sources=Sources(
            gov_uk_search_sources=[GovUkSearchSource(pretty_name=c["docname"], url=c["docurl"]) for c in citations]
        ),
        is_error=failed,
    )


async def _exec_get_gov_uk_pdf(
    ctx: ToolContext, args: GetGovUkPdfArgs, call: ToolCall, db_session: AsyncSession
) -> ToolExecResult:
    content, citations, failed = await get_gov_uk_pdf_for_tool(args.url, args.start_page)
    return ToolExecResult(
        content=content,
        citations=citations,
        sources=Sources(
            gov_uk_search_sources=[GovUkSearchSource(pretty_name=c["docname"], url=c["docurl"]) for c in citations]
        ),
        is_error=failed,
    )


async def _exec_query_smart_targets(
    ctx: ToolContext, args: QuerySmartTargetsArgs, call: ToolCall, db_session: AsyncSession
) -> ToolExecResult:
    content, _sources = await SmartTargetsService().query_metrics_for_tool(
        requests=[(m.metric_name, m.context_for_filters) for m in args.metrics],
        messages=[*ctx.history, ctx.user_message],
        context=LlmCallContext(
            purpose="smart_targets_select_filters", chat_id=ctx.chat.id, message_id=ctx.user_message.id
        ),
    )
    # Smart Targets currently attaches no sources; `_sources` holds them if they are added back.
    return ToolExecResult(content=content)


def _metrics_label(args: QuerySmartTargetsArgs) -> str:
    return quote_for_label("; ".join(m.metric_name for m in args.metrics))


async def _exec_get_audience_segments(
    ctx: ToolContext, args: GetAudienceSegmentsArgs, call: ToolCall, db_session: AsyncSession
) -> ToolExecResult:
    segments, missing = await AudienceSegmentsService.find_segments(
        args.segment_names[:AUDIENCE_SEGMENTS_MAX_TOOL_SEGMENTS]
    )
    parts = ["<audience-segment-information>", *(segment.wrap_for_context() for segment in segments)]
    if missing:
        library = await AudienceSegmentsService.get_segment_library()
        parts.append(
            f"Unknown segment(s): {', '.join(missing)}. Available: {', '.join(s.pretty_name for s in library)}"
        )
    parts.append("</audience-segment-information>")
    return ToolExecResult(
        content="\n".join(parts),
        sources=Sources(
            audience_segments_sources=[
                AudienceSegmentsSource(pretty_name=segment.pretty_name, url=segment.connect_url) for segment in segments
            ]
            or None
        ),
    )


async def _exec_check_style_guide(
    ctx: ToolContext, args: CheckStyleGuideArgs, call: ToolCall, db_session: AsyncSession
) -> ToolExecResult:
    content, failed = await check_style_guide_for_tool(
        text=args.text,
        document_id=args.document_id,
        checker_context=args.context,
        document_uuids=ctx.document_uuids,
        user_id=ctx.user_id,
        context=LlmCallContext(purpose="style_guide_validation", chat_id=ctx.chat.id, message_id=ctx.user_message.id),
    )
    return ToolExecResult(
        content=content,
        sources=Sources(
            style_guide_sources=[
                StyleGuideSource(pretty_name="GOV.UK style guide", url="https://www.gov.uk/guidance/style-guide/a-to-z")
            ]
        ),
        is_error=failed,
    )


def _segments_label(args: GetAudienceSegmentsArgs) -> str:
    return quote_for_label("; ".join(args.segment_names))


# --- registry ---

TOOL_REGISTRY: dict[str, RegisteredTool[Any]] = {
    tool.name: tool
    for tool in [
        RegisteredTool(
            tool_def=SEARCH_CENTRAL_GUIDANCE,
            executor=_exec_search_central_guidance,
            enabled=lambda: USE_RAG,
            is_available=lambda ctx: ctx.input_data.use_rag,
            unavailable_message=(
                "Central documents search is turned off in this chat. Tell the user they can turn on "
                "'Central documents search' in the chat options."
            ),
            running_label=lambda args: f"Searching GCS guidance for {quote_for_label('; '.join(args.queries))}",
            done_label=lambda args: f"Searched GCS guidance for {quote_for_label('; '.join(args.queries))}",
            error_label="Couldn't search GCS guidance",
        ),
        RegisteredTool(
            tool_def=SEARCH_UPLOADED_DOCUMENTS,
            executor=_exec_search_uploaded_documents,
            is_available=lambda ctx: bool(ctx.document_uuids),
            unavailable_message=(
                "No documents are attached to this chat. Tell the user they can upload or select documents."
            ),
            running_label=lambda args: f"Searching your documents for {quote_for_label('; '.join(args.queries))}",
            done_label=lambda args: f"Searched your documents for {quote_for_label('; '.join(args.queries))}",
            error_label="Couldn't search your documents",
        ),
        RegisteredTool(
            tool_def=READ_UPLOADED_DOCUMENT,
            executor=_exec_read_uploaded_document,
            is_available=lambda ctx: bool(ctx.document_uuids),
            unavailable_message=(
                "No documents are attached to this chat. Tell the user they can upload or select documents."
            ),
            running_label=lambda args: "Reading your document",
            done_label=lambda args: "Read your document",
            error_label="Couldn't read your document",
        ),
        RegisteredTool(
            tool_def=SEARCH_GOV_UK,
            executor=_exec_search_gov_uk,
            is_available=lambda ctx: bool(ctx.input_data.use_gov_uk_search_api),
            unavailable_message=(
                "GOV.UK search is turned off in this chat. Tell the user they can turn on 'Use GOV.UK Search' "
                "in the chat options."
            ),
            running_label=lambda args: _gov_uk_label(args, running=True),
            done_label=lambda args: _gov_uk_label(args, running=False),
            error_label="Couldn't search GOV.UK",
        ),
        RegisteredTool(
            tool_def=GET_GOV_UK_PAGE,
            executor=_exec_get_gov_uk_page,
            is_available=lambda ctx: bool(ctx.input_data.use_gov_uk_search_api),
            unavailable_message=(
                "GOV.UK search is turned off in this chat. Tell the user they can turn on 'Use GOV.UK Search' "
                "in the chat options."
            ),
            running_label=lambda args: f"Reading GOV.UK page {quote_for_label(args.url)}",
            done_label=lambda args: f"Read GOV.UK page {quote_for_label(args.url)}",
            error_label="Couldn't read the GOV.UK page",
        ),
        RegisteredTool(
            tool_def=GET_GOV_UK_PDF,
            executor=_exec_get_gov_uk_pdf,
            is_available=lambda ctx: bool(ctx.input_data.use_gov_uk_search_api),
            unavailable_message=(
                "GOV.UK search is turned off in this chat. Tell the user they can turn on 'Use GOV.UK Search' "
                "in the chat options."
            ),
            running_label=lambda args: f"Reading GOV.UK PDF {quote_for_label(args.url)}",
            done_label=lambda args: f"Read GOV.UK PDF {quote_for_label(args.url)}",
            error_label="Couldn't read the GOV.UK PDF",
        ),
        RegisteredTool(
            tool_def=QUERY_CAMPAIGN_BENCHMARKS,
            executor=_exec_query_smart_targets,
            enabled=lambda: not CAMPAIGN_BENCHMARKS_SERVICE_DISABLED,
            is_available=lambda ctx: bool(ctx.input_data.use_smart_targets),
            unavailable_message=(
                "Campaign benchmarks is turned off in this chat. Tell the user they can turn on the "
                "'Campaign benchmarks' option in the chat options."
            ),
            running_label=lambda args: f"Getting Smart Targets benchmarks for {_metrics_label(args)}",
            done_label=lambda args: f"Got Smart Targets benchmarks for {_metrics_label(args)}",
            error_label="Couldn't get Smart Targets benchmarks",
        ),
        RegisteredTool(
            tool_def=GET_AUDIENCE_SEGMENTS,
            executor=_exec_get_audience_segments,
            running_label=lambda args: f"Reading audience insights for {_segments_label(args)}",
            done_label=lambda args: f"Read audience insights for {_segments_label(args)}",
            error_label="Couldn't get audience insights",
        ),
        RegisteredTool(
            tool_def=CHECK_STYLE_GUIDE,
            executor=_exec_check_style_guide,
            running_label=lambda args: "Checking against the GOV.UK style guide",
            done_label=lambda args: "Checked against the GOV.UK style guide",
            error_label="Couldn't run the style guide check",
        ),
    ]
}


def build_available_tools() -> list[RegisteredTool[Any]]:
    """Tools enabled for this deploy, sorted by name so the spec list is byte-stable across
    requests and chats. Per-chat availability is handled separately, by each tool's
    `is_available`/`unavailable_message` (see `RegisteredTool`), not by filtering this list.
    """
    return sorted((tool for tool in TOOL_REGISTRY.values() if tool.enabled()), key=lambda tool: tool.name)
