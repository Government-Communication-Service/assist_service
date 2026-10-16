from pydantic import Field

from app.bedrock.tool_schema import LenientList, ToolArgs, ToolDef
from app.config import AUDIENCE_SEGMENTS_MAX_TOOL_SEGMENTS


class GetAudienceSegmentsArgs(ToolArgs):
    """Get the insights for one or more segments from the Government Audience Segmentation: evidence-
    backed profiles of UK audiences (attitudes, behaviours, media use, how to reach them). The available
    segments are listed in the system prompt. Use it when the user wants to understand, target or write
    for an audience that one of the segments covers, or asks about a segment by name."""

    segment_names: LenientList[str] = Field(
        description=f"1 to {AUDIENCE_SEGMENTS_MAX_TOOL_SEGMENTS} segment names, exactly as listed in the system prompt"
    )


GET_AUDIENCE_SEGMENTS = ToolDef("get_audience_segments", GetAudienceSegmentsArgs, strict=False)
