"""
Style Guide API Routes
Endpoints for checking documents against GOV.UK style guide rules.
"""
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth.verify_service import verify_auth_token
from app.config import STYLE_GUIDE_LLM_MODEL
from app.style_guide.style_guide_checker import (
    check_case_insensitive_rules,
    check_case_sensitive_rules,
    check_llm_validation_rules,
    generate_summary_and_fix,
    load_rule_mapping,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Load rules once at startup
SCRIPT_DIR = Path(__file__).parent
RULES_FILE = SCRIPT_DIR / "rule_mapping.json"
RULES = load_rule_mapping(RULES_FILE)


class StyleGuideCheckRequest(BaseModel):
    """Request model for style guide check."""

    content: str = Field(..., description="The text content to check against GOV.UK style guide")
    target_audience: Optional[str] = Field(None, description="Target audience for the content")
    specific_concerns: Optional[str] = Field(None, description="Specific areas of concern to focus on")
    skip_llm: bool = Field(False, description="Skip LLM validation and only run deterministic checks")
    llm_model: Optional[str] = Field(None, description="LLM model to use for validation")


class ViolationDetail(BaseModel):
    """Detail of a single violation."""

    rule_id: str
    rule_title: str
    severity: str
    example: str
    sentence: str
    position: int


class StyleGuideCheckResponse(BaseModel):
    """Response model for style guide check."""

    violation_count: int = Field(..., description="Total number of violations found")
    violations: List[ViolationDetail] = Field(..., description="List of all violations")
    summary: Optional[str] = Field(None, description="LLM-generated summary of issues")
    fixed_document: Optional[str] = Field(None, description="LLM-generated corrected version")
    violation_list: Optional[str] = Field(None, description="Formatted list of violations")


@router.post(
    "/check",
    response_model=StyleGuideCheckResponse,
    dependencies=[Depends(verify_auth_token)],
    summary="Check content against GOV.UK style guide",
    description="Analyze text content for violations of Government Digital Service style guide rules",
)
async def check_style_guide(request: StyleGuideCheckRequest) -> StyleGuideCheckResponse:
    """
    Check content against GOV.UK style guide rules.

    Performs both deterministic pattern matching and LLM-based validation
    to identify style guide violations, then generates a summary and corrected version.

    Args:
        request: StyleGuideCheckRequest with content and optional parameters

    Returns:
        StyleGuideCheckResponse with violations, summary, and fixed document

    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        logger.info(f"Checking content ({len(request.content)} characters)")

        # Run deterministic checks
        case_insensitive_violations = check_case_insensitive_rules(request.content, RULES)
        case_sensitive_violations = check_case_sensitive_rules(request.content, RULES)

        # Run LLM validation if not skipped
        llm_violations = []
        if not request.skip_llm:
            logger.info("Running LLM validation...")
            llm_model = request.llm_model or STYLE_GUIDE_LLM_MODEL
            llm_violations = await check_llm_validation_rules(
                request.content, RULES, llm_model=llm_model
            )
        else:
            logger.info("Skipping LLM validation")

        # Combine all violations
        violations = case_insensitive_violations + case_sensitive_violations + llm_violations

        logger.info(f"Found {len(violations)} total violations")

        # Generate summary and fixed document if violations found
        summary_result = None
        if violations:
            llm_model = request.llm_model or STYLE_GUIDE_LLM_MODEL
            summary_result = await generate_summary_and_fix(
                document=request.content,
                violations=violations,
                llm_model=llm_model,
                output_dir=SCRIPT_DIR,
            )

        # Format violations for response
        violation_details = []
        for v in violations:
            violation_details.append(
                ViolationDetail(
                    rule_id=v.get("rule_id", ""),
                    rule_title=v.get("rule_title", ""),
                    severity=v.get("severity", ""),
                    example=v.get("example", ""),
                    sentence=v.get("sentence", ""),
                    position=v.get("position", 0),
                )
            )

        return StyleGuideCheckResponse(
            violation_count=len(violations),
            violations=violation_details,
            summary=summary_result.get("summary") if summary_result else None,
            fixed_document=summary_result.get("fixed_document") if summary_result else None,
            violation_list=summary_result.get("violation_list") if summary_result else None,
        )

    except Exception as e:
        logger.error(f"Error checking style guide: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error checking style guide: {str(e)}") from e
