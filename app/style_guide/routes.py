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
from app.style_guide.prompts import format_checker_context
from app.style_guide.service import describe_violation
from app.style_guide.style_guide_checker import (
    check_case_insensitive_rules,
    check_case_sensitive_rules,
    check_llm_validation_rules,
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

    rule_id: Optional[str] = Field(None, description="The rule's id in rule_mapping.json")
    rule_title: Optional[str] = Field(None, description="The rule's name")
    guidance: Optional[str] = Field(None, description="What the style guide says for this rule")
    detected_by: str = Field(..., description="'pattern match' or 'model review'")
    problem: Optional[str] = Field(None, description="What the pattern match found wrong")
    correct_form: Optional[str] = Field(None, description="The correct form, for casing rules")
    reason: Optional[str] = Field(None, description="Why the model judged it a violation")
    count: int = Field(..., description="How many times it occurs")
    examples: List[str] = Field(..., description="The sentences or text where it occurs")


class StyleGuideCheckResponse(BaseModel):
    """Response model for style guide check."""

    violation_count: int = Field(..., description="Total number of violations found")
    violations: List[ViolationDetail] = Field(..., description="List of all violations")


def _checker_context(request: StyleGuideCheckRequest) -> str | None:
    """Turn the request's audience and concerns into guidance for the validation prompts."""
    parts = []
    if request.target_audience:
        parts.append(f"Target audience: {request.target_audience}")
    if request.specific_concerns:
        parts.append(f"Specific concerns: {request.specific_concerns}")
    return "\n".join(parts) or None


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
    to identify style guide violations.

    Args:
        request: StyleGuideCheckRequest with content and optional parameters

    Returns:
        StyleGuideCheckResponse with the violations found

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
                request.content,
                RULES,
                llm_model=llm_model,
                checker_context=format_checker_context(_checker_context(request)),
            )
        else:
            logger.info("Skipping LLM validation")

        # Combine all violations
        violations = case_insensitive_violations + case_sensitive_violations + llm_violations

        logger.info(f"Found {len(violations)} total violations")

        violation_details = [ViolationDetail(**describe_violation(v)) for v in violations]

        return StyleGuideCheckResponse(
            violation_count=len(violations),
            violations=violation_details,
        )

    except Exception as e:
        logger.error(f"Error checking style guide: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error checking style guide: {str(e)}") from e
