from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field


class DocumentBlacklistStatus(str, Enum):
    OK = "ok"
    BLACKLISTED = "blacklisted"


class NonRagDocument(BaseModel):
    url: str
    title: str
    body: str
    status: DocumentBlacklistStatus


class SearchCost(BaseModel):
    total_cost: Decimal = Field(default_factory=Decimal)
    search_tool_cost: Decimal = Field(default_factory=Decimal)
    relevancy_assessment_cost: Decimal = Field(default_factory=Decimal)
    relevancy_assessment_count: int = 0
