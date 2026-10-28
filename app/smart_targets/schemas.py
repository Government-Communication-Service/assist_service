from uuid import UUID

from pydantic import BaseModel


class Metric(BaseModel):
    name: str
    type: str
    unit: str
    uuid: UUID
    id: int


class SelectedFilter(BaseModel):
    name: str
    values: list[str] | dict


class SelectedFilters(BaseModel):
    reasoning: str
    filters: list[SelectedFilter]
