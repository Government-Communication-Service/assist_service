from uuid import UUID

from pydantic import BaseModel


class AudienceSegmentSchema(BaseModel):
    uuid: UUID
    name: str
    pretty_name: str
    insights_markdown: str
    connect_url: str

    def wrap_for_context(self):
        prompt_context = f"<segment-insights segment-name:{self.pretty_name}>"
        prompt_context += f"{self.insights_markdown}"
        prompt_context += "</segment-insights>"
        return prompt_context
