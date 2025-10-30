from datetime import date

from pydantic import BaseModel


class LatestEdition(BaseModel):
    version_number: int
    date_received: date
    latest_campaign_end_date: date
    earliest_campaign_end_date: date
    n_campaigns: int
    min_media_spend: float | None
    max_media_spend: float | None
