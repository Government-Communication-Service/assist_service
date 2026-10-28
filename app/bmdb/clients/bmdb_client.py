import httpx

from app.bmdb.constants import URL_BMDB_EDITION
from app.bmdb.schemas import LatestEdition


class BmdbClient:
    @staticmethod
    async def get_latest_edition() -> LatestEdition:
        client = httpx.AsyncClient()
        response = await client.get(URL_BMDB_EDITION)
        response.raise_for_status()
        data = response.json()
        return LatestEdition(
            version_number=data["version_number"],
            date_received=data["date_received"],
            latest_campaign_end_date=data["latest_campaign_end_date"],
            earliest_campaign_end_date=data["earliest_campaign_end_date"],
            n_campaigns=data["n_campaigns"],
            min_media_spend=data["min_media_spend"],
            max_media_spend=data["max_media_spend"],
        )
