from app.bmdb.clients import BmdbClient
from app.bmdb.exceptions import GetBenchmarkDatabaseEditionError
from app.bmdb.schemas import LatestEdition


class BmdbEditionService:
    @staticmethod
    async def get_latest_edition() -> LatestEdition:
        try:
            return await BmdbClient.get_latest_edition()
        except Exception as e:
            raise GetBenchmarkDatabaseEditionError(
                "Failed to get information about the latest edition of the Benchmark Database"
            ) from e
