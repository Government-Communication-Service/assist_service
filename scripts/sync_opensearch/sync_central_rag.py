# ruff: noqa: E402
import asyncio
import logging
import sys
from pathlib import Path

# Add the root project directory to Python path at runtime
# This allows the script to import app modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from app.central_guidance.service_index import sync_central_index
from app.database.db_session import async_db_session

logger = logging.getLogger(__name__)
if __name__ == "__main__":

    async def sync():
        async with async_db_session() as db_session:
            await sync_central_index(db_session)

    asyncio.run(sync())
