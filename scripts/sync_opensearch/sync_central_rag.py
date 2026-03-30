# ruff: noqa: E402
import asyncio
import fcntl
import logging
import sys
import time
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

    # This script is invoked both on container startup and by `make sync-central-rag`.
    # Guard against concurrent runs that can interleave index delete/create and cause duplicate docs.
    lock_path = "/tmp/sync_central_rag.lock"
    with open(lock_path, "w") as lock_file:
        start = time.monotonic()
        while True:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as err:
                if time.monotonic() - start > 300:
                    raise TimeoutError("Timed out waiting for central RAG sync lock") from err
                time.sleep(1)

        asyncio.run(sync())
