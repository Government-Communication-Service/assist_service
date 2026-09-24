"""
Sync the chat classification taxonomy to a target API environment.

Install uv at https://docs.astral.sh/uv/getting-started/installation/

Usage:
    uv run scripts/classification/bulk_upload_classifications.py

    # Target a different environment:
    uv run scripts/classification/bulk_upload_classifications.py --base-url https://dev.api.example.com

    # Use a different classifications file:
    uv run scripts/classification/bulk_upload_classifications.py --file data/classifications/custom.json

Auth is read from AUTH_SECRET_KEY in the environment or .env file.

The JSON file must be an array of objects with "title" and optional "description" keys.
The endpoint applies the following rules:
  - New titles are created.
  - Existing titles with a changed description are updated.
  - Titles absent from the list are deprecated (soft-deleted).
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pydantic-settings",
#   "requests",
# ]
# ///

import argparse
import json
import logging
from pathlib import Path

import requests
from pydantic import SecretStr
from pydantic_settings import BaseSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    auth_secret_key: SecretStr
    base_url: str = "http://localhost:5312"
    file_path: str = "data/classifications/classifications.json"

    model_config = {"env_file": ".env", "extra": "ignore"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", dest="base_url", default=None, help="Target API base URL")
    parser.add_argument("--file", dest="file_path", default=None, help="Path to the classifications JSON file")
    return parser.parse_args()


def load_classifications(path: str) -> list:
    data = json.loads(Path(path).read_text())
    logger.info(f"Loaded {len(data)} classifications from {path}")
    return data


def upload_classifications(classifications: list, settings: Settings) -> dict:
    url = f"{settings.base_url}/v1/classifications/bulk"
    response = requests.post(
        url,
        headers={
            "Auth-Token": settings.auth_secret_key.get_secret_value(),
            "Content-Type": "application/json",
        },
        json=classifications,
    )
    response.raise_for_status()
    result = response.json()
    logger.info(
        f"Sync complete — created: {result['created']}, "
        f"updated: {result['updated']}, deprecated: {result['deprecated']}"
    )
    return result


if __name__ == "__main__":
    logger.info("=== Starting classification sync ===")
    args = parse_args()
    overrides = {k: v for k, v in {"base_url": args.base_url, "file_path": args.file_path}.items() if v is not None}
    settings = Settings(**overrides)
    try:
        classifications = load_classifications(settings.file_path)
        upload_classifications(classifications, settings)
        logger.info("=== Classification sync complete ===")
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        raise
