import json
import random
import time
from datetime import datetime
from pathlib import Path

from tasks import UserProtocol

# Relative path to data/ignored/load_testing/
_TEST_DATA = Path(__file__).parent.parent.parent.parent / "data" / "ignored" / "load_testing"

_AB_FILES_CANDIDATES = [
    ("great-campaign.pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    ("pre-school-parents.pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    ("ravening-hordes.txt", "text/plain"),
]
_AB_FILES = [(f, m) for f, m in _AB_FILES_CANDIDATES if (_TEST_DATA / f).exists()]
if not _AB_FILES:
    _AB_FILES = [("medium.txt", "text/plain")]

# Maps each test file to the segment names expected within it.
# great-campaign.pptx has 4 segments — we pick 1 at random per run.
_SEGMENT_MAP = {
    "great-campaign.pptx": ["culture cravers", "heritage hunters", "tech forwards", "safety seekers"],
    "pre-school-parents.pptx": ["pre-school parents"],
    "ravening-hordes.txt": ["ravening hordes"],
}

_POLL_INTERVAL = 5  # seconds between analyse retries
_MAX_POLLS = 24  # 2 minutes total

# Timestamped JSONL file for capturing analysis outputs during this run.
_OUTPUT_FILE = _TEST_DATA / f"analysis_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"


def _write_output(filename: str, segment_names: list[str], document_uuid: str, result: dict) -> None:
    record = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "segment_names": segment_names,
        "document_uuid": document_uuid,
        "result": result,
    }
    with open(_OUTPUT_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def upload_audience_document(user: UserProtocol) -> None:
    """Upload a document, randomly with or without is_audience_builder=True."""
    filename, mime = random.choices(_AB_FILES, k=1)[0]
    filepath = _TEST_DATA / filename
    file_bytes = filepath.read_bytes()

    use_audience_builder = random.random() < 0.5
    user.last_audience_filename = filename

    flag = str(use_audience_builder).lower()
    with user.client.post(
        f"/v1/users/{user.user_uuid}/documents?is_audience_builder={flag}",
        headers=user.auth_headers,
        files={"file": (filename, file_bytes, mime)},
        catch_response=True,
        name=f"POST /v1/users/{{uuid}}/documents?is_audience_builder={flag}",
    ) as resp:
        if resp.status_code == 200:
            doc_uuid = resp.json().get("document_uuid")
            if doc_uuid:
                user.audience_document_uuids.append(doc_uuid)
            resp.success()
        elif resp.status_code in (400, 422):
            resp.success()
        else:
            resp.failure(f"Unexpected {resp.status_code}: {resp.text[:500]}")


def analyse_segments(user: UserProtocol) -> None:
    """Analyse audience segments. Polls if documents are still processing."""
    if not user.audience_document_uuids:
        return

    filename = user.last_audience_filename or ""
    candidates = _SEGMENT_MAP.get(filename, ["general audience"])
    # Pick one segment at a time to keep requests cheap and responses readable
    segment_names = [random.choice(candidates)]

    doc_uuid = user.audience_document_uuids[-1]
    payload = {
        "segment_names": segment_names,
        "document_uuids": [doc_uuid],
    }

    for _attempt in range(_MAX_POLLS):
        with user.client.post(
            "/v1/audience-builder/analyse",
            headers=user.auth_headers,
            json=payload,
            catch_response=True,
            name="POST /v1/audience-builder/analyse",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Unexpected {resp.status_code}: {resp.text[:500]}")
                return

            data = resp.json()
            feedback = data.get("feedback", "")

            if "still being processed" in feedback:
                if _attempt < _MAX_POLLS - 1:
                    time.sleep(_POLL_INTERVAL)
                    continue
                resp.failure("Document never finished processing within polling timeout")
                user.last_analysis_result = None
                return
            if "An error occurred during analysis" in feedback:
                resp.failure("An error occurred during analysis")
                user.last_analysis_result = None
                return
            if "An error occurred during document processing" in feedback:
                resp.failure("Error in document processing")
                user.last_analysis_result = None
                return

            # Analysis completed — capture output to file
            user.last_analysis_result = data
            _write_output(filename, segment_names, doc_uuid, data)
            resp.success()
            return

    # Shouldn't reach this point
    user.last_analysis_result = None


def fetch_citations(user: UserProtocol) -> None:
    """Fetch citations for previously analysed segments."""
    if not user.last_analysis_result or not user.audience_document_uuids:
        return

    segments = []
    for seg in user.last_analysis_result.get("segments", []):
        profile = seg.get("profile")
        if profile:
            segments.append(
                {
                    "segment_name": seg["segment_name"],
                    "profile": profile,
                }
            )

    if not segments:
        return

    payload = {
        "segments": segments,
        "document_uuids": [user.audience_document_uuids[-1]],
    }

    with user.client.post(
        "/v1/audience-builder/citations",
        headers=user.auth_headers,
        json=payload,
        catch_response=True,
        name="POST /v1/audience-builder/citations",
    ) as resp:
        if resp.status_code == 200:
            resp.success()
        else:
            resp.failure(f"Unexpected {resp.status_code}: {resp.text[:500]}")


def save_segment(user: UserProtocol) -> None:
    """Save a segment profile. Non-fatal — GCS Data API may not be available."""
    if not user.last_analysis_result:
        return

    # Find the first segment with a profile
    profile = None
    for seg in user.last_analysis_result.get("segments", []):
        if seg.get("profile"):
            profile = seg["profile"]
            break

    if not profile:
        return

    with user.client.post(
        "/v1/audience-builder/save",
        headers=user.auth_headers,
        json={"profile": profile},
        catch_response=True,
        name="POST /v1/audience-builder/save",
    ) as resp:
        if resp.status_code == 200:
            resp.success()
        else:
            # Non-fatal — log but don't fail the pipeline
            resp.success()


def full_pipeline(user: UserProtocol) -> None:
    """Run the full audience builder pipeline: upload → analyse → citations (50%) → save."""
    upload_audience_document(user)

    # Wait for background processing to start before polling analyse
    time.sleep(30)

    analyse_segments(user)

    if user.last_analysis_result:
        # Citations pass is optional — run 50% of the time
        if random.random() < 0.5:
            fetch_citations(user)

        save_segment(user)

    # Reset for next pipeline run
    user.last_analysis_result = None
