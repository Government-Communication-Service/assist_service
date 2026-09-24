from collections.abc import AsyncIterator
from contextlib import AbstractContextManager, asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from anthropic.types import ToolUseBlock

from app.chat.schemas import ChatTitleRequest
from app.classification_and_title.prompts import TOOL_NAME_TITLE_AND_CLASSIFICATION
from app.classification_and_title.schemas import ClassificationInput
from app.classification_and_title.service import (
    TitleAndClassificationResult,
    _bulk_sync_classifications,
    bulk_sync_classifications,
    create_title_and_classification,
    get_classifications,
    update_chat_title,
)
from app.config import settings

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_classification(id_, title):
    classification = MagicMock()
    classification.id = id_
    classification.title = title
    return classification


def make_tool_use_block(title="Draft a press release", category="Media handling and press releases", **extra_input):
    block = MagicMock(spec=ToolUseBlock)
    block.name = TOOL_NAME_TITLE_AND_CLASSIFICATION
    block.input = {"title": title, "category": category, **extra_input}
    return block


def make_llm_response(blocks, llm_internal_response_id=99):
    response = MagicMock()
    response.content = blocks
    response.llm_internal_response_id = llm_internal_response_id
    return response


def make_chat(chat_id=1, user_id=1):
    chat = MagicMock()
    chat.id = chat_id
    chat.user_id = user_id
    return chat


def make_user_query_result(job_title=None):
    user = MagicMock()
    user.job_title = job_title
    result = MagicMock()
    result.scalar_one_or_none.return_value = user
    return result


def make_chat_row(title: str = "A new title") -> MagicMock:
    chat_row = MagicMock()
    now = datetime.now(timezone.utc)
    chat_row.uuid = uuid4()
    chat_row.title = title
    chat_row.created_at = now
    chat_row.updated_at = now
    return chat_row


# ---------------------------------------------------------------------------
# create_title_and_classification
# ---------------------------------------------------------------------------


class TestCreateTitleAndClassification:
    async def _run(self, llm_response=None, db_session=None, invoke_side_effect=None, max_attempts=2):
        db_session = db_session or AsyncMock()
        db_session.execute = AsyncMock(return_value=make_user_query_result())
        classifications = [
            make_classification(1, "Media handling and press releases"),
            make_classification(2, "Internal communications"),
        ]

        with (
            patch(
                "app.classification_and_title.service.DbOperations.get_classifications",
                new=AsyncMock(return_value=classifications),
            ),
            patch(
                "app.classification_and_title.service.DbOperations.fetch_undeleted_chat_documents",
                new=AsyncMock(return_value=[]),
            ),
            patch("app.classification_and_title.service.LLMTable") as mock_llm_table,
            patch("app.classification_and_title.service.BedrockHandler") as mock_bedrock,
            patch.object(settings, "classification_title_max_attempts", max_attempts),
        ):
            mock_llm_table.return_value.get_by_model.return_value = MagicMock()
            mock_bedrock_instance = MagicMock()
            mock_bedrock_instance.format_content_for_chat_title.return_value = [{"role": "user", "content": "hi"}]
            if invoke_side_effect is not None:
                mock_bedrock_instance.invoke_async = AsyncMock(side_effect=invoke_side_effect)
            else:
                mock_bedrock_instance.invoke_async = AsyncMock(return_value=llm_response)
            mock_bedrock.return_value = mock_bedrock_instance

            chat = make_chat()
            data = ChatTitleRequest(query="Please draft a press release about potholes")
            result = await create_title_and_classification(db_session, chat, data)
            return result, mock_bedrock_instance.invoke_async

    @pytest.mark.asyncio
    async def test_parses_title_category_task_type_and_discipline(self):
        block = make_tool_use_block(
            title="Pothole press release",
            category="Media handling and press releases",
            task_type="Drafting",
            discipline="Media",
        )
        result, _ = await self._run(make_llm_response([block]))

        assert result.title == "Pothole press release"
        assert result.classification_id == 1
        assert result.task_type == "Drafting"
        assert result.discipline == "Media"
        assert result.llm_internal_response_id == 99

    @pytest.mark.asyncio
    async def test_category_other_maps_to_no_classification_id(self):
        block = make_tool_use_block(category="Other", task_type="Research", discipline="Unknown")
        result, _ = await self._run(make_llm_response([block]))

        assert result.classification_id is None

    @pytest.mark.asyncio
    async def test_category_not_found_maps_to_no_classification_id(self):
        block = make_tool_use_block(category="Some unknown category", task_type="Research", discipline="Unknown")
        result, _ = await self._run(make_llm_response([block]))

        assert result.classification_id is None

    @pytest.mark.asyncio
    async def test_discipline_unknown_is_stored_as_unknown(self):
        block = make_tool_use_block(discipline="Unknown", task_type="Research")
        result, _ = await self._run(make_llm_response([block]))

        assert result.discipline == "Unknown"

    @pytest.mark.asyncio
    async def test_task_type_unknown_is_stored_as_unknown(self):
        block = make_tool_use_block(task_type="Unknown", discipline="Unknown")
        result, _ = await self._run(make_llm_response([block]))

        assert result.task_type == "Unknown"

    @pytest.mark.asyncio
    async def test_no_tool_use_block_returns_empty_title_after_exhausting_attempts(self):
        result, invoke_mock = await self._run(make_llm_response([MagicMock()]), max_attempts=2)

        assert result.title == ""
        assert result.classification_id is None
        assert result.task_type is None
        assert result.discipline is None
        assert invoke_mock.await_count == 2

    @pytest.mark.asyncio
    async def test_title_over_255_chars_is_truncated(self):
        block = make_tool_use_block(title="X" * 300, task_type="Drafting", discipline="Media")
        result, _ = await self._run(make_llm_response([block]))

        assert len(result.title) == 255
        assert result.title.endswith("...")

    @pytest.mark.asyncio
    async def test_bedrock_exception_exhausts_attempts_and_returns_empty_result(self):
        result, invoke_mock = await self._run(invoke_side_effect=RuntimeError("bedrock is down"), max_attempts=2)

        assert result.title == ""
        assert result.classification_id is None
        assert result.task_type is None
        assert result.discipline is None
        assert result.llm_internal_response_id is None
        assert invoke_mock.await_count == 2

    @pytest.mark.asyncio
    async def test_max_attempts_of_one_disables_retry(self):
        result, invoke_mock = await self._run(invoke_side_effect=RuntimeError("bedrock is down"), max_attempts=1)

        assert result.title == ""
        assert invoke_mock.await_count == 1

    @pytest.mark.asyncio
    async def test_retries_after_transient_exception_then_succeeds(self):
        block = make_tool_use_block(task_type="Drafting", discipline="Media")
        result, invoke_mock = await self._run(
            invoke_side_effect=[RuntimeError("bedrock is down"), make_llm_response([block])],
            max_attempts=2,
        )

        assert result.title == "Draft a press release"
        assert invoke_mock.await_count == 2

    @pytest.mark.asyncio
    async def test_retries_after_missing_tool_use_block_then_succeeds(self):
        block = make_tool_use_block(task_type="Drafting", discipline="Media")
        result, invoke_mock = await self._run(
            invoke_side_effect=[make_llm_response([MagicMock()]), make_llm_response([block])],
            max_attempts=2,
        )

        assert result.title == "Draft a press release"
        assert invoke_mock.await_count == 2


# ---------------------------------------------------------------------------
# update_chat_title
# ---------------------------------------------------------------------------


class TestUpdateChatTitle:
    """update_chat_title opens three transactions: generate, title, classification mapping."""

    def _patch_sessions(self) -> tuple[AbstractContextManager, list[AsyncMock]]:
        """Patch async_db_session with one AsyncMock session per block, recording rollbacks."""
        sessions = []

        @asynccontextmanager
        async def fake_async_db_session() -> AsyncIterator[AsyncMock]:
            session = AsyncMock()
            session.rolled_back = False
            sessions.append(session)
            try:
                yield session
            except Exception:
                session.rolled_back = True
                raise

        return patch("app.classification_and_title.service.async_db_session", new=fake_async_db_session), sessions

    def _make_result(
        self,
        title: str = "A new title",
        classification_id: int | None = 1,
        task_type: str | None = "Drafting",
        discipline: str | None = "Media",
    ) -> TitleAndClassificationResult:
        return TitleAndClassificationResult(
            title=title,
            classification_id=classification_id,
            task_type=task_type,
            discipline=discipline,
            llm_internal_response_id=99,
        )

    def _make_data(self) -> MagicMock:
        data = MagicMock()
        data.to_dict.return_value = {"query": "hello"}
        return data

    @pytest.mark.asyncio
    async def test_persists_title_and_mapping_in_separate_transactions(self) -> None:
        session_patch, sessions = self._patch_sessions()
        with (
            session_patch,
            patch(
                "app.classification_and_title.service.create_title_and_classification",
                new=AsyncMock(return_value=self._make_result()),
            ) as mock_create,
            patch(
                "app.classification_and_title.service.DbOperations.chat_update_title",
                new=AsyncMock(return_value=make_chat_row()),
            ) as mock_update_title,
            patch(
                "app.classification_and_title.service._insert_chat_classification_mapping",
                new=AsyncMock(),
            ) as mock_insert_mapping,
        ):
            chat = make_chat(chat_id=7)
            response = await update_chat_title(chat, self._make_data())

        assert len(sessions) == 3
        generate_session, title_session, mapping_session = sessions
        assert mock_create.call_args.args[0] is generate_session
        assert mock_update_title.call_args.args[0] is title_session
        mock_insert_mapping.assert_awaited_once_with(
            db_session=mapping_session,
            chat_id=7,
            classification_id=1,
            task_type="Drafting",
            discipline="Media",
            llm_internal_response_id=99,
        )
        assert not any(s.rolled_back for s in sessions)
        assert response.title == "A new title"

    @pytest.mark.asyncio
    async def test_chat_update_title_failure_propagates_and_skips_mapping(self) -> None:
        session_patch, sessions = self._patch_sessions()
        with (
            session_patch,
            patch(
                "app.classification_and_title.service.create_title_and_classification",
                new=AsyncMock(return_value=self._make_result()),
            ),
            patch(
                "app.classification_and_title.service.DbOperations.chat_update_title",
                new=AsyncMock(side_effect=RuntimeError("db is down")),
            ),
            patch(
                "app.classification_and_title.service._insert_chat_classification_mapping",
                new=AsyncMock(),
            ) as mock_insert_mapping,
        ):
            with pytest.raises(RuntimeError, match="db is down"):
                await update_chat_title(make_chat(), self._make_data())

        mock_insert_mapping.assert_not_awaited()
        generate_session, title_session = sessions
        assert not generate_session.rolled_back
        assert title_session.rolled_back

    @pytest.mark.asyncio
    async def test_mapping_insert_failure_rolls_back_only_its_own_transaction(self) -> None:
        session_patch, sessions = self._patch_sessions()
        with (
            session_patch,
            patch(
                "app.classification_and_title.service.create_title_and_classification",
                new=AsyncMock(return_value=self._make_result()),
            ),
            patch(
                "app.classification_and_title.service.DbOperations.chat_update_title",
                new=AsyncMock(return_value=make_chat_row()),
            ),
            patch(
                "app.classification_and_title.service._insert_chat_classification_mapping",
                new=AsyncMock(side_effect=RuntimeError("insert failed")),
            ),
        ):
            response = await update_chat_title(make_chat(), self._make_data())

        assert response.title == "A new title"
        generate_session, title_session, mapping_session = sessions
        assert not generate_session.rolled_back
        assert not title_session.rolled_back
        assert mapping_session.rolled_back

    @pytest.mark.asyncio
    async def test_empty_title_skips_title_write_but_still_writes_mapping(self) -> None:
        session_patch, sessions = self._patch_sessions()
        with (
            session_patch,
            patch(
                "app.classification_and_title.service.create_title_and_classification",
                new=AsyncMock(
                    return_value=self._make_result(title="", classification_id=None, task_type=None, discipline=None)
                ),
            ),
            patch(
                "app.classification_and_title.service.DbOperations.chat_update_title",
                new=AsyncMock(),
            ) as mock_update_title,
            patch(
                "app.classification_and_title.service._insert_chat_classification_mapping",
                new=AsyncMock(),
            ) as mock_insert_mapping,
        ):
            response = await update_chat_title(make_chat_row(title="Existing title"), self._make_data())

        mock_update_title.assert_not_awaited()
        mock_insert_mapping.assert_awaited_once()
        assert len(sessions) == 2
        assert response.title == "Existing title"


# ---------------------------------------------------------------------------
# get_classifications
# ---------------------------------------------------------------------------


class TestGetClassifications:
    @pytest.mark.asyncio
    async def test_returns_classifications_response(self):
        row = MagicMock()
        row.uuid = "11111111-1111-1111-1111-111111111111"
        row.title = "Media handling and press releases"
        row.description = "Handles media."

        with patch(
            "app.classification_and_title.service.DbOperations.get_classifications",
            new=AsyncMock(return_value=[row]),
        ):
            response = await get_classifications(AsyncMock())

        assert len(response) == 1
        assert response[0]["title"] == "Media handling and press releases"
        assert response[0]["description"] == "Handles media."


# ---------------------------------------------------------------------------
# bulk_sync_classifications / _bulk_sync_classifications
# ---------------------------------------------------------------------------


class TestBulkSyncClassifications:
    @pytest.mark.asyncio
    async def test_wraps_and_logs_exception_from_internal_sync(self):
        with patch(
            "app.classification_and_title.service._bulk_sync_classifications",
            new=AsyncMock(side_effect=RuntimeError("db is down")),
        ):
            with pytest.raises(RuntimeError, match="db is down"):
                await bulk_sync_classifications(AsyncMock(), [])

    @pytest.mark.asyncio
    async def test_new_title_is_created(self):
        db_session = AsyncMock()
        existing_result = MagicMock()
        existing_result.scalars.return_value.all.return_value = []
        db_session.execute = AsyncMock(return_value=existing_result)

        inputs = [ClassificationInput(title="Brand new category", description="A new one")]
        result = await _bulk_sync_classifications(db_session, inputs)

        assert result["created"] == 1
        assert result["updated"] == 0
        assert result["deprecated"] == 0

    @pytest.mark.asyncio
    async def test_existing_title_with_changed_description_is_updated(self):
        existing_row = MagicMock()
        existing_row.id = 1
        existing_row.title = "Media handling and press releases"
        existing_row.description = "Old description"
        existing_row.deleted_at = None

        db_session = AsyncMock()
        existing_result = MagicMock()
        existing_result.scalars.return_value.all.return_value = [existing_row]
        db_session.execute = AsyncMock(return_value=existing_result)

        inputs = [ClassificationInput(title="Media handling and press releases", description="New description")]
        result = await _bulk_sync_classifications(db_session, inputs)

        assert result["updated"] == 1
        assert result["created"] == 0
        assert result["deprecated"] == 0

    @pytest.mark.asyncio
    async def test_soft_deleted_title_is_revived(self):
        existing_row = MagicMock()
        existing_row.id = 1
        existing_row.title = "Media handling and press releases"
        existing_row.description = "Same description"
        existing_row.deleted_at = datetime.now(timezone.utc)

        db_session = AsyncMock()
        existing_result = MagicMock()
        existing_result.scalars.return_value.all.return_value = [existing_row]
        db_session.execute = AsyncMock(return_value=existing_result)

        inputs = [ClassificationInput(title="Media handling and press releases", description="Same description")]
        result = await _bulk_sync_classifications(db_session, inputs)

        assert result["created"] == 0
        assert result["updated"] == 0
        assert db_session.execute.await_count == 2  # initial select + revive update

    @pytest.mark.asyncio
    async def test_title_absent_from_incoming_is_deprecated(self):
        existing_row = MagicMock()
        existing_row.id = 1
        existing_row.title = "Recruitment"
        existing_row.description = None
        existing_row.deleted_at = None

        db_session = AsyncMock()
        existing_result = MagicMock()
        existing_result.scalars.return_value.all.return_value = [existing_row]

        mapping_check_result = MagicMock()
        mapping_check_result.all.return_value = []

        db_session.execute = AsyncMock(side_effect=[existing_result, mapping_check_result, MagicMock()])

        result = await _bulk_sync_classifications(db_session, [])

        assert result["deprecated"] == 1
        assert result["created"] == 0
        assert result["updated"] == 0
