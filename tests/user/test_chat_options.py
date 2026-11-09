from datetime import datetime, timezone

import pytest
from sqlalchemy import select, update

from app.api.endpoints import ENDPOINTS
from app.config import DEFAULT_CHAT_OPTIONS
from app.database.models import UserChatOptions
from app.database.table import async_db_session
from app.user import chat_options

api = ENDPOINTS()

pytestmark = [pytest.mark.e2e]


@pytest.fixture(autouse=True)
def record_user_chat_options(monkeypatch):
    monkeypatch.setattr(chat_options, "RECORD_USER_CHAT_OPTIONS", True)


@pytest.fixture(autouse=True)
def pinned_default_chat_options(monkeypatch):
    """Pin the defaults these tests are written against, whatever app/config.py sets."""
    for field, value in {"use_rag": True, "use_gov_uk_search_api": False, "use_smart_targets": False}.items():
        monkeypatch.setitem(chat_options.DEFAULT_CHAT_OPTIONS, field, value)


@pytest.fixture
def chat_api(async_client, async_http_requester, user_id):
    """Helpers for creating chats, sending follow-ups and reading chats and the user's chat options."""

    class ChatApi:
        async def create(self, **body) -> dict:
            return await async_http_requester(
                "create chat", async_client.post, api.chats(user_uuid=user_id), json={"query": "hello", **body}
            )

        async def follow_up(self, chat_uuid, **body) -> dict:
            return await async_http_requester(
                "follow-up", async_client.put, api.get_chat_item(user_id, chat_uuid), json={"query": "more", **body}
            )

        async def get_chat(self, chat_uuid) -> dict:
            return await async_http_requester("get chat", async_client.get, api.get_chat_item(user_id, chat_uuid))

        async def options(self) -> dict:
            return await async_http_requester("get chat options", async_client.get, api.user_chat_options(user_id))

    return ChatApi()


async def _option_rows(user_id: int) -> list[UserChatOptions]:
    async with async_db_session() as session:
        result = await session.execute(
            select(UserChatOptions).where(UserChatOptions.user_id == user_id).order_by(UserChatOptions.id)
        )
        return list(result.scalars().all())


def _rows_as_tuples(rows: list[UserChatOptions]) -> list[tuple]:
    return [(r.use_rag, r.use_gov_uk_search_api, r.use_smart_targets) for r in rows]


def _last_turn_text(mock_chat_model) -> str:
    """Text of the user message sent in the latest model call, including its <context> block."""
    content = mock_chat_model.requests[-1]["messages"][-1]["content"]
    return "\n".join(block["text"] for block in content if block.get("type") == "text")


async def test_new_user_gets_default_options(chat_api, user):
    assert await chat_api.options() == DEFAULT_CHAT_OPTIONS
    assert await _option_rows(user.id) == []


async def test_creating_a_chat_with_changed_options_records_only_the_changes(chat_api, user):
    response = await chat_api.create(use_rag=False, use_gov_uk_search_api=True)

    assert await chat_api.options() == {"use_rag": False, "use_gov_uk_search_api": True, "use_smart_targets": False}
    rows = await _option_rows(user.id)
    assert _rows_as_tuples(rows) == [(False, True, None)]
    assert rows[0].message_id is not None
    assert response["use_rag"] is False
    assert response["use_gov_uk_search_api"] is True


async def test_sending_the_current_options_adds_no_row(chat_api, user):
    await chat_api.create(**DEFAULT_CHAT_OPTIONS)

    assert await _option_rows(user.id) == []


async def test_omitted_options_use_the_users_current_options(chat_api, user, mock_chat_model):
    chat = await chat_api.create(use_gov_uk_search_api=True)

    await chat_api.follow_up(chat["uuid"])

    assert "- search_gov_uk: available" in _last_turn_text(mock_chat_model)
    assert len(await _option_rows(user.id)) == 1

    new_chat = await chat_api.create()

    assert new_chat["use_gov_uk_search_api"] is True
    assert "- search_gov_uk: available" in _last_turn_text(mock_chat_model)


async def test_changing_options_mid_chat(chat_api, user, mock_chat_model):
    chat = await chat_api.create(use_gov_uk_search_api=True)

    await chat_api.follow_up(chat["uuid"], use_gov_uk_search_api=False, use_smart_targets=True)

    assert "- search_gov_uk: turned off." in _last_turn_text(mock_chat_model)
    assert await chat_api.options() == {"use_rag": True, "use_gov_uk_search_api": False, "use_smart_targets": True}
    # use_rag was never set, so it stays None in both rows.
    assert _rows_as_tuples(await _option_rows(user.id)) == [(None, True, None), (None, False, True)]

    # The chat row holds the chat's current options.
    chat_details = await chat_api.get_chat(chat["uuid"])
    assert chat_details["use_gov_uk_search_api"] is False
    assert chat_details["use_smart_targets"] is True


async def test_changing_options_in_one_chat_leaves_other_chats_alone(chat_api, user, mock_chat_model):
    chat_a = await chat_api.create()
    chat_b = await chat_api.create()

    await chat_api.follow_up(chat_a["uuid"], use_gov_uk_search_api=True)
    await chat_api.follow_up(chat_b["uuid"])

    assert "- search_gov_uk: available" not in _last_turn_text(mock_chat_model)
    assert (await chat_api.get_chat(chat_b["uuid"]))["use_gov_uk_search_api"] is False
    assert (await chat_api.options())["use_gov_uk_search_api"] is True


async def test_sending_an_older_chats_options_keeps_choices_made_elsewhere(chat_api, user):
    chat_a = await chat_api.create()
    chat_b = await chat_api.create()
    await chat_api.follow_up(chat_b["uuid"], use_gov_uk_search_api=True)

    await chat_api.follow_up(chat_a["uuid"], **DEFAULT_CHAT_OPTIONS)

    assert _rows_as_tuples(await _option_rows(user.id)) == [(None, True, None)]
    assert (await chat_api.options())["use_gov_uk_search_api"] is True


async def test_nothing_is_recorded_while_recording_is_off(chat_api, user, mock_chat_model, monkeypatch):
    monkeypatch.setattr(chat_options, "RECORD_USER_CHAT_OPTIONS", False)

    await chat_api.create(use_gov_uk_search_api=True)

    assert "- search_gov_uk: available" in _last_turn_text(mock_chat_model)
    assert await _option_rows(user.id) == []
    assert await chat_api.options() == DEFAULT_CHAT_OPTIONS


async def test_changed_default_reaches_users_who_never_set_that_option(chat_api, monkeypatch):
    await chat_api.create(use_gov_uk_search_api=True)

    monkeypatch.setitem(chat_options.DEFAULT_CHAT_OPTIONS, "use_rag", False)

    assert await chat_api.options() == {"use_rag": False, "use_gov_uk_search_api": True, "use_smart_targets": False}


async def test_explicit_choice_survives_a_changed_default(chat_api, monkeypatch):
    chat = await chat_api.create(use_rag=False)
    await chat_api.follow_up(chat["uuid"], use_rag=True)

    monkeypatch.setitem(chat_options.DEFAULT_CHAT_OPTIONS, "use_rag", False)

    assert (await chat_api.options())["use_rag"] is True


async def test_reset_ignores_earlier_choices(chat_api, user, monkeypatch):
    await chat_api.create(use_rag=False, use_gov_uk_search_api=True)

    monkeypatch.setattr(chat_options, "CHAT_OPTIONS_RESET_AT", datetime.now(timezone.utc).replace(tzinfo=None))

    assert await chat_api.options() == DEFAULT_CHAT_OPTIONS

    await chat_api.create(use_smart_targets=True)

    # The row after the reset doesn't carry forward the choices made before it.
    assert _rows_as_tuples(await _option_rows(user.id)) == [(False, True, None), (None, None, True)]
    assert await chat_api.options() == {"use_rag": True, "use_gov_uk_search_api": False, "use_smart_targets": True}


async def test_soft_deleted_rows_are_ignored(chat_api, user):
    chat = await chat_api.create(use_gov_uk_search_api=True)
    await chat_api.follow_up(chat["uuid"], use_smart_targets=True)
    latest = (await _option_rows(user.id))[-1]

    async with async_db_session() as session:
        await session.execute(
            update(UserChatOptions).where(UserChatOptions.id == latest.id).values(deleted_at=datetime.now())
        )
        await session.commit()

    assert await chat_api.options() == {"use_rag": True, "use_gov_uk_search_api": True, "use_smart_targets": False}


async def test_chat_options_require_matching_user(async_client, async_http_requester, another_user_auth_session):
    other_user_id = "6f1c2a3e-5b4d-4c8e-9a7f-0e1d2c3b4a59"
    await another_user_auth_session(other_user_id)

    await async_http_requester(
        "get another user's chat options",
        async_client.get,
        api.user_chat_options(other_user_id),
        response_code=403,
    )
