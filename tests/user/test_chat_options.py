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

        async def set_options(self, **body) -> dict:
            return await async_http_requester(
                "set chat options", async_client.put, api.user_chat_options(user_id), json=body
            )

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


async def test_options_sent_with_a_new_chat_apply_to_that_chat_only(chat_api, user, mock_chat_model):
    response = await chat_api.create(use_rag=False, use_gov_uk_search_api=True)

    assert "- search_gov_uk: available" in _last_turn_text(mock_chat_model)
    assert response["use_rag"] is False
    assert response["use_gov_uk_search_api"] is True
    assert await chat_api.options() == DEFAULT_CHAT_OPTIONS
    assert await _option_rows(user.id) == []


async def test_omitted_options_use_the_users_current_options(chat_api, mock_chat_model):
    await chat_api.set_options(use_gov_uk_search_api=True)

    chat = await chat_api.create()

    assert chat["use_gov_uk_search_api"] is True
    assert "- search_gov_uk: available" in _last_turn_text(mock_chat_model)

    await chat_api.follow_up(chat["uuid"])

    assert "- search_gov_uk: available" in _last_turn_text(mock_chat_model)


async def test_changing_options_mid_chat_applies_to_that_message_only(chat_api, user, mock_chat_model):
    chat = await chat_api.create(use_gov_uk_search_api=True)

    await chat_api.follow_up(chat["uuid"], use_gov_uk_search_api=False, use_smart_targets=True)

    assert "- search_gov_uk: turned off." in _last_turn_text(mock_chat_model)
    assert await chat_api.options() == DEFAULT_CHAT_OPTIONS
    assert await _option_rows(user.id) == []


async def test_options_sent_in_one_chat_dont_reach_other_chats(chat_api, mock_chat_model):
    chat_a = await chat_api.create()
    chat_b = await chat_api.create()

    await chat_api.follow_up(chat_a["uuid"], use_gov_uk_search_api=True)
    await chat_api.follow_up(chat_b["uuid"])

    assert "- search_gov_uk: available" not in _last_turn_text(mock_chat_model)


async def test_an_existing_chat_shows_the_users_current_options(chat_api):
    chat = await chat_api.create()
    await chat_api.set_options(use_gov_uk_search_api=True, use_smart_targets=True)

    chat_details = await chat_api.get_chat(chat["uuid"])

    assert chat_details["use_gov_uk_search_api"] is True
    assert chat_details["use_smart_targets"] is True


async def test_a_shared_chat_shows_the_options_its_latest_message_used(
    async_client, async_http_requester, chat_api, user_id
):
    chat = await chat_api.create(use_gov_uk_search_api=True)
    await chat_api.set_options(use_smart_targets=True)
    share = await async_http_requester(
        "share chat", async_client.patch, api.chat_share(user_id, chat["uuid"]), json={"share": True}
    )

    shared = await async_http_requester("get shared chat", async_client.get, api.shared_chat(share["share_code"]))

    assert shared["use_gov_uk_search_api"] is True
    assert shared["use_smart_targets"] is False


async def test_setting_options_saves_only_the_ones_sent(chat_api, user, mock_chat_model):
    response = await chat_api.set_options(use_gov_uk_search_api=True)

    assert response == {"use_rag": True, "use_gov_uk_search_api": True, "use_smart_targets": False}
    rows = await _option_rows(user.id)
    assert _rows_as_tuples(rows) == [(None, True, None)]
    assert rows[0].message_id is None

    await chat_api.create()

    assert "- search_gov_uk: available" in _last_turn_text(mock_chat_model)


async def test_setting_the_current_options_adds_no_row(chat_api, user):
    await chat_api.set_options(**DEFAULT_CHAT_OPTIONS)
    await chat_api.set_options()

    assert await _option_rows(user.id) == []


async def test_nothing_is_recorded_while_recording_is_off(chat_api, user, mock_chat_model, monkeypatch):
    monkeypatch.setattr(chat_options, "RECORD_USER_CHAT_OPTIONS", False)

    await chat_api.create(use_gov_uk_search_api=True)

    assert "- search_gov_uk: available" in _last_turn_text(mock_chat_model)
    assert await chat_api.set_options(use_smart_targets=True) == DEFAULT_CHAT_OPTIONS
    assert await _option_rows(user.id) == []
    assert await chat_api.options() == DEFAULT_CHAT_OPTIONS


async def test_changed_default_reaches_users_who_never_set_that_option(chat_api, monkeypatch):
    await chat_api.set_options(use_gov_uk_search_api=True)

    monkeypatch.setitem(chat_options.DEFAULT_CHAT_OPTIONS, "use_rag", False)

    assert await chat_api.options() == {"use_rag": False, "use_gov_uk_search_api": True, "use_smart_targets": False}


async def test_explicit_choice_survives_a_changed_default(chat_api, monkeypatch):
    await chat_api.set_options(use_rag=False)
    await chat_api.set_options(use_rag=True)

    monkeypatch.setitem(chat_options.DEFAULT_CHAT_OPTIONS, "use_rag", False)

    assert (await chat_api.options())["use_rag"] is True


async def test_reset_ignores_earlier_choices(chat_api, user, monkeypatch):
    await chat_api.set_options(use_rag=False, use_gov_uk_search_api=True)

    monkeypatch.setattr(chat_options, "CHAT_OPTIONS_RESET_AT", datetime.now(timezone.utc).replace(tzinfo=None))

    assert await chat_api.options() == DEFAULT_CHAT_OPTIONS

    await chat_api.set_options(use_smart_targets=True)

    # The row after the reset doesn't carry forward the choices made before it.
    assert _rows_as_tuples(await _option_rows(user.id)) == [(False, True, None), (None, None, True)]
    assert await chat_api.options() == {"use_rag": True, "use_gov_uk_search_api": False, "use_smart_targets": True}


async def test_soft_deleted_rows_are_ignored(chat_api, user):
    await chat_api.set_options(use_gov_uk_search_api=True)
    await chat_api.set_options(use_smart_targets=True)
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
    await async_http_requester(
        "set another user's chat options",
        async_client.put,
        api.user_chat_options(other_user_id),
        json={"use_rag": False},
        response_code=403,
    )
