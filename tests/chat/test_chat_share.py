import logging
import uuid

import pytest
from sqlalchemy.future import select

from app.database.models import Chat
from app.database.table import async_db_session

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.chat_share,
]


class TestChatShare:
    @pytest.mark.asyncio
    async def test_enable_chat_share_generates_share_code(self, chat, user_id, async_client, async_http_requester):
        """Test enabling chat share generates a share_code"""
        endpoint = f"/v1/chats/users/{user_id}/chats/{chat.uuid}/share"

        response = await async_http_requester("enable chat share", async_client.patch, endpoint, json={"share": True})

        assert response["status"] == "success"
        assert "uuid" in response
        assert "title" in response
        assert "share_code" in response
        assert response["share_code"] is not None
        assert len(response["share_code"]) == 10

    @pytest.mark.asyncio
    async def test_disable_chat_share_preserves_share_code(
        self, chat, user_id, async_client, async_http_requester, db_session
    ):
        """Test disabling chat share preserves the existing share_code"""
        endpoint = f"/v1/chats/users/{user_id}/chats/{chat.uuid}/share"

        # Enable sharing first
        enable_response = await async_http_requester(
            "enable chat share", async_client.patch, endpoint, json={"share": True}
        )
        original_share_code = enable_response["share_code"]

        # Disable sharing
        disable_response = await async_http_requester(
            "disable chat share", async_client.patch, endpoint, json={"share": False}
        )

        assert disable_response["status"] == "success"
        assert disable_response["share_code"] == original_share_code

    @pytest.mark.asyncio
    async def test_share_code_persists_across_multiple_enables(self, chat, user_id, async_client, async_http_requester):
        """Test that share_code is only generated once and persists"""
        endpoint = f"/v1/chats/users/{user_id}/chats/{chat.uuid}/share"

        # Enable sharing first time
        first_response = await async_http_requester("first enable", async_client.patch, endpoint, json={"share": True})
        first_share_code = first_response["share_code"]

        # Disable and re-enable
        await async_http_requester("disable", async_client.patch, endpoint, json={"share": False})

        second_response = await async_http_requester(
            "second enable", async_client.patch, endpoint, json={"share": True}
        )
        second_share_code = second_response["share_code"]

        assert first_share_code == second_share_code

    @pytest.mark.asyncio
    async def test_share_code_is_unique_per_chat(self, user_id, async_client, async_http_requester):
        """Test that different chats get different share_codes"""
        # Create two chats
        create_url = f"/v1/chats/users/{user_id}"
        chat1_response = await async_http_requester(
            "create chat 1", async_client.post, create_url, json={"query": "test 1", "use_rag": False}
        )
        chat2_response = await async_http_requester(
            "create chat 2", async_client.post, create_url, json={"query": "test 2", "use_rag": False}
        )

        # Enable sharing for both
        share_endpoint_1 = f"/v1/chats/users/{user_id}/chats/{chat1_response['uuid']}/share"
        share_endpoint_2 = f"/v1/chats/users/{user_id}/chats/{chat2_response['uuid']}/share"

        share1_response = await async_http_requester(
            "share chat 1", async_client.patch, share_endpoint_1, json={"share": True}
        )
        share2_response = await async_http_requester(
            "share chat 2", async_client.patch, share_endpoint_2, json={"share": True}
        )

        assert share1_response["share_code"] != share2_response["share_code"]

    @pytest.mark.asyncio
    async def test_update_chat_share_default_false(self, chat, user_id, async_client, async_http_requester):
        """Test updating chat share with empty body defaults to false"""
        endpoint = f"/v1/chats/users/{user_id}/chats/{chat.uuid}/share"

        response = await async_http_requester("update chat share with default", async_client.patch, endpoint, json={})

        assert response["status"] == "success"
        assert "uuid" in response
        assert "title" in response

    @pytest.mark.asyncio
    async def test_update_chat_share_unauthorized(self, chat, async_client, async_http_requester):
        """Test updating chat share for non-existent user returns 403"""
        non_existent_user = str(uuid.uuid4())
        endpoint = f"/v1/chats/users/{non_existent_user}/chats/{chat.uuid}/share"

        await async_http_requester(
            "update chat share unauthorized",
            async_client.patch,
            endpoint,
            json={"share": True},
            response_code=403,
        )

    @pytest.mark.asyncio
    async def test_share_field_persists_in_database(self, chat, user_id, async_client, async_http_requester):
        """Test that share field is correctly persisted in database"""
        endpoint = f"/v1/chats/users/{user_id}/chats/{chat.uuid}/share"

        # Enable sharing
        await async_http_requester("enable share", async_client.patch, endpoint, json={"share": True})

        # Verify in database
        async with async_db_session() as db_session:
            result = await db_session.execute(select(Chat).filter(Chat.uuid == chat.uuid))
            db_chat = result.scalar_one()
            assert db_chat.share is True
            assert db_chat.share_code is not None

    @pytest.mark.asyncio
    async def test_share_code_in_chat_list_response(self, chat, user_id, async_client, async_http_requester):
        """Test that share_code appears in chat list responses"""
        # Enable sharing
        share_endpoint = f"/v1/chats/users/{user_id}/chats/{chat.uuid}/share"
        share_response = await async_http_requester(
            "enable share", async_client.patch, share_endpoint, json={"share": True}
        )
        share_code = share_response["share_code"]

        # Get chat list
        list_endpoint = f"/v1/chats/users/{user_id}/chats"
        list_response = await async_http_requester("get chat list", async_client.get, list_endpoint)

        # Find our chat in the list
        our_chat = next((c for c in list_response["chats"] if c["uuid"] == str(chat.uuid)), None)
        assert our_chat is not None
        assert our_chat["share_code"] == share_code


class TestSharedChatAccess:
    @pytest.mark.asyncio
    async def test_access_shared_chat_with_valid_code(
        self, chat, user_id, async_client, async_http_requester, another_user_auth_session, auth_token
    ):
        """Test that authenticated user can access shared chat via share_code"""
        # Enable sharing
        share_endpoint = f"/v1/chats/users/{user_id}/chats/{chat.uuid}/share"
        share_response = await async_http_requester(
            "enable share", async_client.patch, share_endpoint, json={"share": True}
        )
        share_code = share_response["share_code"]

        # Access as different user
        other_user_uuid = str(uuid.uuid4())
        other_session = await another_user_auth_session(other_user_uuid)

        from app.auth.constants import AUTH_TOKEN_ALIAS, SESSION_AUTH_ALIAS, USER_KEY_UUID_ALIAS

        other_user_headers = {
            USER_KEY_UUID_ALIAS: other_user_uuid,
            SESSION_AUTH_ALIAS: other_session,
            AUTH_TOKEN_ALIAS: auth_token,
        }

        shared_endpoint = f"/v1/chats/shared/{share_code}"
        response = await async_http_requester(
            "access shared chat", async_client.get, shared_endpoint, headers=other_user_headers
        )

        assert response["uuid"] == str(chat.uuid)
        assert "messages" in response

    @pytest.mark.asyncio
    async def test_access_shared_chat_with_invalid_code(
        self, async_client, async_http_requester, another_user_auth_session, auth_token
    ):
        """Test that invalid share_code returns 404"""
        other_user_uuid = str(uuid.uuid4())
        other_session = await another_user_auth_session(other_user_uuid)

        from app.auth.constants import AUTH_TOKEN_ALIAS, SESSION_AUTH_ALIAS, USER_KEY_UUID_ALIAS

        other_user_headers = {
            USER_KEY_UUID_ALIAS: other_user_uuid,
            SESSION_AUTH_ALIAS: other_session,
            AUTH_TOKEN_ALIAS: auth_token,
        }

        invalid_code = "invalid123"
        shared_endpoint = f"/v1/chats/shared/{invalid_code}"

        await async_http_requester(
            "access with invalid code",
            async_client.get,
            shared_endpoint,
            headers=other_user_headers,
            response_code=404,
        )

    @pytest.mark.asyncio
    async def test_access_unshared_chat_returns_403(
        self, chat, user_id, async_client, async_http_requester, another_user_auth_session, auth_token
    ):
        """Test that accessing a chat with share=false returns 403"""
        # Enable sharing first to get share_code
        share_endpoint = f"/v1/chats/users/{user_id}/chats/{chat.uuid}/share"
        share_response = await async_http_requester(
            "enable share", async_client.patch, share_endpoint, json={"share": True}
        )
        share_code = share_response["share_code"]

        # Disable sharing
        await async_http_requester("disable share", async_client.patch, share_endpoint, json={"share": False})

        # Try to access as different user
        other_user_uuid = str(uuid.uuid4())
        other_session = await another_user_auth_session(other_user_uuid)

        from app.auth.constants import AUTH_TOKEN_ALIAS, SESSION_AUTH_ALIAS, USER_KEY_UUID_ALIAS

        other_user_headers = {
            USER_KEY_UUID_ALIAS: other_user_uuid,
            SESSION_AUTH_ALIAS: other_session,
            AUTH_TOKEN_ALIAS: auth_token,
        }

        shared_endpoint = f"/v1/chats/shared/{share_code}"
        await async_http_requester(
            "access unshared chat",
            async_client.get,
            shared_endpoint,
            headers=other_user_headers,
            response_code=403,
        )

    @pytest.mark.asyncio
    async def test_shared_chat_requires_authentication(self, chat, user_id, async_client, async_http_requester):
        """Test that accessing shared chat without authentication fails"""
        # Enable sharing
        share_endpoint = f"/v1/chats/users/{user_id}/chats/{chat.uuid}/share"
        share_response = await async_http_requester(
            "enable share", async_client.patch, share_endpoint, json={"share": True}
        )
        share_code = share_response["share_code"]

        # Try to access without auth headers
        shared_endpoint = f"/v1/chats/shared/{share_code}"
        response = await async_client.get(shared_endpoint, headers={})

        assert response.status_code == 401
