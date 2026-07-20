import logging
import uuid
from unittest.mock import patch

import pytest
from sqlalchemy.future import select

from app.auth.constants import AUTH_TOKEN_ALIAS, SESSION_AUTH_ALIAS, USER_KEY_UUID_ALIAS
from app.database.db_operations import DbOperations
from app.database.models import Chat
from app.database.table import async_db_session

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.chat_share,
]


def share_endpoint(user_id, chat_uuid):
    return f"/v1/chats/users/{user_id}/chats/{chat_uuid}/share"


def share_users_endpoint(user_id, chat_uuid):
    return f"/v1/chats/users/{user_id}/chats/{chat_uuid}/share/users"


def share_user_endpoint(user_id, chat_uuid, shared_user_uuid):
    return f"/v1/chats/users/{user_id}/chats/{chat_uuid}/share/users/{shared_user_uuid}"


class TestPrivateChatShare:
    @pytest.mark.asyncio
    async def test_enable_private_chat_share(self, chat, user_id, async_client, async_http_requester):
        """Enabling a private share generates a share_code and sets share_private"""
        response = await async_http_requester(
            "enable private chat share",
            async_client.patch,
            share_endpoint(user_id, chat.uuid),
            json={"share": True, "share_private": True},
        )

        assert response["status"] == "success"
        assert response["share_code"] is not None
        assert len(response["share_code"]) == 10
        assert response["share_private"] is True

    @pytest.mark.asyncio
    async def test_public_share_defaults_to_not_private(self, chat, user_id, async_client, async_http_requester):
        """Enabling a share without share_private keeps the existing public behaviour"""
        response = await async_http_requester(
            "enable public chat share",
            async_client.patch,
            share_endpoint(user_id, chat.uuid),
            json={"share": True},
        )

        assert response["status"] == "success"
        assert response["share_private"] is False

    @pytest.mark.asyncio
    async def test_share_private_persists_in_database(self, chat, user_id, async_client, async_http_requester):
        """share_private is correctly persisted in the database"""
        await async_http_requester(
            "enable private share",
            async_client.patch,
            share_endpoint(user_id, chat.uuid),
            json={"share": True, "share_private": True},
        )

        async with async_db_session() as db_session:
            result = await db_session.execute(select(Chat).filter(Chat.uuid == chat.uuid))
            db_chat = result.scalar_one()
            assert db_chat.share is True
            assert db_chat.share_private is True

    @pytest.mark.asyncio
    async def test_legacy_share_toggle_preserves_share_private(self, chat, user_id, async_client, async_http_requester):
        """Toggling share with a legacy body (no share_private key) does not reset share_private"""
        endpoint = share_endpoint(user_id, chat.uuid)

        await async_http_requester(
            "enable private share", async_client.patch, endpoint, json={"share": True, "share_private": True}
        )

        # A client that predates private shares toggles sharing off and on again
        await async_http_requester("legacy disable", async_client.patch, endpoint, json={"share": False})
        response = await async_http_requester("legacy enable", async_client.patch, endpoint, json={"share": True})

        assert response["share_private"] is True

    @pytest.mark.asyncio
    async def test_switch_private_share_to_public(self, chat, user_id, async_client, async_http_requester):
        """The owner can explicitly switch a private share back to public"""
        endpoint = share_endpoint(user_id, chat.uuid)

        await async_http_requester(
            "enable private share", async_client.patch, endpoint, json={"share": True, "share_private": True}
        )
        response = await async_http_requester(
            "switch to public share", async_client.patch, endpoint, json={"share": True, "share_private": False}
        )

        assert response["share_private"] is False


class TestPrivateChatShareUserManagement:
    @pytest.mark.asyncio
    async def test_add_shared_user(self, chat, user_id, async_client, async_http_requester):
        """The owner can add a user's uuid to the private share"""
        shared_user_uuid = str(uuid.uuid4())

        response = await async_http_requester(
            "add shared user",
            async_client.post,
            share_users_endpoint(user_id, chat.uuid),
            json={"shared_user_uuid": shared_user_uuid},
        )

        assert response["status"] == "success"
        assert response["uuid"] == str(chat.uuid)
        assert shared_user_uuid in response["shared_user_uuids"]

    @pytest.mark.asyncio
    async def test_add_shared_user_is_idempotent(self, chat, user_id, async_client, async_http_requester):
        """Adding the same user twice does not create duplicates"""
        shared_user_uuid = str(uuid.uuid4())
        endpoint = share_users_endpoint(user_id, chat.uuid)

        await async_http_requester(
            "add shared user", async_client.post, endpoint, json={"shared_user_uuid": shared_user_uuid}
        )
        response = await async_http_requester(
            "add shared user again", async_client.post, endpoint, json={"shared_user_uuid": shared_user_uuid}
        )

        assert response["shared_user_uuids"].count(shared_user_uuid) == 1

    @pytest.mark.asyncio
    async def test_add_shared_user_survives_concurrent_insert_race(self, chat):
        """Under concurrency the pre-check can miss a row a racing request is inserting.
        add_chat_share_user must not raise on the resulting unique-constraint conflict: it
        relies on ON CONFLICT DO NOTHING and falls back to the mapping the winner created,
        staying idempotent instead of surfacing a 500."""
        shared_user_uuid = str(uuid.uuid4())

        # A racing request has already added the user; this transaction commits on block exit.
        async with async_db_session() as session:
            db_chat = (await session.execute(select(Chat).where(Chat.uuid == chat.uuid))).scalar_one()
            shared_user = await DbOperations.upsert_user_by_uuid(session, shared_user_uuid)
            winner = await DbOperations.add_chat_share_user(session, db_chat, shared_user)
            winner_id = winner.id

        # Fresh transaction: simulate our pre-check having run before the winner's insert, so it
        # returns None the first time; the fallback lookup after the conflict finds the winner.
        original_lookup = DbOperations.get_chat_share_user_mapping
        calls = {"n": 0}

        async def lookup_missing_first(db_session, chat_id, user_id):
            calls["n"] += 1
            if calls["n"] == 1:
                return None
            return await original_lookup(db_session, chat_id, user_id)

        async with async_db_session() as session:
            db_chat = (await session.execute(select(Chat).where(Chat.uuid == chat.uuid))).scalar_one()
            shared_user = await DbOperations.upsert_user_by_uuid(session, shared_user_uuid)
            with patch.object(DbOperations, "get_chat_share_user_mapping", new=lookup_missing_first):
                result = await DbOperations.add_chat_share_user(session, db_chat, shared_user)

            assert result is not None
            assert result.id == winner_id
            assert result.chat_id == db_chat.id
            assert result.user_id == shared_user.id

    @pytest.mark.asyncio
    async def test_add_shared_user_malformed_uuid(self, chat, user_id, async_client, async_http_requester):
        """Adding a malformed uuid returns 400"""
        await async_http_requester(
            "add malformed shared user",
            async_client.post,
            share_users_endpoint(user_id, chat.uuid),
            json={"shared_user_uuid": "not-a-uuid"},
            response_code=400,
        )

    @pytest.mark.asyncio
    async def test_add_owner_uuid_rejected(self, chat, user_id, async_client, async_http_requester):
        """Adding the chat owner's own uuid returns 400"""
        await async_http_requester(
            "add owner as shared user",
            async_client.post,
            share_users_endpoint(user_id, chat.uuid),
            json={"shared_user_uuid": user_id},
            response_code=400,
        )

    @pytest.mark.asyncio
    async def test_list_shared_users(self, chat, user_id, async_client, async_http_requester):
        """The owner can list the users added to the private share"""
        endpoint = share_users_endpoint(user_id, chat.uuid)
        first_uuid = str(uuid.uuid4())
        second_uuid = str(uuid.uuid4())

        await async_http_requester(
            "add first shared user", async_client.post, endpoint, json={"shared_user_uuid": first_uuid}
        )
        await async_http_requester(
            "add second shared user", async_client.post, endpoint, json={"shared_user_uuid": second_uuid}
        )

        response = await async_http_requester("list shared users", async_client.get, endpoint)

        assert response["shared_user_uuids"] == [first_uuid, second_uuid]

    @pytest.mark.asyncio
    async def test_remove_shared_user(self, chat, user_id, async_client, async_http_requester):
        """The owner can remove a user from the private share"""
        shared_user_uuid = str(uuid.uuid4())

        await async_http_requester(
            "add shared user",
            async_client.post,
            share_users_endpoint(user_id, chat.uuid),
            json={"shared_user_uuid": shared_user_uuid},
        )
        response = await async_http_requester(
            "remove shared user",
            async_client.delete,
            share_user_endpoint(user_id, chat.uuid, shared_user_uuid),
        )

        assert shared_user_uuid not in response["shared_user_uuids"]

    @pytest.mark.asyncio
    async def test_remove_user_not_in_share_returns_404(self, chat, user_id, async_client, async_http_requester):
        """Removing a user who is not part of the share returns 404"""
        await async_http_requester(
            "remove unknown shared user",
            async_client.delete,
            share_user_endpoint(user_id, chat.uuid, str(uuid.uuid4())),
            response_code=404,
        )

    @pytest.mark.asyncio
    async def test_manage_shared_users_requires_ownership(self, chat, async_client, async_http_requester):
        """A user who does not own the chat cannot manage its shared users"""
        non_owner = str(uuid.uuid4())

        await async_http_requester(
            "add shared user as non-owner",
            async_client.post,
            share_users_endpoint(non_owner, chat.uuid),
            json={"shared_user_uuid": str(uuid.uuid4())},
            response_code=403,
        )


class TestChatShareUserNotification:
    async def _add_user(self, async_http_requester, async_client, user_id, chat, shared_user_uuid):
        return await async_http_requester(
            "add shared user",
            async_client.post,
            share_users_endpoint(user_id, chat.uuid),
            json={"shared_user_uuid": shared_user_uuid},
        )

    @pytest.mark.asyncio
    async def test_added_user_starts_unnotified(self, chat, user_id, async_client, async_http_requester):
        """A freshly added user appears in shared_users with a null notified_at"""
        shared_user_uuid = str(uuid.uuid4())

        response = await self._add_user(async_http_requester, async_client, user_id, chat, shared_user_uuid)

        assert response["shared_users"] == [{"uuid": shared_user_uuid, "notified_at": None}]
        assert response["shared_user_uuids"] == [shared_user_uuid]

    @pytest.mark.asyncio
    async def test_notify_shared_user_sets_notified_at(self, chat, user_id, async_client, async_http_requester):
        """PATCH {"notified": true} stamps notified_at for that user only"""
        notified_uuid = str(uuid.uuid4())
        other_uuid = str(uuid.uuid4())
        await self._add_user(async_http_requester, async_client, user_id, chat, notified_uuid)
        await self._add_user(async_http_requester, async_client, user_id, chat, other_uuid)

        response = await async_http_requester(
            "mark shared user notified",
            async_client.patch,
            share_user_endpoint(user_id, chat.uuid, notified_uuid),
            json={"notified": True},
        )

        by_uuid = {u["uuid"]: u["notified_at"] for u in response["shared_users"]}
        assert by_uuid[notified_uuid] is not None
        assert by_uuid[other_uuid] is None

    @pytest.mark.asyncio
    async def test_notified_state_persists_in_list(self, chat, user_id, async_client, async_http_requester):
        """notified_at set via PATCH is returned by subsequent GETs"""
        shared_user_uuid = str(uuid.uuid4())
        await self._add_user(async_http_requester, async_client, user_id, chat, shared_user_uuid)

        patch_response = await async_http_requester(
            "mark shared user notified",
            async_client.patch,
            share_user_endpoint(user_id, chat.uuid, shared_user_uuid),
            json={"notified": True},
        )
        notified_at = patch_response["shared_users"][0]["notified_at"]

        get_response = await async_http_requester(
            "list shared users", async_client.get, share_users_endpoint(user_id, chat.uuid)
        )

        assert get_response["shared_users"][0]["notified_at"] == notified_at

    @pytest.mark.asyncio
    async def test_notify_can_be_cleared(self, chat, user_id, async_client, async_http_requester):
        """PATCH {"notified": false} clears notified_at"""
        shared_user_uuid = str(uuid.uuid4())
        await self._add_user(async_http_requester, async_client, user_id, chat, shared_user_uuid)
        endpoint = share_user_endpoint(user_id, chat.uuid, shared_user_uuid)

        await async_http_requester("mark notified", async_client.patch, endpoint, json={"notified": True})
        response = await async_http_requester("clear notified", async_client.patch, endpoint, json={"notified": False})

        assert response["shared_users"][0]["notified_at"] is None

    @pytest.mark.asyncio
    async def test_notify_user_not_in_share_returns_404(self, chat, user_id, async_client, async_http_requester):
        """Notifying a user who is not on the list returns 404"""
        await async_http_requester(
            "notify unknown shared user",
            async_client.patch,
            share_user_endpoint(user_id, chat.uuid, str(uuid.uuid4())),
            json={"notified": True},
            response_code=404,
        )

    @pytest.mark.asyncio
    async def test_readding_user_resets_notification(self, chat, user_id, async_client, async_http_requester):
        """Removing a user and re-adding them starts with a fresh, un-notified state"""
        shared_user_uuid = str(uuid.uuid4())
        await self._add_user(async_http_requester, async_client, user_id, chat, shared_user_uuid)
        await async_http_requester(
            "mark notified",
            async_client.patch,
            share_user_endpoint(user_id, chat.uuid, shared_user_uuid),
            json={"notified": True},
        )

        await async_http_requester(
            "remove shared user", async_client.delete, share_user_endpoint(user_id, chat.uuid, shared_user_uuid)
        )
        response = await self._add_user(async_http_requester, async_client, user_id, chat, shared_user_uuid)

        assert response["shared_users"] == [{"uuid": shared_user_uuid, "notified_at": None}]


class TestPrivateShareWithoutNotifications:
    """The notification feature is built ahead of the PHP client. These tests prove that a
    client which only sends the original private-share data — and never calls the notify
    PATCH — gets the full share lifecycle, with notified_at left untouched throughout."""

    async def _add_user(self, async_http_requester, async_client, user_id, chat, shared_user_uuid):
        return await async_http_requester(
            "add shared user",
            async_client.post,
            share_users_endpoint(user_id, chat.uuid),
            json={"shared_user_uuid": shared_user_uuid},
        )

    @pytest.mark.asyncio
    async def test_full_share_lifecycle_without_touching_notifications(
        self, chat, user_id, async_client, async_http_requester
    ):
        """A client can add, list and remove shared users without ever sending notification
        data. The legacy shared_user_uuids field keeps working and notified_at stays null."""
        first_uuid = str(uuid.uuid4())
        second_uuid = str(uuid.uuid4())

        add_response = await self._add_user(async_http_requester, async_client, user_id, chat, first_uuid)
        # The legacy uuid list is still populated for clients that never read shared_users.
        assert add_response["shared_user_uuids"] == [first_uuid]

        await self._add_user(async_http_requester, async_client, user_id, chat, second_uuid)

        list_response = await async_http_requester(
            "list shared users", async_client.get, share_users_endpoint(user_id, chat.uuid)
        )
        assert list_response["shared_user_uuids"] == [first_uuid, second_uuid]
        # notified_at is never required on the way in; it defaults to null for everyone.
        assert list_response["shared_users"] == [
            {"uuid": first_uuid, "notified_at": None},
            {"uuid": second_uuid, "notified_at": None},
        ]

        remove_response = await async_http_requester(
            "remove shared user",
            async_client.delete,
            share_user_endpoint(user_id, chat.uuid, first_uuid),
        )
        assert remove_response["shared_user_uuids"] == [second_uuid]
        assert remove_response["shared_users"] == [{"uuid": second_uuid, "notified_at": None}]


class TestPrivateSharedChatAccess:
    async def _other_user_headers(self, another_user_auth_session, auth_token, other_user_uuid=None):
        other_user_uuid = other_user_uuid or str(uuid.uuid4())
        other_session = await another_user_auth_session(other_user_uuid)
        return other_user_uuid, {
            USER_KEY_UUID_ALIAS: other_user_uuid,
            SESSION_AUTH_ALIAS: other_session,
            AUTH_TOKEN_ALIAS: auth_token,
        }

    async def _enable_private_share(self, async_http_requester, async_client, user_id, chat):
        response = await async_http_requester(
            "enable private share",
            async_client.patch,
            share_endpoint(user_id, chat.uuid),
            json={"share": True, "share_private": True},
        )
        return response["share_code"]

    @pytest.mark.asyncio
    async def test_shared_user_can_access_private_chat(
        self, chat, user_id, async_client, async_http_requester, another_user_auth_session, auth_token
    ):
        """A user added to the private share can view the chat via the share link"""
        share_code = await self._enable_private_share(async_http_requester, async_client, user_id, chat)

        other_user_uuid, other_user_headers = await self._other_user_headers(another_user_auth_session, auth_token)
        await async_http_requester(
            "add shared user",
            async_client.post,
            share_users_endpoint(user_id, chat.uuid),
            json={"shared_user_uuid": other_user_uuid},
        )

        response = await async_http_requester(
            "access private shared chat as shared user",
            async_client.get,
            f"/v1/chats/shared/{share_code}",
            headers=other_user_headers,
        )

        assert response["uuid"] == str(chat.uuid)
        assert "messages" in response

    @pytest.mark.asyncio
    async def test_non_shared_user_cannot_access_private_chat(
        self, chat, user_id, async_client, async_http_requester, another_user_auth_session, auth_token
    ):
        """A user who was not added to the private share gets 403"""
        share_code = await self._enable_private_share(async_http_requester, async_client, user_id, chat)

        _, other_user_headers = await self._other_user_headers(another_user_auth_session, auth_token)

        response = await async_http_requester(
            "access private shared chat as non-shared user",
            async_client.get,
            f"/v1/chats/shared/{share_code}",
            headers=other_user_headers,
            response_code=403,
        )

        # The machine-readable error code lets the frontend distinguish "not on the
        # allow-list" from "sharing turned off" (both are 403s).
        assert response["detail"]["error_code"] == "private_share_access_denied"

    @pytest.mark.asyncio
    async def test_unsharing_returns_not_shared_error_to_previously_approved_user(
        self, chat, user_id, async_client, async_http_requester, another_user_auth_session, auth_token
    ):
        """When sharing is turned off entirely, a previously approved user gets the
        'not shared' error, not the private-share access-denied error."""
        share_code = await self._enable_private_share(async_http_requester, async_client, user_id, chat)

        other_user_uuid, other_user_headers = await self._other_user_headers(another_user_auth_session, auth_token)
        await async_http_requester(
            "add shared user",
            async_client.post,
            share_users_endpoint(user_id, chat.uuid),
            json={"shared_user_uuid": other_user_uuid},
        )

        await async_http_requester(
            "disable sharing entirely",
            async_client.patch,
            share_endpoint(user_id, chat.uuid),
            json={"share": False},
        )

        response = await async_http_requester(
            "access unshared chat as previously approved user",
            async_client.get,
            f"/v1/chats/shared/{share_code}",
            headers=other_user_headers,
            response_code=403,
        )

        assert response["detail"] == "This chat is not shared"

    @pytest.mark.asyncio
    async def test_owner_can_access_own_private_chat(self, chat, user_id, async_client, async_http_requester):
        """The chat owner can always view their own privately shared chat"""
        share_code = await self._enable_private_share(async_http_requester, async_client, user_id, chat)

        response = await async_http_requester(
            "access private shared chat as owner",
            async_client.get,
            f"/v1/chats/shared/{share_code}",
        )

        assert response["uuid"] == str(chat.uuid)

    @pytest.mark.asyncio
    async def test_access_revoked_after_removal(
        self, chat, user_id, async_client, async_http_requester, another_user_auth_session, auth_token
    ):
        """A user removed from the private share loses access"""
        share_code = await self._enable_private_share(async_http_requester, async_client, user_id, chat)

        other_user_uuid, other_user_headers = await self._other_user_headers(another_user_auth_session, auth_token)
        await async_http_requester(
            "add shared user",
            async_client.post,
            share_users_endpoint(user_id, chat.uuid),
            json={"shared_user_uuid": other_user_uuid},
        )
        await async_http_requester(
            "access before removal",
            async_client.get,
            f"/v1/chats/shared/{share_code}",
            headers=other_user_headers,
        )

        await async_http_requester(
            "remove shared user",
            async_client.delete,
            share_user_endpoint(user_id, chat.uuid, other_user_uuid),
        )

        await async_http_requester(
            "access after removal",
            async_client.get,
            f"/v1/chats/shared/{share_code}",
            headers=other_user_headers,
            response_code=403,
        )

    @pytest.mark.asyncio
    async def test_public_share_remains_open_to_any_authenticated_user(
        self, chat, user_id, async_client, async_http_requester, another_user_auth_session, auth_token
    ):
        """Regression: public shares are still accessible to any authenticated user"""
        response = await async_http_requester(
            "enable public share",
            async_client.patch,
            share_endpoint(user_id, chat.uuid),
            json={"share": True},
        )
        share_code = response["share_code"]

        _, other_user_headers = await self._other_user_headers(another_user_auth_session, auth_token)

        response = await async_http_requester(
            "access public shared chat",
            async_client.get,
            f"/v1/chats/shared/{share_code}",
            headers=other_user_headers,
        )

        assert response["uuid"] == str(chat.uuid)

    @pytest.mark.asyncio
    async def test_private_chat_switched_to_public_is_open_again(
        self, chat, user_id, async_client, async_http_requester, another_user_auth_session, auth_token
    ):
        """Switching a private share back to public re-opens it to any authenticated user"""
        share_code = await self._enable_private_share(async_http_requester, async_client, user_id, chat)

        _, other_user_headers = await self._other_user_headers(another_user_auth_session, auth_token)
        await async_http_requester(
            "access while private",
            async_client.get,
            f"/v1/chats/shared/{share_code}",
            headers=other_user_headers,
            response_code=403,
        )

        await async_http_requester(
            "switch to public",
            async_client.patch,
            share_endpoint(user_id, chat.uuid),
            json={"share": True, "share_private": False},
        )

        response = await async_http_requester(
            "access once public",
            async_client.get,
            f"/v1/chats/shared/{share_code}",
            headers=other_user_headers,
        )

        assert response["uuid"] == str(chat.uuid)
