import asyncio
import json
import logging
import uuid
from datetime import datetime

import pytest
from sqlalchemy import select, update

from app.api.endpoints import ENDPOINTS
from app.auth.constants import AUTH_TOKEN_ALIAS, SESSION_AUTH_ALIAS, USER_KEY_UUID_ALIAS
from app.database.models import Chat, Project
from app.database.table import async_db_session
from app.projects import queries

api = ENDPOINTS()

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.projects,
    pytest.mark.usefixtures("mock_llm_response", "mock_chat_model"),
]


@pytest.fixture
def project_payload():
    return {"name": "Campaign planning", "description": "Campaign plans and supporting conversations."}


@pytest.fixture
async def create_project(async_client, user_id, async_http_requester):
    async def _create(name, description=None, response_code=201, owner_uuid=None, headers=None):
        body = {"name": name}
        if description is not None:
            body["description"] = description
        return await async_http_requester(
            "create project",
            async_client.post,
            api.user_projects(owner_uuid or user_id),
            json=body,
            response_code=response_code,
            headers=headers,
        )

    return _create


@pytest.fixture
async def project_item(create_project, project_payload):
    return await create_project(**project_payload)


@pytest.fixture
async def create_chat(async_client, user_id, async_http_requester, put_chat_in_project):
    async def _create(project_uuid=None, owner_uuid=None, headers=None):
        chat = await async_http_requester(
            "create chat",
            async_client.post,
            api.chats(owner_uuid or user_id),
            json={"query": "Say Hi", "use_rag": False},
            headers=headers,
        )
        if project_uuid is not None:
            chat = await put_chat_in_project(project_uuid, chat["uuid"])
        return chat

    return _create


@pytest.fixture
async def other_user(another_user_auth_session, auth_token):
    """A second real user, with their own valid headers."""
    other_user_uuid = str(uuid.uuid4())
    other_session = await another_user_auth_session(other_user_uuid)
    return other_user_uuid, {
        USER_KEY_UUID_ALIAS: other_user_uuid,
        SESSION_AUTH_ALIAS: other_session,
        AUTH_TOKEN_ALIAS: auth_token,
    }


@pytest.fixture
async def get_history(async_client, user_id, async_http_requester):
    async def _get():
        response = await async_http_requester("chat history", async_client.get, api.get_chats_by_user(user_id))
        return {chat["uuid"]: chat for chat in response["chats"]}

    return _get


@pytest.fixture
async def put_chat_in_project(async_client, user_id, async_http_requester):
    async def _put(project_uuid, chat_uuid, response_code=200):
        return await async_http_requester(
            "add chat to project",
            async_client.put,
            api.user_project_chat(user_id, project_uuid, chat_uuid),
            response_code=response_code,
        )

    return _put


@pytest.fixture
async def get_project_chats(async_client, user_id, async_http_requester):
    async def _get(project_uuid):
        return await async_http_requester(
            "list project chats", async_client.get, api.user_project_chats(user_id, project_uuid)
        )

    return _get


def assert_project_response(response, name, description):
    assert uuid.UUID(response["uuid"])
    assert response["name"] == name
    assert response["description"] == description
    assert response["created_at"]
    assert response["updated_at"]


class TestCreateProject:
    async def test_create_project_success(self, project_item, project_payload):
        assert_project_response(project_item, project_payload["name"], project_payload["description"])

    async def test_create_project_trims_name_and_blank_description_is_null(self, create_project):
        response = await create_project("  Policy research  ", description="   ")

        assert_project_response(response, "Policy research", None)

    async def test_create_project_without_description(self, create_project):
        response = await create_project("Team updates")

        assert_project_response(response, "Team updates", None)

    @pytest.mark.parametrize("name", ["", "   ", "x" * 256])
    async def test_create_project_invalid_name(self, create_project, name):
        await create_project(name, response_code=422)

    async def test_create_project_description_too_long(self, create_project):
        await create_project("Long description", description="x" * 1001, response_code=422)

    async def test_create_project_duplicate_name_ignoring_case(self, project_item, create_project):
        response = await create_project(project_item["name"].upper(), response_code=409)

        assert response["detail"] == "A project with this name already exists"

    async def test_create_project_name_reusable_after_delete(
        self, project_item, create_project, async_client, user_id, async_http_requester
    ):
        await async_http_requester(
            "delete project",
            async_client.delete,
            api.user_project(user_id, project_item["uuid"]),
            response_code=204,
            response_type=None,
        )

        await create_project(project_item["name"])

    async def test_same_name_allowed_for_different_users(self, project_item, create_project, other_user):
        other_user_uuid, other_headers = other_user

        await create_project(project_item["name"], owner_uuid=other_user_uuid, headers=other_headers)

    async def test_create_project_user_mismatch(self, create_project):
        await create_project("Mismatch", owner_uuid=str(uuid.uuid4()), response_code=403)


class TestListProjects:
    async def test_list_projects_alphabetical(self, create_project, async_client, user_id, async_http_requester):
        await create_project("team updates")
        await create_project("Campaign planning")
        await create_project("policy research")

        response = await async_http_requester("list projects", async_client.get, api.user_projects(user_id))

        assert [p["name"] for p in response["projects"]] == ["Campaign planning", "policy research", "team updates"]

    async def test_list_projects_excludes_deleted_and_other_users_projects(
        self, create_project, other_user, async_client, user_id, async_http_requester
    ):
        other_user_uuid, other_headers = other_user
        await create_project("Kept")
        deleted = await create_project("Deleted")
        await create_project("Not mine", owner_uuid=other_user_uuid, headers=other_headers)
        await async_http_requester(
            "delete project",
            async_client.delete,
            api.user_project(user_id, deleted["uuid"]),
            response_code=204,
            response_type=None,
        )

        response = await async_http_requester("list projects", async_client.get, api.user_projects(user_id))

        assert [p["name"] for p in response["projects"]] == ["Kept"]

    async def test_list_projects_empty(self, async_client, user_id, async_http_requester):
        response = await async_http_requester("list projects", async_client.get, api.user_projects(user_id))

        assert response == {"projects": []}


class TestGetProject:
    async def test_get_project_success(
        self, project_item, project_payload, async_client, user_id, async_http_requester
    ):
        response = await async_http_requester(
            "get project", async_client.get, api.user_project(user_id, project_item["uuid"])
        )

        assert_project_response(response, project_payload["name"], project_payload["description"])
        assert response["uuid"] == project_item["uuid"]

    async def test_get_other_users_project_not_found(
        self, project_item, other_user, async_client, async_http_requester
    ):
        other_user_uuid, other_headers = other_user

        response = await async_http_requester(
            "get other user's project",
            async_client.get,
            api.user_project(other_user_uuid, project_item["uuid"]),
            headers=other_headers,
            response_code=404,
        )

        assert response["detail"] == "Project not found"

    async def test_get_unknown_project_not_found(self, async_client, user_id, async_http_requester):
        await async_http_requester(
            "get unknown project", async_client.get, api.user_project(user_id, uuid.uuid4()), response_code=404
        )

    async def test_get_project_malformed_uuid(self, async_client, user_id, async_http_requester):
        await async_http_requester(
            "get malformed project", async_client.get, api.user_project(user_id, "not-a-uuid"), response_code=422
        )

    async def test_get_deleted_project_not_found(self, project_item, async_client, user_id, async_http_requester):
        url = api.user_project(user_id, project_item["uuid"])
        await async_http_requester("delete project", async_client.delete, url, response_code=204, response_type=None)

        await async_http_requester("get deleted project", async_client.get, url, response_code=404)


class TestUpdateProject:
    async def test_update_name_only(self, project_item, project_payload, async_client, user_id, async_http_requester):
        response = await async_http_requester(
            "rename project",
            async_client.patch,
            api.user_project(user_id, project_item["uuid"]),
            json={"name": "  Renamed  "},
        )

        assert_project_response(response, "Renamed", project_payload["description"])

    async def test_update_description_only(
        self, project_item, project_payload, async_client, user_id, async_http_requester
    ):
        response = await async_http_requester(
            "update description",
            async_client.patch,
            api.user_project(user_id, project_item["uuid"]),
            json={"description": "New description"},
        )

        assert_project_response(response, project_payload["name"], "New description")

    async def test_clear_description(self, project_item, project_payload, async_client, user_id, async_http_requester):
        response = await async_http_requester(
            "clear description",
            async_client.patch,
            api.user_project(user_id, project_item["uuid"]),
            json={"description": None},
        )

        assert_project_response(response, project_payload["name"], None)

    @pytest.mark.parametrize("body", [{}, {"name": None}, {"name": "  "}])
    async def test_update_invalid_body(self, project_item, async_client, user_id, async_http_requester, body):
        await async_http_requester(
            "invalid update",
            async_client.patch,
            api.user_project(user_id, project_item["uuid"]),
            json=body,
            response_code=422,
        )

    async def test_rename_onto_existing_name_conflicts(
        self, project_item, create_project, async_client, user_id, async_http_requester
    ):
        other = await create_project("Policy research")

        await async_http_requester(
            "rename onto existing",
            async_client.patch,
            api.user_project(user_id, other["uuid"]),
            json={"name": project_item["name"].lower()},
            response_code=409,
        )

    async def test_case_only_rename_of_own_name(self, project_item, async_client, user_id, async_http_requester):
        response = await async_http_requester(
            "case-only rename",
            async_client.patch,
            api.user_project(user_id, project_item["uuid"]),
            json={"name": project_item["name"].upper()},
        )

        assert response["name"] == project_item["name"].upper()

    async def test_update_other_users_project_not_found(
        self, project_item, other_user, async_client, async_http_requester
    ):
        other_user_uuid, other_headers = other_user

        await async_http_requester(
            "update other user's project",
            async_client.patch,
            api.user_project(other_user_uuid, project_item["uuid"]),
            json={"name": "Hijacked"},
            headers=other_headers,
            response_code=404,
        )


class TestDeleteProject:
    async def test_delete_project_keeps_chats_in_history(
        self, project_item, create_chat, get_history, async_client, user_id, async_http_requester
    ):
        chat = await create_chat(project_uuid=project_item["uuid"])
        url = api.user_project(user_id, project_item["uuid"])

        response = await async_http_requester(
            "delete project", async_client.delete, url, response_code=204, response_type=None
        )

        assert response == b""
        history = await get_history()
        assert chat["uuid"] in history
        assert history[chat["uuid"]]["project_uuid"] is None
        await async_http_requester("get deleted project", async_client.get, url, response_code=404)

    async def test_chat_left_pointing_at_deleted_project_reads_as_unassigned(
        self,
        project_item,
        create_project,
        create_chat,
        get_history,
        put_chat_in_project,
        async_client,
        user_id,
        async_http_requester,
    ):
        # A deleted project whose chats were never detached.
        chat = await create_chat(project_uuid=project_item["uuid"])
        async with async_db_session() as session:
            await session.execute(
                update(Project).where(Project.uuid == project_item["uuid"]).values(deleted_at=datetime.now())
            )

        assert (await get_history())[chat["uuid"]]["project_uuid"] is None
        chat_item = await async_http_requester("get chat", async_client.get, api.get_chat_item(user_id, chat["uuid"]))
        assert chat_item["project_uuid"] is None
        # And it can still be put into another project.
        other_project = await create_project("Policy research")
        moved = await put_chat_in_project(other_project["uuid"], chat["uuid"])
        assert moved["project_uuid"] == other_project["uuid"]

    async def test_add_chat_waits_for_in_flight_delete_then_not_found(
        self, project_item, create_chat, async_client, user_id, async_http_requester
    ):
        chat = await create_chat()
        url = api.user_project_chat(user_id, project_item["uuid"], chat["uuid"])

        async with async_db_session() as session:
            project = await session.scalar(
                select(Project).where(Project.uuid == project_item["uuid"]).with_for_update()
            )
            await queries.delete_project(db_session=session, project=project)
            add = asyncio.create_task(
                async_http_requester("add during delete", async_client.put, url, response_code=404)
            )
            # Let the add reach the project lock before the delete commits.
            await asyncio.sleep(0.3)
            assert not add.done()

        await add
        async with async_db_session() as session:
            assert await session.scalar(select(Chat.project_id).where(Chat.uuid == chat["uuid"])) is None

    async def test_patch_waits_for_in_flight_delete_then_not_found(
        self, project_item, async_client, user_id, async_http_requester
    ):
        url = api.user_project(user_id, project_item["uuid"])

        async with async_db_session() as session:
            project = await session.scalar(
                select(Project).where(Project.uuid == project_item["uuid"]).with_for_update()
            )
            await queries.delete_project(db_session=session, project=project)
            patch = asyncio.create_task(
                async_http_requester(
                    "patch during delete", async_client.patch, url, json={"name": "Renamed"}, response_code=404
                )
            )
            await asyncio.sleep(0.3)
            assert not patch.done()

        await patch

    async def test_delete_other_users_project_not_found(
        self, project_item, other_user, async_client, user_id, async_http_requester
    ):
        other_user_uuid, other_headers = other_user

        await async_http_requester(
            "delete other user's project",
            async_client.delete,
            api.user_project(other_user_uuid, project_item["uuid"]),
            headers=other_headers,
            response_code=404,
        )
        await async_http_requester(
            "project still exists", async_client.get, api.user_project(user_id, project_item["uuid"])
        )


class TestChatResponsesIncludeProject:
    async def test_chat_in_project(
        self, project_item, create_chat, get_history, async_client, user_id, async_http_requester
    ):
        chat = await create_chat(project_uuid=project_item["uuid"])

        assert (await get_history())[chat["uuid"]]["project_uuid"] == project_item["uuid"]
        chat_item = await async_http_requester("get chat", async_client.get, api.get_chat_item(user_id, chat["uuid"]))
        assert chat_item["project_uuid"] == project_item["uuid"]

    async def test_create_chat_without_project(self, create_chat):
        chat = await create_chat()

        assert chat["project_uuid"] is None

    async def test_create_chat_stream_without_project_sends_null_project_uuid(
        self, async_client, user_id, async_http_requester
    ):
        response = await async_http_requester(
            "create chat stream without project",
            async_client.post,
            api.create_chat_stream(user_id),
            json={"query": "Say Hi", "use_rag": False},
            response_type=None,
        )

        chunk, _ = json.JSONDecoder().raw_decode(response.decode())
        assert "project_uuid" in chunk
        assert chunk["project_uuid"] is None

    @pytest.mark.parametrize("in_project", [True, False])
    async def test_add_message_stream_sends_project_uuid(
        self, in_project, project_item, create_chat, async_client, user_id, async_http_requester
    ):
        chat = await create_chat(project_uuid=project_item["uuid"] if in_project else None)

        response = await async_http_requester(
            "add message stream",
            async_client.put,
            api.get_chat_stream(user_id, chat["uuid"]),
            json={"query": "And again"},
            response_type=None,
        )

        chunk, _ = json.JSONDecoder().raw_decode(response.decode())
        assert "project_uuid" in chunk
        assert chunk["project_uuid"] == (project_item["uuid"] if in_project else None)


class TestAddChatToProject:
    async def test_add_existing_chat(self, project_item, create_chat, put_chat_in_project, get_history):
        chat = await create_chat()
        history_before = await get_history()

        response = await put_chat_in_project(project_item["uuid"], chat["uuid"])

        assert response["uuid"] == chat["uuid"]
        assert response["project_uuid"] == project_item["uuid"]
        # Moving a chat into a project must not reorder the chat history.
        history_after = await get_history()
        assert list(history_after) == list(history_before)
        assert history_after[chat["uuid"]]["updated_at"] == history_before[chat["uuid"]]["updated_at"]

    async def test_add_chat_is_idempotent(self, project_item, create_chat, put_chat_in_project, get_project_chats):
        chat = await create_chat()

        await put_chat_in_project(project_item["uuid"], chat["uuid"])
        await put_chat_in_project(project_item["uuid"], chat["uuid"])

        project_chats = await get_project_chats(project_item["uuid"])
        assert [c["uuid"] for c in project_chats["chats"]] == [chat["uuid"]]

    async def test_move_chat_between_projects(
        self, project_item, create_project, create_chat, put_chat_in_project, get_project_chats
    ):
        other_project = await create_project("Policy research")
        chat = await create_chat(project_uuid=project_item["uuid"])

        response = await put_chat_in_project(other_project["uuid"], chat["uuid"])

        assert response["project_uuid"] == other_project["uuid"]
        assert (await get_project_chats(project_item["uuid"]))["chats"] == []
        assert [c["uuid"] for c in (await get_project_chats(other_project["uuid"]))["chats"]] == [chat["uuid"]]

    async def test_add_other_users_chat_not_found(self, project_item, create_chat, put_chat_in_project, other_user):
        other_user_uuid, other_headers = other_user
        other_chat = await create_chat(owner_uuid=other_user_uuid, headers=other_headers)

        response = await put_chat_in_project(project_item["uuid"], other_chat["uuid"], response_code=404)

        assert response["detail"] == "Chat not found"

    async def test_add_archived_chat_not_found(
        self, project_item, create_chat, put_chat_in_project, async_client, user_id, async_http_requester
    ):
        chat = await create_chat()
        await async_http_requester("archive chat", async_client.patch, api.archive_chat(user_id, chat["uuid"]))

        await put_chat_in_project(project_item["uuid"], chat["uuid"], response_code=404)

    async def test_add_chat_to_unknown_project(self, create_chat, put_chat_in_project):
        chat = await create_chat()

        response = await put_chat_in_project(uuid.uuid4(), chat["uuid"], response_code=404)

        assert response["detail"] == "Project not found"


class TestRemoveChatFromProject:
    async def test_remove_chat(
        self, project_item, create_chat, get_history, async_client, user_id, async_http_requester
    ):
        chat = await create_chat(project_uuid=project_item["uuid"])
        url = api.user_project_chat(user_id, project_item["uuid"], chat["uuid"])

        await async_http_requester("remove chat", async_client.delete, url, response_code=204, response_type=None)

        history = await get_history()
        assert history[chat["uuid"]]["project_uuid"] is None
        await async_http_requester("remove chat again", async_client.delete, url, response_code=404)

    async def test_remove_chat_not_in_project(
        self, project_item, create_chat, async_client, user_id, async_http_requester
    ):
        chat = await create_chat()

        await async_http_requester(
            "remove chat not in project",
            async_client.delete,
            api.user_project_chat(user_id, project_item["uuid"], chat["uuid"]),
            response_code=404,
        )


class TestListProjectChats:
    async def test_most_recently_updated_first(
        self, project_item, create_chat, get_project_chats, async_client, user_id, async_http_requester
    ):
        first, second = [await create_chat(project_uuid=project_item["uuid"]) for _ in range(2)]
        # Retitling bumps updated_at.
        await async_http_requester(
            "retitle chat", async_client.patch, api.create_chat_title(user_id, first["uuid"]), json={"title": "Newer"}
        )

        response = await get_project_chats(project_item["uuid"])

        assert [c["uuid"] for c in response["chats"]] == [first["uuid"], second["uuid"]]

    async def test_excludes_archived_chats(
        self, project_item, create_chat, get_project_chats, async_client, user_id, async_http_requester
    ):
        kept = await create_chat(project_uuid=project_item["uuid"])
        archived = await create_chat(project_uuid=project_item["uuid"])
        await async_http_requester("archive chat", async_client.patch, api.archive_chat(user_id, archived["uuid"]))

        response = await get_project_chats(project_item["uuid"])

        assert response["project_uuid"] == project_item["uuid"]
        assert [c["uuid"] for c in response["chats"]] == [kept["uuid"]]

    async def test_other_users_project_chats_not_found(
        self, project_item, other_user, async_client, async_http_requester
    ):
        other_user_uuid, other_headers = other_user

        await async_http_requester(
            "list other user's project chats",
            async_client.get,
            api.user_project_chats(other_user_uuid, project_item["uuid"]),
            headers=other_headers,
            response_code=404,
        )
