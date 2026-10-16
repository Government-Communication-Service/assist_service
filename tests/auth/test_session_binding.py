import logging
from datetime import datetime
from uuid import uuid4

import pytest
from sqlalchemy import update

from app.api.endpoints import ENDPOINTS
from app.auth.constants import AUTH_TOKEN_ALIAS, SESSION_AUTH_ALIAS, USER_KEY_UUID_ALIAS
from app.database.db_session import get_db_session
from app.database.models import AuthSession
from app.database.table import async_db_session
from app.main import app

api = ENDPOINTS()
logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.auth,
]


def _user_chats_url(user_uuid: str) -> str:
    return api.build_url(ENDPOINTS.USER_GET_CHATS, user_uuid=user_uuid)


@pytest.mark.e2e
class TestSessionAuthIsBoundToUser:
    async def test_own_session_is_accepted(self, async_client, user_id, default_headers):
        response = await async_client.get(_user_chats_url(user_id), headers=default_headers)
        assert response.status_code == 200, response.text

    async def test_another_users_session_is_rejected(self, async_client, auth_token, another_user_auth_session):
        user_a = str(uuid4())
        user_b = str(uuid4())
        session_a = await another_user_auth_session(user_a)

        headers = {AUTH_TOKEN_ALIAS: auth_token, USER_KEY_UUID_ALIAS: user_b, SESSION_AUTH_ALIAS: session_a}
        response = await async_client.get(_user_chats_url(user_b), headers=headers)

        assert response.status_code == 404, response.text

    async def test_deleted_session_is_rejected(self, async_client, user_id, auth_session, default_headers):
        async with async_db_session() as session:
            await session.execute(
                update(AuthSession).where(AuthSession.id == auth_session.id).values(deleted_at=datetime.now())
            )

        response = await async_client.get(_user_chats_url(user_id), headers=default_headers)
        assert response.status_code == 404, response.text


@pytest.mark.unit
def test_every_db_session_dependency_is_function_scoped():
    """
    Mixing scopes in one request gives two separate sessions, and a request-scoped session stays open until a
    streamed response has finished. So every use of get_db_session must go through the DbSession alias.
    """

    def api_routes(routes):
        for route in routes:
            if hasattr(route, "dependant"):
                yield route
            # Included routers and mounts hold their own routes
            nested = getattr(getattr(route, "original_router", route), "routes", None)
            if nested is not None and nested is not routes:
                yield from api_routes(nested)

    def walk(dependant):
        for sub in dependant.dependencies:
            yield sub
            yield from walk(sub)

    db_session_deps = [
        (route.path, sub)
        for route in api_routes(app.routes)
        for sub in walk(route.dependant)
        if sub.call is get_db_session
    ]

    assert db_session_deps, "Expected at least one route to depend on get_db_session"
    not_function_scoped = [path for path, sub in db_session_deps if sub.scope != "function"]
    assert not not_function_scoped, f"Routes with a request-scoped get_db_session: {sorted(set(not_function_scoped))}"
