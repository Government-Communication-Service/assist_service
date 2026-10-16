import logging
import uuid

import pytest

from app.api.endpoints import ENDPOINTS
from app.auth.constants import SESSION_AUTH_ALIAS

api = ENDPOINTS()
logger = logging.getLogger(__name__)


@pytest.fixture
def session_check_on(monkeypatch):
    monkeypatch.setattr("app.user.utils.USER_UPDATE_REQUIRES_SESSION_AUTH", True)


class TestUserManagement:
    async def test_create_user_success(self, async_client, auth_token_only_headers):
        new_user_uuid = str(uuid.uuid4())
        payload = {
            "uuid": new_user_uuid,
            "job_title": "Software Engineer",
            "region": "North",
            "sector": "Technology",
            "organisation": "Tech Corp",
            "grade": "A",
            "communicator_role": True,
        }

        # Send the POST request to the /users endpoint using the async client.
        url = api.create_user()
        response = await async_client.post(url, json=payload, headers=auth_token_only_headers)

        # Assert that the status code is 201 (Created).
        assert response.status_code == 201, f"Expected status code 200, got {response.status_code}"

        # Parse the JSON response.
        data = response.json()

        # Check that the response indicates success.
        assert data.get("success") is True, f"Expected success True, got {data.get('success')}"

        # Check that the message indicates the user was created.
        expected_message = f"Successfully created user with UUID: {new_user_uuid}"
        assert expected_message in data.get("message", ""), f"Expected message to include '{expected_message}'"

    async def test_create_duplicate_user(self, async_client, auth_token_only_headers):
        user_uuid = str(uuid.uuid4())
        payload = {
            "uuid": user_uuid,
            "job_title": "Software Engineer",
            "region": "North",
            "sector": "Technology",
            "organisation": "Tech Corp",
            "grade": "A",
            "communicator_role": True,
        }

        url = api.create_user()

        # Create the user for the first time.
        response1 = await async_client.post(url, json=payload, headers=auth_token_only_headers)
        assert response1.status_code == 201, f"Expected status code 201, got {response1.status_code}"
        data1 = response1.json()
        assert data1.get("success") is True, "The first user creation should succeed."

        # Attempt to create the same user again. 409 Conflict should be returned.
        response2 = await async_client.post(url, json=payload, headers=auth_token_only_headers)
        assert response2.status_code == 409, f"Expected status code 409, got {response2.status_code}"
        data2 = response2.json()
        expected_message = f"User already exists with UUID: {user_uuid}"
        assert expected_message in data2["detail"]

    async def test_update_user_success(self, async_client, auth_token_only_headers):
        # create a test user
        user_uuid = str(uuid.uuid4())
        create_payload = {
            "uuid": user_uuid,
            "job_title": "Software Engineer",
            "region": "North",
            "sector": "Technology",
            "organisation": "Tech Corp",
            "grade": "A",
            "communicator_role": True,
        }

        # Create the user first
        create_url = api.create_user()
        create_response = await async_client.post(create_url, json=create_payload, headers=auth_token_only_headers)
        assert create_response.status_code == 201

        # Prepare update payload with modified data
        update_payload = {
            "job_title": "Senior Software Engineer",
            "region": "South",
            "sector": "Technology",
            "organisation": "New Tech Corp",
            "grade": "B",
            "communicator_role": False,
        }

        # The update needs the user's own User-Key-UUID and Session-Auth
        user_headers = {**auth_token_only_headers, "User-Key-UUID": user_uuid}
        session_response = await async_client.post(api.get_sessions(), headers=user_headers)
        assert session_response.status_code == 200
        user_headers[SESSION_AUTH_ALIAS] = session_response.json()[SESSION_AUTH_ALIAS]

        # Send the PUT request to update the user
        update_url = api.update_user(user_uuid)
        update_response = await async_client.put(update_url, json=update_payload, headers=user_headers)

        # Assert the response status and content
        assert update_response.status_code == 200
        data = update_response.json()
        assert data["success"] is True
        expected_message = f"Successfully updated user with UUID: {user_uuid}"
        assert expected_message in data["message"]

    @staticmethod
    def _update_payload():
        return {
            "job_title": "Senior Software Engineer",
            "region": "South",
            "sector": "Technology",
            "organisation": "New Tech Corp",
            "grade": "B",
            "communicator_role": False,
        }

    async def test_update_nonexistent_user(self, async_client, auth_token_only_headers):
        """With the session check off, a token-only update of an unknown user still returns 404."""
        nonexistent_uuid = str(uuid.uuid4())

        response = await async_client.put(
            api.update_user(nonexistent_uuid), json=self._update_payload(), headers=auth_token_only_headers
        )
        assert response.status_code == 404

    async def test_update_user_with_token_only_when_session_check_off(self, async_client, auth_token_only_headers):
        user_uuid = str(uuid.uuid4())
        create_response = await async_client.post(
            api.create_user(), json={"uuid": user_uuid}, headers=auth_token_only_headers
        )
        assert create_response.status_code == 201

        response = await async_client.put(
            api.update_user(user_uuid), json=self._update_payload(), headers=auth_token_only_headers
        )
        assert response.status_code == 200, response.text

    async def test_update_user_requires_session_auth(self, async_client, auth_token_only_headers, session_check_on):
        user_uuid = str(uuid.uuid4())
        headers = {**auth_token_only_headers, "User-Key-UUID": user_uuid}

        response = await async_client.put(api.update_user(user_uuid), json=self._update_payload(), headers=headers)
        assert response.status_code == 400, response.text

    async def test_update_user_requires_user_key_uuid(self, async_client, auth_token_only_headers, session_check_on):
        response = await async_client.put(
            api.update_user(str(uuid.uuid4())), json=self._update_payload(), headers=auth_token_only_headers
        )
        assert response.status_code == 400, response.text

    async def test_update_another_user_is_rejected(self, async_client, default_headers, session_check_on):
        """A caller can't update another user's profile, even with a valid Session-Auth of their own."""
        response = await async_client.put(
            api.update_user(str(uuid.uuid4())), json=self._update_payload(), headers=default_headers
        )
        assert response.status_code == 403, response.text

    async def test_update_own_user_with_session(self, async_client, user_id, default_headers, session_check_on):
        response = await async_client.put(
            api.update_user(user_id), json=self._update_payload(), headers=default_headers
        )
        assert response.status_code == 200, response.text

    async def test_create_auth_session_all_user_properties(self, async_client, auth_token_only_headers):
        """
        Create a user with all properties provided, then create an auth session for that user.
        Test the auth session has been created successfully and returns a valid UUID.
        """
        # Generate a new UUID for the user.
        user_uuid = str(uuid.uuid4())
        user_payload = {
            "uuid": user_uuid,
            "job_title": "Software Engineer",
            "region": "North",
            "sector": "Technology",
            "organisation": "Tech Corp",
            "grade": "A",
            "communicator_role": True,
        }

        # Create the user via the /users endpoint.
        create_user_url = api.create_user()
        create_response = await async_client.post(create_user_url, json=user_payload, headers=auth_token_only_headers)
        assert create_response.status_code == 201, (
            f"User creation failed with status code {create_response.status_code}"
        )

        headers = {**auth_token_only_headers, "User-Key-UUID": user_uuid}
        session_url = api.get_sessions()
        session_response = await async_client.post(session_url, headers=headers)
        assert session_response.status_code == 200, (
            f"Expected 200 OK from session creation, got {session_response.status_code}"
        )

        data = session_response.json()
        session_uuid = data.get(SESSION_AUTH_ALIAS)
        assert session_uuid is not None, "Session UUID missing in response"

        # Verify that the returned session UUID is valid.
        try:
            uuid.UUID(session_uuid)
        except ValueError as err:
            raise AssertionError(f"Returned session UUID is not valid: {session_uuid}") from err

    async def test_create_auth_session_with_null_user_properties(self, async_client, auth_token_only_headers):
        """
        Create a user with some null properties, then create an auth session for that user.
        """
        # Generate a new UUID for the user.
        user_uuid = str(uuid.uuid4())
        user_payload = {
            "uuid": user_uuid,
            "job_title": None,
            "region": None,
            "sector": None,
            "organisation": None,
            "grade": "B",
            "communicator_role": None,
        }

        # Create the user (with some null properties) via the /users endpoint.
        create_user_url = api.create_user()
        create_response = await async_client.post(create_user_url, json=user_payload, headers=auth_token_only_headers)
        assert create_response.status_code == 201, (
            f"User creation (with null properties) failed with status code {create_response.status_code}"
        )

        # Call the session creation endpoint using the "User-Key-UUID" header.
        headers = {**auth_token_only_headers, "User-Key-UUID": user_uuid}
        session_url = api.get_sessions()
        session_response = await async_client.post(session_url, headers=headers)
        assert session_response.status_code == 200, (
            f"Expected 200 OK from session creation, got {session_response.status_code}"
        )

        data = session_response.json()
        session_uuid = data.get(SESSION_AUTH_ALIAS)
        assert session_uuid is not None, "Session UUID missing in response"

        # Verify that the returned session UUID is valid.
        try:
            uuid.UUID(session_uuid)
        except ValueError as err:
            raise AssertionError(f"Returned session UUID is not valid: {session_uuid}") from err
