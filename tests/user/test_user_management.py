import logging
import uuid

from app.api import ENDPOINTS, ApiConfig

api = ENDPOINTS()
logger = logging.getLogger(__name__)


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

        # Send the PUT request to update the user
        update_url = api.update_user(user_uuid)
        update_response = await async_client.put(update_url, json=update_payload, headers=auth_token_only_headers)

        # Assert the response status and content
        assert update_response.status_code == 200
        data = update_response.json()
        assert data["success"] is True
        expected_message = f"Successfully updated user with UUID: {user_uuid}"
        assert expected_message in data["message"]

    async def test_update_nonexistent_user(self, async_client, async_http_requester, auth_token_only_headers):
        # Generate a random UUID that shouldn't exist in the system

        nonexistent_uuid = str(uuid.uuid4())

        update_payload = {
            "job_title": "Senior Software Engineer",
            "region": "South",
            "sector": "Technology",
            "organisation": "New Tech Corp",
            "grade": "B",
            "communicator_role": False,
        }

        update_url = api.update_user(nonexistent_uuid)

        response = await async_client.put(update_url, json=update_payload, headers=auth_token_only_headers)
        assert response.status_code == 404

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
        session_uuid = data.get(ApiConfig.SESSION_AUTH_ALIAS)
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
        session_uuid = data.get(ApiConfig.SESSION_AUTH_ALIAS)
        assert session_uuid is not None, "Session UUID missing in response"

        # Verify that the returned session UUID is valid.
        try:
            uuid.UUID(session_uuid)
        except ValueError as err:
            raise AssertionError(f"Returned session UUID is not valid: {session_uuid}") from err
