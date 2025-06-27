import json
import logging
import uuid

import pytest
from pydantic import ValidationError

from app.api import ENDPOINTS
from tests.mock_request import fail_test

logger = logging.getLogger(__name__)

api = ENDPOINTS()


def validate_user_prompt_response(response_data):
    try:
        logger.debug("Validating user prompt response structure and data app_types.")
        if "uuid" not in response_data:
            fail_test("UUID key is missing in response data")
        uuid.UUID(response_data["uuid"])
        logger.debug("Valid UUID confirmed for user prompt response.")

        # Check for 'title' key and validate it
        logger.debug(f"response_data: {response_data}")
        if "title" not in response_data:
            fail_test("title key is missing in chat response data")
        if not isinstance(response_data["title"], str):
            fail_test("title key is not of str type")
        logger.debug("title key is present and is of str type : " + str(response_data["title"]))

        # Check for 'content' key and validate it
        if "content" not in response_data:
            fail_test("content key is missing in chat response data")
        if not isinstance(response_data["content"], str):
            fail_test("content key is not of str type")
        logger.debug("content key is present and is of str type : " + str(response_data["content"]))

    except (ValidationError, ValueError, KeyError) as e:
        fail_test("Validation failed", e)


class TestPPLUserPrompts:
    # Tests for GET requests to /user/{user_uuid}/prompts
    # Test the happy path
    async def test_get_user_prompts_uuid_success(self, async_client, user_id, async_http_requester, session):
        logger.debug("Test the happy path for GET requests to /user/{user_uuid}/prompts/}")

        # create a new user prompt
        logger.debug(f"Creating user prompt for user int ID: {user_id}")

        post_url = api.create_user_prompt(user_uuid=user_id)
        logger.debug(f"POST URL for user ID: {user_id}, url: {post_url}")

        payload_1 = {
            "title": "testing, testing, 1, 2, 3...",
            "content": "Test prompt body. You can be a tree, which tree would you like to be?",
        }

        post_response_1 = await async_http_requester(
            "first POST to user_prompts_endpoint", async_client.post, post_url, json=payload_1
        )
        logger.debug(f"post_response: {post_response_1}")

        payload_2 = {"title": "Hello, hello...", "content": "Is it me you're looking for?"}

        post_response_2 = await async_http_requester(
            "second POST to user_prompts_endpoint", async_client.post, post_url, json=payload_2
        )
        logger.debug(f"post_response: {post_response_2}")

        # get all user prompts
        get_url = api.get_user_prompts(user_uuid=user_id)
        body = await async_http_requester("get all PPL user prompts by user UUID", async_client.get, get_url)

        logger.debug(f"GET Response body: {body}")

        assert body, "The response was empty."
        assert body != "", "The response was empty."
        assert isinstance(body["user_prompts"], list), "The response was not a list."

    # Tests for GET requests to /user/{user_uuid}/prompts
    # Test the 400 response (Unauthorised) path
    async def test_get_user_prompts_uuid_unauthorised(self, async_client, user_id, async_http_requester, session):
        logger.debug("Test the 400 response (Unauthorised) path for GET requests to /user/{user_uuid}/prompts/}")

        non_existent_user_id = uuid.uuid4()
        logger.debug(f"Overriding user_id: {user_id} with {non_existent_user_id}")

        # get all user prompts
        get_url = api.get_user_prompts(user_uuid=non_existent_user_id)
        get_response = await async_http_requester(
            "get all PPL user prompts by user UUID", async_client.get, get_url, response_code=400
        )

        logger.debug(f"GET Response body: {get_response}")
        assert {"detail": "user UUIDs do not match"} == get_response

    # Tests for GET requests to /user/{user_uuid}/prompts
    # Test the 200 response for no records found path
    async def test_get_user_prompts_uuid_no_records(self, async_client, user_id, async_http_requester, session):
        logger.debug("Test the 200 response for no records found path for GET requests to /user/{user_uuid}/prompts/}")

        # get all user prompts
        get_url = api.get_user_prompts(user_uuid=user_id)
        get_response = await async_http_requester(
            "get all PPL user prompts by user UUID",
            async_client.get,
            get_url,
            response_code=200,
            response_type=None,
        )

        logger.debug(f"GET Response body: {get_response}")
        assert b'{"user_prompts":[]}' == get_response

    # Tests for GET requests to /user/{user_uuid}/prompts
    # Test the 200 response for no records found path, make sure no soft-deleted records are returned
    async def test_get_user_prompts_uuid_without_deleted_records(
        self, async_client, user_id, async_http_requester, session
    ):
        logger.debug(
            "Test the 200 response for soft deleted records path for GET requests to /user/{user_uuid}/prompts/}"
        )

        # create a new user prompt
        logger.debug(f"Creating user prompt for user int ID: {user_id}")

        post_url = api.create_user_prompt(user_uuid=user_id)
        logger.debug(f"POST URL for user ID: {user_id}, url: {post_url}")

        payload_1 = {
            "title": "testing, testing, 1, 2, 3...",
            "content": "Test prompt body. You can be a tree, which tree would you like to be?",
        }

        post_response_1 = await async_http_requester(
            "first POST to user_prompts_endpoint", async_client.post, post_url, json=payload_1
        )
        logger.debug(f"post_response: {post_response_1}")

        payload_2 = {"title": "Hello, hello...", "content": "Is it me you're looking for?"}

        post_response_2 = await async_http_requester(
            "second POST to user_prompts_endpoint", async_client.post, post_url, json=payload_2
        )
        logger.debug(f"post_response: {post_response_2}")

        # get all user prompts
        get_url = api.get_user_prompts(user_uuid=user_id)
        get_response_1 = await async_http_requester(
            "get all PPL user prompts by user UUID",
            async_client.get,
            get_url,
            response_code=200,
            response_type=None,
        )

        logger.debug(f"get_response_1 = {get_response_1}")

        assert 2 == len(json.loads(get_response_1)["user_prompts"])

        # delete user prompt
        logger.debug(f"Deleting user prompt id: {post_response_1['uuid']} for user ID: {user_id}")

        delete_url = api.delete_user_prompt(user_uuid=user_id, user_prompt_uuid=post_response_1["uuid"])
        logger.debug(f"DELETE URL for user ID: {user_id}, url: {get_url}")

        delete_response = await async_http_requester(
            "DELETE from user_prompt_endpoint",
            async_client.delete,
            delete_url,
            response_code=204,
            response_type=None,
        )

        logger.debug(f"DELETE Response body: {delete_response}")

        get_response_2 = await async_http_requester(
            "get all PPL user prompts by user UUID",
            async_client.get,
            get_url,
            response_code=200,
            response_type=None,
        )
        logger.debug(f"GET Response body: {get_response_2}")

        assert 1 == len(json.loads(get_response_2)["user_prompts"])


class TestPPLUserPrompt:
    # Tests for POST requests to /user/{user_uuid}/prompts
    # Test the happy path
    @pytest.mark.asyncio
    async def test_post_user_prompt_success(
        self, async_client, user_id, async_http_requester, session, user_prompt_item
    ):
        # create a new user prompt
        logger.debug(f"Creating user prompt for user int ID: {user_id}")

        post_url = api.create_user_prompt(user_uuid=user_id)
        logger.debug(f"POST URL for user ID: {user_id}, url: {post_url}")

        payload = {
            "title": "testing, testing, 1, 2, 3...",
            "content": "Test prompt body. You can be a tree, which tree would you like to be?",
        }

        # create a new user prompt
        post_response = user_prompt_item
        logger.debug(f"post_response: {post_response}")

        validate_user_prompt_response(post_response)
        assert payload["title"] == post_response["title"]
        assert payload["content"] == post_response["content"]

    # Tests for POST requests to /user/{user_uuid}/prompts/
    # Test the 400 response (Unauthorised) path
    async def test_post_user_prompt_uuid_unauthorised(
        self, async_client, user_id, async_http_requester, session, user_prompt_item, user_prompt_payload
    ):
        logger.debug(
            "Test the 400 response (Unauthorised) path for GET requests to /user/{user_uuid}/prompts/{user_prompt_uuid}"
        )

        # create a new user prompt
        non_existent_user_id = uuid.uuid4()
        logger.debug(f"Overriding user_id: {user_id} with {non_existent_user_id}")

        post_url = api.create_user_prompt(user_uuid=non_existent_user_id)
        logger.debug(f"POST URL for user ID: {user_id}, url: {post_url}")

        payload = {
            "title": "testing, testing, 1, 2, 3...",
            "content": "Test prompt body. You can be a tree, which tree would you like to be?",
        }

        logger.debug(f"Creating user prompt for user int ID: {non_existent_user_id}")
        post_response = await async_http_requester(
            "POST to user_prompts_endpoint",
            async_client.post,
            post_url,
            json=payload,
            response_code=400,
            response_type=None,
        )
        logger.debug(f"post_response: {post_response}")

        assert b'{"detail":"user UUIDs do not match"}' == post_response

    # Tests for GET requests to /user/{user_uuid}/prompts/{user_prompt_uuid}
    # Test the happy path
    async def test_get_user_prompt_uuid_success(
        self, async_client, user_id, async_http_requester, session, user_prompt_item, user_prompt_payload
    ):
        logger.debug("Test the happy path for GET requests to /user/{user_uuid}/prompts/{user_prompt_uuid}")

        # create a new user prompt
        post_response = user_prompt_item
        logger.debug(f"post_response: {post_response}")

        # retrieve user prompt
        logger.debug(f"Retrieving user prompt id: {post_response['uuid']} for user ID: {user_id}")

        get_url = api.get_user_prompt(user_uuid=user_id, user_prompt_uuid=post_response["uuid"])
        logger.debug(f"URL for user ID: {user_id}, url: {get_url}")

        get_response = await async_http_requester("GET from user_prompt_endpoint", async_client.get, get_url)

        logger.debug(f"Response body: {get_response}")

        validate_user_prompt_response(get_response)
        assert post_response["uuid"] == get_response["uuid"]
        assert post_response["title"] == get_response["title"]
        assert post_response["content"] == get_response["content"]

    # Tests 404 for GET requests to /user/{user_uuid}/prompts/{user_prompt_uuid}
    # Test the 404 response for deleted records
    async def test_get_user_prompt_uuid_404_for_deleted_record(
        self, async_client, user_id, async_http_requester, session, user_prompt_item, user_prompt_payload
    ):
        logger.debug(
            "Test the 404 response for GET for deleted records request to /user/{user_uuid}/prompts/{user_prompt_uuid}"
        )

        # create a new user prompt
        post_response = user_prompt_item
        logger.debug(f"post_response: {post_response}")

        # retrieve user prompt
        logger.debug(f"Retrieving user prompt id: {post_response['uuid']} for user ID: {user_id}")

        get_url = api.get_user_prompt(user_uuid=user_id, user_prompt_uuid=post_response["uuid"])
        logger.debug(f"URL for user ID: {user_id}, url: {get_url}")

        get_response = await async_http_requester("GET from user_prompt_endpoint", async_client.get, get_url)

        logger.debug(f"Response body: {get_response}")

        validate_user_prompt_response(get_response)
        assert post_response["uuid"] == get_response["uuid"]
        assert post_response["title"] == get_response["title"]
        assert post_response["content"] == get_response["content"]

        # delete user prompt
        logger.debug(f"Deleting user prompt id: {post_response['uuid']} for user ID: {post_response['user_id']}")

        delete_url = api.delete_user_prompt(user_uuid=user_id, user_prompt_uuid=post_response["uuid"])
        logger.debug(f"DELETE URL for user ID: {user_id}, url: {delete_url}")

        delete_response = await async_http_requester(
            "DELETE from user_prompt_endpoint",
            async_client.delete,
            delete_url,
            response_code=204,
            response_type=None,
        )

        logger.debug(f"DELETE Response body: {delete_response}")

        get_response_after_delete = await async_http_requester(
            "get all PPL user prompts by user UUID",
            async_client.get,
            get_url,
            response_code=404,
            response_type=None,
        )
        logger.debug(f"GET Response body: {get_response_after_delete}")

        assert (b'"Record not found"') == get_response_after_delete

    # Tests for GET requests to /user/{user_uuid}/prompts/{user_prompt_uuid}
    # Test the 400 response (Unauthorised) path
    async def test_get_user_prompt_uuid_unauthorised(
        self, async_client, user_id, async_http_requester, session, user_prompt_item, user_prompt_payload
    ):
        logger.debug(
            "Test the 400 response (Unauthorised) path for GET requests to /user/{user_uuid}/prompts/{user_prompt_uuid}"
        )

        # create a new user prompt
        post_response = user_prompt_item
        logger.debug(f"post_response: {post_response}")

        # retrieve user prompt
        logger.debug(f"Retrieving user prompt id: {post_response['uuid']} for user ID: {user_id}")
        non_existent_user_id = uuid.uuid4()
        logger.debug(f"Overriding user_id: {user_id} with {non_existent_user_id}")

        get_url = api.get_user_prompt(user_uuid=non_existent_user_id, user_prompt_uuid=post_response["uuid"])
        logger.debug(f"URL for user ID: {non_existent_user_id}, url: {get_url}")

        get_response = await async_http_requester(
            "GET from user_prompt_endpoint",
            async_client.get,
            get_url,
            response_code=400,
            response_type=None,
        )

        logger.debug(f"GET Response body: {get_response}")
        assert (b'{"detail":"user UUIDs do not match"}') == get_response

    # Tests for GET requests to /user/{user_uuid}/prompts/{user_prompt_uuid}
    # Test the 404 response (Not Found) path
    async def test_get_user_prompt_uuid_not_found(self, async_client, user_id, async_http_requester, session):
        logger.debug(
            "Test the 404 response (Not Found) path for GET requests to /user/{user_uuid}/prompts/{user_prompt_uuid}"
        )

        # create a new user prompt
        logger.debug(f"Creating user prompt for user int ID: {user_id}")

        post_url = api.create_user_prompt(user_uuid=user_id)
        logger.debug(f"POST URL for user ID: {user_id}, url: {post_url}")

        payload = {
            "title": "testing, testing, 1, 2, 3...",
            "content": "Test prompt body. You can be a tree, which tree would you like to be?",
        }

        post_response = await async_http_requester(
            "POST to user_prompts_endpoint", async_client.post, post_url, json=payload
        )
        logger.debug(f"post_response: {post_response}")

        # retrieve user prompt
        non_existent_user_prompt_uuid = uuid.uuid4()
        logger.debug(f"Using non-existent user_prompt_uuid: {non_existent_user_prompt_uuid}")

        get_url = api.get_user_prompt(user_uuid=user_id, user_prompt_uuid=non_existent_user_prompt_uuid)
        logger.debug(f"URL for user prompt ID: {non_existent_user_prompt_uuid}, url: {get_url}")

        get_response = await async_http_requester(
            "GET from user_prompt_endpoint",
            async_client.get,
            get_url,
            response_code=404,
            response_type=None,
        )

        logger.debug(f"GET Response body: {get_response}")
        assert (b'"Record not found"') == get_response

    # Tests for PATCH requests to /user/{user_uuid}/prompts/{user_prompt_uuid}
    # Test the happy path
    async def test_patch_user_prompt_uuid_success(
        self, async_client, user_id, async_http_requester, session, user_prompt_item, user_prompt_payload
    ):
        logger.debug("Test the happy path for PATCH requests to /user/{user_uuid}/prompts/{user_prompt_uuid}")

        # create a new user prompt
        post_response = user_prompt_item
        logger.debug(f"post_response: {post_response}")

        # update user prompt
        patch_url = api.patch_user_prompt(user_uuid=user_id, user_prompt_uuid=post_response["uuid"])
        logger.debug(f"PATCH URL for user ID: {user_id}, url: {patch_url}")

        patch_payload = {
            "title": "UPDATED testing, testing, 1, 2, 3...",
            "content": "UPDATED Test prompt body. You can be a tree, which tree would you like to be?",
        }

        patch_response = await async_http_requester(
            "PATCH from user_prompt_endpoint", async_client.patch, patch_url, json=patch_payload
        )

        logger.debug(f"PATCH Response body: {patch_response}")

        validate_user_prompt_response(patch_response)
        assert post_response["uuid"] == patch_response["uuid"]
        assert patch_payload["title"] == patch_response["title"]
        assert patch_payload["content"] == patch_response["content"]

    # Tests 404 for PATCH requests to /user/{user_uuid}/prompts/{user_prompt_uuid}
    # Test the 404 response for PATCH for deleted records request
    async def test_patch_user_prompt_uuid_404_for_deleted_record(
        self, async_client, user_id, async_http_requester, session, user_prompt_item, user_prompt_payload
    ):
        logger.debug(
            "Test the 404 response for PATCH for deleted records request to "
            "/user/{user_uuid}/prompts/{user_prompt_uuid}"
        )

        # create a new user prompt
        post_response = user_prompt_item
        logger.debug(f"post_response: {post_response}")

        # update user prompt
        patch_url = api.patch_user_prompt(user_uuid=user_id, user_prompt_uuid=post_response["uuid"])
        logger.debug(f"PATCH URL for user ID: {user_id}, url: {patch_url}")

        patch_payload = {
            "title": "UPDATED testing, testing, 1, 2, 3...",
            "content": "UPDATED Test prompt body. You can be a tree, which tree would you like to be?",
        }

        patch_response = await async_http_requester(
            "PATCH from user_prompt_endpoint", async_client.patch, patch_url, json=patch_payload
        )

        logger.debug(f"PATCH Response body: {patch_response}")

        validate_user_prompt_response(patch_response)
        assert post_response["uuid"] == patch_response["uuid"]
        assert patch_payload["title"] == patch_response["title"]
        assert patch_payload["content"] == patch_response["content"]

        # delete user prompt
        logger.debug(f"Deleting user prompt id: {post_response['uuid']} for user ID: {post_response['user_id']}")

        delete_url = api.delete_user_prompt(user_uuid=user_id, user_prompt_uuid=post_response["uuid"])
        logger.debug(f"DELETE URL for user ID: {user_id}, url: {delete_url}")

        delete_response = await async_http_requester(
            "DELETE from user_prompt_endpoint",
            async_client.delete,
            delete_url,
            response_code=204,
            response_type=None,
        )

        logger.debug(f"DELETE Response body: {delete_response}")

        get_response_after_delete = await async_http_requester(
            "get all PPL user prompts by user UUID",
            async_client.get,
            patch_url,
            response_code=404,
            response_type=None,
        )
        logger.debug(f"GET Response body: {get_response_after_delete}")

        assert (b'"Record not found"') == get_response_after_delete

    # Tests for PATCH requests to /user/{user_uuid}/prompts/{user_prompt_uuid}
    # Test the 400 response (Unauthorised) path
    async def test_patch_user_prompt_uuid_unauthorised(
        self, async_client, user_id, async_http_requester, session, user_prompt_item, user_prompt_payload
    ):
        logger.debug(
            "Test the 400 response (Unauthorised) path for PATCH requests to "
            + "/user/{user_uuid}/prompts/{user_prompt_uuid}"
        )

        # create a new user prompt
        post_response = user_prompt_item
        logger.debug(f"post_response: {post_response}")

        # update user prompt
        non_existent_user_id = uuid.uuid4()
        logger.debug(f"Overriding user_id: {user_id} with {non_existent_user_id}")

        patch_url = api.patch_user_prompt(user_uuid=non_existent_user_id, user_prompt_uuid=post_response["uuid"])
        logger.debug(f"PATCH URL for user ID: {user_id}, url: {patch_url}")

        patch_payload = {
            "title": "UPDATED testing, testing, 1, 2, 3...",
            "content": "UPDATED Test prompt body. You can be a tree, which tree would you like to be?",
        }

        patch_response = await async_http_requester(
            "PATCH from user_prompt_endpoint",
            async_client.patch,
            patch_url,
            json=patch_payload,
            response_code=400,
            response_type=None,
        )

        logger.debug(f"PATCH Response body: {patch_response}")
        assert (b'{"detail":"user UUIDs do not match"}') == patch_response

    # Tests for PATCH requests to /user/{user_uuid}/prompts/{user_prompt_uuid}
    # Test the 404 response (Not Found) path
    async def test_patch_user_prompt_uuid_not_found(self, async_client, user_id, async_http_requester, session):
        logger.debug(
            "Test the 404 response (Not Found) path for PATCH requests to /user/{user_uuid}/prompts/{user_prompt_uuid}"
        )

        # update user prompt
        non_existent_user_prompt_uuid = uuid.uuid4()
        logger.debug(f"Using non-existent user_prompt_uuid: {non_existent_user_prompt_uuid}")

        patch_url = api.patch_user_prompt(user_uuid=user_id, user_prompt_uuid=non_existent_user_prompt_uuid)
        logger.debug(f"PATCH URL for user prompt ID: {non_existent_user_prompt_uuid}, url: {patch_url}")

        patch_payload = {
            "title": "UPDATED testing, testing, 1, 2, 3...",
            "content": "UPDATED Test prompt body. You can be a tree, which tree would you like to be?",
        }

        patch_response = await async_http_requester(
            "PATCH from user_prompt_endpoint",
            async_client.patch,
            patch_url,
            json=patch_payload,
            response_code=404,
            response_type=None,
        )

        logger.debug(f"PATCH Response body: {patch_response}")
        assert (b'"Record not found"') == patch_response

    # Tests for DELETE requests to /user/{user_uuid}/prompts/{user_prompt_uuid}
    # Test the happy path
    async def test_delete_user_prompt_uuid_success(
        self, async_client, user_id, async_http_requester, session, user_prompt_item, user_prompt_payload
    ):
        logger.debug("Test the happy path for DELETE requests to /user/{user_uuid}/prompts/{user_prompt_uuid}")

        # create a new user prompt
        post_response = user_prompt_item
        logger.debug(f"post_response: {post_response}")

        # delete user prompt
        logger.debug(f"Deleting user prompt id: {post_response['uuid']} for user ID: {user_id}")

        get_url = api.delete_user_prompt(user_uuid=user_id, user_prompt_uuid=post_response["uuid"])
        logger.debug(f"DELETE URL for user ID: {user_id}, url: {get_url}")

        delete_response = await async_http_requester(
            "DELETE from user_prompt_endpoint",
            async_client.delete,
            get_url,
            response_code=204,
            response_type=None,
        )

        logger.debug(f"DELETE Response body: {delete_response}")

        assert b"" == delete_response

    # Tests for DELETE requests to /user/{user_uuid}/prompts/{user_prompt_uuid}
    # Test the 400 response (Unauthorised) path
    async def test_delete_user_prompt_uuid_unauthorised(
        self, async_client, user_id, async_http_requester, session, user_prompt_item, user_prompt_payload
    ):
        logger.debug(
            "Test the 400 response (Unauthorised) path for DELETE requests to "
            + "/user/{user_uuid}/prompts/{user_prompt_uuid}"
        )

        # create a new user prompt
        post_response = user_prompt_item
        logger.debug(f"post_response: {post_response}")

        # delete user prompt
        logger.debug(f"Deleting user prompt id: {post_response['uuid']} for user ID: {user_id}")

        non_existent_user_id = uuid.uuid4()
        logger.debug(f"Overriding user_id: {user_id} with {non_existent_user_id}")

        delete_url = api.delete_user_prompt(user_uuid=non_existent_user_id, user_prompt_uuid=post_response["uuid"])
        logger.debug(f"DELETE URL for user ID: {non_existent_user_id}, url: {delete_url}")

        delete_response = await async_http_requester(
            "DELETE from user_prompt_endpoint",
            async_client.delete,
            delete_url,
            response_code=400,
            response_type=None,
        )

        logger.debug(f"DELETE Response body: {delete_response}")
        assert (b'{"detail":"user UUIDs do not match"}') == delete_response

    # Tests for DELETE requests to /user/{user_uuid}/prompts/{user_prompt_uuid}
    # Test the 404 response (Not Found) path
    async def test_delete_user_prompt_uuid_not_found(self, async_client, user_id, async_http_requester, session):
        logger.debug(
            "Test the 404 response (Not Found) path for DELETE requests to /user/{user_uuid}/prompts/{user_prompt_uuid}"
        )

        non_existent_user_prompt_uuid = uuid.uuid4()
        logger.debug(f"Using non-existent user_prompt_uuid: {non_existent_user_prompt_uuid}")

        delete_url = api.delete_user_prompt(user_uuid=user_id, user_prompt_uuid=non_existent_user_prompt_uuid)
        logger.debug(f"DELETE URL for user prompt ID: {non_existent_user_prompt_uuid}, url: {delete_url}")

        delete_response = await async_http_requester(
            "DELETE from user_prompt_endpoint",
            async_client.delete,
            delete_url,
            response_code=404,
            response_type=None,
        )

        logger.debug(f"DELETE Response body: {delete_response}")
        assert (b'"Record not found"') == delete_response
