import logging
import uuid
from decimal import Decimal
from unittest.mock import patch

import pytest
from pydantic import ValidationError

# from pydantic import ValidationError
from sqlalchemy.future import select

from app.api.endpoints import ENDPOINTS
from app.auth.constants import AUTH_TOKEN_ALIAS, SESSION_AUTH_ALIAS, USER_KEY_UUID_ALIAS
from app.bedrock import BedrockHandler
from app.bedrock.schemas import LLMTransaction
from app.chat.schemas import ChatWithLatestMessage, ItemTitleResponse
from app.database.models import LLM, Chat, Message
from app.database.table import ChatTable
from tests.mock_request import fail_test

api = ENDPOINTS()


logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.streaming,
]


@pytest.mark.asyncio
async def test_stream_response(default_headers, user_id, async_client, async_http_requester):
    """
    Test the chat stream response returns an event stream response when the user creates a chat stream.
    Since FastAPI test client does not support true streaming / http response chunking,
    here only content type is checked.
    """
    from app.api.endpoints import ENDPOINTS

    logger.debug(f"Creating chat stream for user ID: {user_id}")

    api_class = ENDPOINTS()
    endpoint = api_class.create_chat_stream(user_uuid=user_id)
    response_data = await async_http_requester(
        "test_stream_response",
        async_client.post,
        endpoint,
        response_type="text",
        response_content_type="text/event-stream; charset=utf-8",
        json={"query": "write a 20-word single paragraph about GCS", "use_rag": False},
    )

    print(response_data)
    assert response_data is not None


@pytest.mark.asyncio
async def test_stream_response_valid_json(default_headers, user_id, async_client, async_http_requester):
    """
    Tests that the response from a chat stream contains valid JSON packets.
    This test will fail if JSON packets are malformed or concatenated incorrectly.
    """
    import json
    import re

    from app.api.endpoints import ENDPOINTS

    logger.debug(f"Testing JSON validity for chat stream, user ID: {user_id}")

    api_class = ENDPOINTS()
    endpoint = api_class.create_chat_stream(user_uuid=user_id)

    # This prompt was failing on production due to the words 'Abby'
    # being appended to the end of the response, outside of the JSON packet.
    prompt = """can you make this email more engaging: Good Morning!
Thank you for signing up to be part of the Cabinet Office Gemini pilot.
By now you should have access to the tool and explore how it can support you in your daily workflows.
As part of this pilot, GCS are looking at how we can best apply the AI tools we have access to \
in order to unlock efficiencies across teams and build overall AI capabilities.
Can you please complete this initial survey to help us understand our current AI landscape:
https://forms.gle/a274zEowvbHZJD5v7
Best regards,
John
"""

    response_data = await async_http_requester(
        "test_stream_json_validity",
        async_client.post,
        endpoint,
        response_type="text",
        response_content_type="text/event-stream; charset=utf-8",
        json={"query": prompt, "use_rag": False},
    )

    # Convert bytes to string if needed
    if isinstance(response_data, bytes):
        response_text = response_data.decode("utf-8")
    else:
        response_text = str(response_data)

    # logger.info(f"response_text: {response_text}")

    # Split the response into JSON objects
    # Use regex to find JSON objects - look for patterns starting with { and ending with }
    json_pattern = r"(\{[^{]*?\})"
    potential_json_objects = re.findall(json_pattern, response_text)

    # Verify each potential JSON object
    valid_json_objects = []
    for json_str in potential_json_objects:
        try:
            parsed = json.loads(json_str)
            valid_json_objects.append(parsed)
        except json.JSONDecodeError as e:
            pytest.fail(f"Failed to parse JSON packet: {json_str}\nError: {e}")

    # Ensure we got at least one valid JSON object
    assert len(valid_json_objects) > 0

    # Check that the number of valid JSON objects matches the number of potential objects
    # If they don't match, there might be incorrectly concatenated JSON
    assert len(valid_json_objects) == len(potential_json_objects), (
        f"Found {len(potential_json_objects)} potential JSON objects but only "
        f"{len(valid_json_objects)} were valid. This suggests malformed JSON."
    )

    # Show the last characters of the raw response
    logger.info(f"Last 10 characters of raw response: {response_text[-10:]}")

    # Assert that the last character is a }
    assert response_text[-1] == "}", f"The last character should be a '}}', got '{response_text[-1]}'"


def validate_message_response(message):
    try:
        if not isinstance(message["content"], str):
            fail_test(f"Message content '{message['content']}' is not a string")
        if message["role"] not in ["user", "assistant"]:
            fail_test(f"Role '{message['role']}' is not valid")

        print("returned successful message from response:", message)

    except (ValidationError, ValueError, KeyError) as e:
        fail_test("Validation failed", e)


def validate_chat_response(response_data, message=True):
    try:
        logger.debug("Validating chat response structure and data app_types.")
        if "uuid" not in response_data:
            fail_test("UUID key is missing in response data")
        uuid.UUID(response_data["uuid"])
        logger.debug("Valid UUID confirmed for chat response.")

        # Check for 'use_rag' key and validate it
        if "use_rag" not in response_data:
            fail_test("use_rag key is missing in chat response data")
        if not isinstance(response_data["use_rag"], bool):
            fail_test("use_rag key is not of boolean type")
        logger.debug("use_rag key is present and is of boolean type : " + str(response_data["use_rag"]))

        # check for use_gov_uk_search_api key and validate it
        if "use_gov_uk_search_api" not in response_data:
            fail_test("use_gov_uk_search_api key is missing in chat response data")
        if not isinstance(response_data["use_gov_uk_search_api"], bool):
            fail_test("use_gov_uk_search_api key is not of boolean type")
        logger.debug(
            "use_gov_uk_search_api key is present and is of boolean type : "
            + str(response_data["use_gov_uk_search_api"])
        )

        if message:
            if "message" not in response_data:
                fail_test("Message key is missing in response data")
            validate_message_response(response_data["message"])

    except (ValidationError, ValueError, KeyError) as e:
        fail_test("Validation failed", e)


class TestUserChats:
    # Tests for GET requests to /user/chats/{id}
    # Test the happy path
    @pytest.mark.asyncio
    async def test_get_user_chats_id(self, async_client, user_id, async_http_requester, session):
        logger.debug("Test the happy path for GET requests to /user/chats/{id}")

        get_url = api.get_chats_by_user(user_id)
        get_response = await async_http_requester("get all chats by user UUID", async_client.get, get_url)

        logging.info(f"GET Response body: {get_response}")

        assert get_response, "The response was empty."
        assert get_response != "", "The response was empty."
        assert isinstance(get_response["chats"], list), "The response was not a list."

    # Tests for GET requests to /user/chats/{id}
    # Test the 403 response (Forbidden) path
    @pytest.mark.asyncio
    async def test_get_user_chats_id_unauthorised(self, async_client, user_id, async_http_requester, session):
        logger.debug("Test the 403 response path for GET requests to /user/chats/{id}")

        non_existent_user_id = uuid.uuid4()
        logger.debug(f"Overriding user_id: {user_id} with {non_existent_user_id}")

        get_url = api.get_chats_by_user(non_existent_user_id)
        get_response = await async_http_requester(
            "get all chats by user UUID", async_client.get, get_url, response_code=403
        )

        logging.info(f"GET Response body: {get_response}")


class TestUserChatsV1:
    async def test_accessing_another_user_chat_denied(
        self,
        chat,
        user_id,
        async_client,
        async_http_requester,
        auth_token,
        another_user_auth_session,
    ):
        # create endpoint with other user
        non_owning_user = str(uuid.uuid4())
        # create session for other user
        other_session = await another_user_auth_session(non_owning_user)

        non_owning_user_session_params = {
            USER_KEY_UUID_ALIAS: non_owning_user,
            SESSION_AUTH_ALIAS: other_session,
            AUTH_TOKEN_ALIAS: auth_token,
        }
        chat_url = api.get_chat_item(non_owning_user, chat.uuid)
        response = await async_http_requester(
            "get_chat_item",
            async_client.get,
            chat_url,
            response_code=401,
            headers=non_owning_user_session_params,
        )
        assert response == {"detail": f"Access denied to chat '{chat.uuid}'"}

    async def test_accessing_another_user_chat_messages_denied(
        self,
        chat,
        user_id,
        async_client,
        async_http_requester,
        auth_token,
        another_user_auth_session,
    ):
        # create endpoint with other user
        non_owning_user = str(uuid.uuid4())
        # create session for other user
        other_session = await another_user_auth_session(non_owning_user)

        non_owning_user_session_params = {
            USER_KEY_UUID_ALIAS: non_owning_user,
            SESSION_AUTH_ALIAS: other_session,
            AUTH_TOKEN_ALIAS: auth_token,
        }
        chat_messages_url = api.get_chat_messages(non_owning_user, chat.uuid)
        response = await async_http_requester(
            "get_chat_messages",
            async_client.get,
            chat_messages_url,
            response_code=401,
            headers=non_owning_user_session_params,
        )
        assert response == {"detail": f"Access denied to chat '{chat.uuid}'"}

    async def test_posting_another_user_chat_messages_denied(
        self,
        chat,
        user_id,
        async_client,
        async_http_requester,
        auth_token,
        another_user_auth_session,
    ):
        # create endpoint with other user
        non_owning_user = str(uuid.uuid4())
        # create session for other user
        other_session = await another_user_auth_session(non_owning_user)

        non_owning_user_session_params = {
            USER_KEY_UUID_ALIAS: non_owning_user,
            SESSION_AUTH_ALIAS: other_session,
            AUTH_TOKEN_ALIAS: auth_token,
        }
        chat_messages_url = api.get_chat_item(non_owning_user, chat.uuid)
        response = await async_http_requester(
            "posting_to_another_user_chat",
            async_client.put,
            chat_messages_url,
            response_code=401,
            headers=non_owning_user_session_params,
        )
        assert response == {"detail": f"Access denied to chat '{chat.uuid}'"}

    async def test_create_chat_title_for_another_user_chat_denied(
        self,
        chat,
        user_id,
        async_client,
        async_http_requester,
        auth_token,
        another_user_auth_session,
    ):
        # create endpoint with other user
        non_owning_user = str(uuid.uuid4())
        # create session for other user
        other_session = await another_user_auth_session(non_owning_user)

        non_owning_user_session_params = {
            USER_KEY_UUID_ALIAS: non_owning_user,
            SESSION_AUTH_ALIAS: other_session,
            AUTH_TOKEN_ALIAS: auth_token,
        }
        chat_messages_url = api.create_chat_title(non_owning_user, chat.uuid)
        response = await async_http_requester(
            "create_chat_title",
            async_client.put,
            chat_messages_url,
            response_code=401,
            headers=non_owning_user_session_params,
        )
        assert response == {"detail": f"Access denied to chat '{chat.uuid}'"}

    async def test_add_message_to_chat_stream_for_another_user_chat_denied(
        self,
        chat,
        user_id,
        async_client,
        async_http_requester,
        auth_token,
        another_user_auth_session,
    ):
        # create endpoint with other user
        non_owning_user = str(uuid.uuid4())
        # create session for other user
        other_session = await another_user_auth_session(non_owning_user)

        non_owning_user_session_params = {
            USER_KEY_UUID_ALIAS: non_owning_user,
            SESSION_AUTH_ALIAS: other_session,
            AUTH_TOKEN_ALIAS: auth_token,
        }
        chat_messages_url = api.get_chat_stream(non_owning_user, chat.uuid)
        response = await async_http_requester(
            "create_chat_title",
            async_client.put,
            chat_messages_url,
            response_code=401,
            headers=non_owning_user_session_params,
        )
        assert response == {"detail": f"Access denied to chat '{chat.uuid}'"}

    @pytest.mark.asyncio
    async def test_post_chat(self, async_client, user_id, async_http_requester):
        logger.debug(f"Creating chat for user ID: {user_id}")
        url = api.chats(user_uuid=user_id)
        response = await async_http_requester(
            "chat_endpoint", async_client.post, url, json={"query": "hello", "use_rag": False}
        )
        validate_chat_response(response)

    @pytest.mark.asyncio
    async def test_post_chat_unauthorised(self, async_client, user_id, async_http_requester):
        logger.debug("Test the 403 response (Unauthorised) path for GET requests to /user/chats/")

        non_existent_user_id = uuid.uuid4()
        logger.debug(f"Overriding user_id: {user_id} with {non_existent_user_id}")

        logger.debug(f"Creating chat for user ID: {non_existent_user_id}")
        url = api.chats(user_uuid=non_existent_user_id)
        post_response = await async_http_requester(
            "chat_endpoint",
            async_client.post,
            url,
            response_code=403,
            json={"query": "hello", "use_rag": False},
        )
        logger.debug(f"GET Response body: {post_response}")

    @pytest.mark.asyncio
    async def test_get_chat(self, async_client, chat_item, user_id, async_http_requester):
        url = api.get_chat_item(user_id, chat_item["uuid"])
        logger.debug(f"GET chat_endpoint: {url}")
        data = await async_http_requester("test_get_chat", async_client.get, url)

        validate_chat_response(data, message=False)
        previous_timestamp = None
        for message in data["messages"]:
            validate_message_response(message)

            current_timestamp = message["created_at"]
            if previous_timestamp is not None:
                assert current_timestamp >= previous_timestamp, "Messages are not in the correct order"

            previous_timestamp = current_timestamp

    @pytest.mark.asyncio
    async def test_put_chat(self, async_client, chat_item, user_id, async_http_requester):
        url = api.get_chat_item(user_id, chat_item["uuid"])
        response = await async_http_requester(
            "test_put_chat", async_client.put, url, json={"query": "how are you", "use_rag": False}
        )

        validate_chat_response(response)
        logger.debug("test_put_chat passed.")

    @pytest.mark.asyncio
    async def test_long_response(self, async_client, user_id, async_http_requester):
        """
        Checks that a long response can be generated and is at least 1800 characters long.
        """
        # Although the test is for 1800 characters, we request the LLM to produce
        # an answer that is 2000 characters to give it room for error.

        # Previously, the default LiteLLM settings resulted in answers that
        # were about 1200 characters long and would stop mid-sentence.
        # The LiteLLM max_token has been manually set to counteract this.
        # On gpt-4-turbo-2024-04-09 the response to this prompt has 3914 characters (accessed on 2024-05-07).
        prompt_for_long_response = """Give me 10 ideas for using Large Language Models in GCS.
        For each idea identify a risk. Make sure the response is at least 2500 characters long."""
        logger.debug("Testing that a long response can be generated in full.")
        url = api.chats(user_uuid=user_id)
        response_data = await async_http_requester(
            "test_long_response",
            async_client.post,
            url,
            json={"query": prompt_for_long_response, "use_rag": False},
        )

        message = response_data["message"]
        if message["role"] == "assistant":
            test_string = message["content"]

            logger.debug(f"Long chat generated: '{test_string}'")
            if len(test_string) < 1800:
                fail_test(
                    f"The response from the LLM was not long enough. Length: {len(test_string)}. "
                    "Response: {test_string}"
                )

        logger.debug("test_long_response passed")

    # @pytest.mark.asyncio
    async def test_create_chat_title_does_not_interfere_with_previous_chats(
        self, async_client, chat, user_id, async_http_requester, caplog
    ):
        """
        Creates two chat and chat titles, and checks second chat title does not use messages constructed for the first
        chat. Each chat title creation should use its own messages, and should not interfere with other chat messages.
        """
        with patch.object(
            BedrockHandler,
            "create_chat_title",
            return_value=LLMTransaction(
                input_tokens=1,
                output_tokens=1,
                input_cost=0,
                output_cost=0,
                completion_cost=0,
                content="",
            ),
        ) as mock_create_chat_title:
            mocked_first_chat_content = "random chat text"

            create_chat_title_url = api.create_chat_title(user_uuid=user_id, chat_uuid=chat.uuid)
            await async_http_requester(
                "test_create_chat_title_does_not_interfere_with_previous_chats",
                async_client.put,
                create_chat_title_url,
                json={"query": mocked_first_chat_content, "use_rag": False},
            )

            args, kwargs = mock_create_chat_title.call_args

            assert len(args) == 1  # check only one-message length array constructed when calling LLM
            function_arg = args[0]
            title_content_dict = function_arg[0]  # it is a list with dictionary
            assert mocked_first_chat_content in title_content_dict["content"]

            # make another chat
            url = api.chats(user_uuid=user_id)
            response = await async_http_requester(
                "chat_endpoint",
                async_client.post,
                url,
                json={
                    "query": "generate a number between 0 and 10 and only include the number in the response",
                    "use_rag": False,
                },
            )
            mocked_second_chat_content = "this is a different chat"

            chat2 = ChatWithLatestMessage(**response)

            # generate chat title for the second chat
            create_chat_title_url = api.create_chat_title(user_uuid=user_id, chat_uuid=chat2.uuid)
            await async_http_requester(
                "test_create_chat_title_does_not_interfere_with_previous_chats",
                async_client.put,
                create_chat_title_url,
                json={"query": mocked_second_chat_content, "use_rag": False},
            )

            args, _ = mock_create_chat_title.call_args

            assert len(args) == 1  # check only one-message length array constructed when calling LLM
            function_arg = args[0]
            title_content_dict = function_arg[0]  # it is a list with dictionary

            assert mocked_first_chat_content not in title_content_dict["content"]
            assert mocked_second_chat_content in title_content_dict["content"]

    # Chat title generation tests
    async def test_create_chat_title_success(self, async_client, user_id, chat, async_http_requester):
        """
        Creates a new chat and tests the chat_create_title function with a successful response.
        Checks Chat title has been updated with the generated title in the database.
        """

        chat_content = chat.message.content
        create_chat_title_url = api.create_chat_title(user_uuid=user_id, chat_uuid=chat.uuid)
        title_success_response = await async_http_requester(
            "test_create_title_success",
            async_client.put,
            create_chat_title_url,
            json={"query": chat_content, "use_rag": False},
        )

        chat_response = ItemTitleResponse(**title_success_response)
        title = chat_response.title
        assert len(title) < 255, f"Chat title is too long (max 255 characters), received: {title}"
        chat_model = ChatTable().get_by_uuid(chat.uuid)
        assert chat_model.title == title

    @pytest.mark.asyncio
    async def test_create_chat_title_too_long_is_logged_and_trimmed(
        self, async_client, user_id, chat, async_http_requester, caplog
    ):
        """
        Creates a new chat and tests the chat_create_title function with a title that is too long.
        Mocks the BedrockHandler create_chat_title method to return a long response.
        Checks Chat title has been updated with the generated title in the database.
        """
        with patch.object(
            BedrockHandler,
            "create_chat_title",
            return_value=LLMTransaction(
                input_tokens=1,
                output_tokens=1,
                input_cost=0,
                output_cost=0,
                completion_cost=0,
                content="X" * 256,
            ),
        ):
            chat_content = chat.message.content
            create_chat_title_url = api.create_chat_title(user_uuid=user_id, chat_uuid=chat.uuid)
            title_response = await async_http_requester(
                "test_create_chat_title_too_long_is_logged",
                async_client.put,
                create_chat_title_url,
                json={"query": chat_content, "use_rag": False},
            )

            chat_response = ItemTitleResponse(**title_response)
            title = chat_response.title
            assert len(title) == 255
            assert "Title exceeds 255 characters. Truncating:" in caplog.text
            chat_model = ChatTable().get_by_uuid(chat.uuid)
            assert chat_model.title == title

    async def test_create_chat_title_throws_exception(self, async_client, user_id, chat, async_http_requester, caplog):
        """
        Creates a new chat and tests the chat_create_title function throwing an exception.
        Mocks the BedrockHandler create_chat_title method to throw an exception.
        Checks an exception is thrown and logged.
        """
        excepted_exception = Exception("An error occurred")
        with patch.object(BedrockHandler, "create_chat_title", side_effect=excepted_exception):
            chat_content = chat.message.content
            create_chat_title_url = api.create_chat_title(user_uuid=user_id, chat_uuid=chat.uuid)

            with pytest.raises(Exception) as ex:
                await async_http_requester(
                    "test_create_chat_title_too_long_is_logged",
                    async_client.put,
                    create_chat_title_url,
                    response_code=500,
                    json={"query": chat_content, "use_rag": False},
                )
                assert ex == excepted_exception

            assert "Error in chat_create_title:" in caplog.text

    # Chat message id and parent id association test
    @pytest.mark.asyncio
    async def test_create_chat_messages_linked_by_parent_id(
        self, async_client, user_id, chat, async_http_requester, db_session, caplog
    ):
        """
        Creates a new chat and sends two messages to the chat.
        Checks that messages are linked by parent_message_id in the database.
        Checks that the role of the messages are correct,
        where first message's role is user and second message's role is assistant.
        """

        # second text
        create_chat_message_url = api.get_chat_item(user_uuid=user_id, chat_uuid=chat.uuid)
        await async_http_requester(
            "test_create_chat_messages_linked_by_parent_id",
            async_client.put,
            create_chat_message_url,
            json={"query": "Shorten the answer", "use_rag": False},
        )

        # third text
        create_chat_message_url = api.get_chat_item(user_uuid=user_id, chat_uuid=chat.uuid)
        await async_http_requester(
            "test_create_chat_messages_linked_by_parent_id",
            async_client.put,
            create_chat_message_url,
            json={"query": "Lengthen the answer", "use_rag": False},
        )

        execute = await db_session.execute(select(Chat).filter(Chat.uuid == chat.uuid))
        chat_model = execute.scalar_one()
        execute = await db_session.execute(
            select(Message).filter(Message.chat_id == chat_model.id).order_by(Message.created_at)
        )
        messages = list(execute.scalars())
        assert len(messages) == 6, f"Expected 6 messages, but got {len(messages)}"
        for idx, message in enumerate(messages):
            if idx == 0:
                assert message.parent_message_id is None
            else:
                assert message.parent_message_id == messages[idx - 1].id

            # assert roles, first message is user, second is assistant
            if idx % 2 == 0:
                assert message.role == "user"
            else:
                assert message.role == "assistant"
        assert messages[1].content == chat.message.content, "Assistant's first response is empty"
        assert messages[2].content == "Shorten the answer", "Second user message content doesn't match"
        assert messages[3].content != "", "Assistant's second response is empty"
        assert messages[4].content == "Lengthen the answer", "Third user message content doesn't match"
        assert messages[5].content != "", "Assistant's third response is empty"

        assert all(message.chat_id == chat_model.id for message in messages), (
            "Not all messages have the correct chat_id"
        )

    @pytest.mark.asyncio
    async def test_calculate_message_completion_cost(
        self, async_client, user_id, chat, async_http_requester, db_session, caplog
    ):
        """
        Creates a new chat and sends two messages to the chat.
        Checks that message completion cost is calculated correctly
        and completion cost is applied to assistant (AI) messages only
        """

        # second text
        create_chat_message_url = api.get_chat_item(user_uuid=user_id, chat_uuid=chat.uuid)
        await async_http_requester(
            "test_create_chat_messages_linked_by_parent_id",
            async_client.put,
            create_chat_message_url,
            json={"query": "Shorten the answer", "use_rag": False},
        )

        # third text
        create_chat_message_url = api.get_chat_item(user_uuid=user_id, chat_uuid=chat.uuid)
        await async_http_requester(
            "test_create_chat_messages_linked_by_parent_id",
            async_client.put,
            create_chat_message_url,
            json={"query": "Lengthen the answer", "use_rag": False},
        )

        execute = await db_session.execute(select(Chat).filter(Chat.uuid == chat.uuid))
        chat_model = execute.scalar_one()

        # should have 6 messages in total, one pair for each request and response
        execute = await db_session.execute(
            select(Message).filter(Message.chat_id == chat_model.id).order_by(Message.created_at)
        )
        messages = list(execute.scalars())

        assert len(messages) == 6, f"Expected 6 messages, but got {len(messages)}"
        input_token = 0
        for idx, message in enumerate(messages):
            llm_id = message.llm_id
            execute = await db_session.execute(select(LLM).filter(LLM.id == llm_id))
            llm = execute.scalar_one()
            # todo: change llm table  input_cost_per_token and input_cost_per_token to decimal from double.
            # as it causes rounding issues.
            llm_input_cost_per_token = round(Decimal(llm.input_cost_per_token), 10)
            llm_output_cost_per_token = round(Decimal(llm.output_cost_per_token), 10)

            if idx % 2 == 0:
                # number of input tokens are saved in the user message, not stored in the assistant message.
                # therefore need to capture input token from previous user message.
                input_token = message.tokens
                assert message.completion_cost is None, "Cost calculation does not apply to user messages"

            # check cost calculation assistant message
            if idx % 2 == 1:
                # number of output tokens are saved in the assistant message
                # check completion cost for each assistant message
                output_token = message.tokens
                message_cost = round(message.completion_cost, 10)
                logger.info(
                    "input_cost_per_token %s, output_cost_per_token %s, input_token %s, output_token %s"
                    % (
                        llm_input_cost_per_token,
                        llm_output_cost_per_token,
                        input_token,
                        output_token,
                    )
                )

                completion_cost = (input_token * llm_input_cost_per_token) + (output_token * llm_output_cost_per_token)
                assert message_cost > 0
                assert message_cost == completion_cost
