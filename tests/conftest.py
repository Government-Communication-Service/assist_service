# ruff: noqa: E501

import asyncio
import logging
import os
from collections.abc import Generator
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from typing import Dict, Optional, TypeVar
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import anyio
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.api.endpoints import ENDPOINTS
from app.auth.config import AUTH_TOKEN
from app.auth.constants import AUTH_TOKEN_ALIAS, SESSION_AUTH_ALIAS, USER_KEY_UUID_ALIAS
from app.auth.create_auth_session_service import create_auth_session
from app.auth.verify_service import verify_and_get_user_from_header
from app.central_guidance.service_index import sync_central_index
from app.chat.schemas import ChatWithLatestMessage
from app.database.db_session import get_db_session
from app.database.models import AuthSession, User
from app.database.table import async_db_session
from app.main import app as app_under_test
from app.opensearch.schemas import OpenSearchRecord
from app.opensearch.service import AsyncOpenSearchClient, create_client

logger = logging.getLogger(__name__)

# Add this line to ignore the specific test file
collect_ignore = [
    "gov_uk_search/test_gov_uk_search_evals.py",
    "central_guidance/test_evals_central_guidance.py",
]

T = TypeVar("T")

YieldFixture = Generator[T, None, None]
api = ENDPOINTS()


@pytest.fixture()
def test_db():
    """Creates an event loop that is shared by test scenarios and the application"""
    database = os.getenv("TEST_POSTGRES_DB")
    logger.debug(f"conftest.py using POSTGRES_DB = {database}")
    return database


@pytest.fixture(scope="session")
def event_loop(request):
    """Creates an event loop that is shared by test scenarios and the application"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_session_provider():
    @asynccontextmanager
    async def _session():
        async with async_db_session() as s:
            yield s

    return _session


@pytest.fixture
async def db_session(db_session_provider):
    async with db_session_provider() as s:
        yield s


@pytest.fixture
async def sync_central_rag_index(db_session_provider):
    async with db_session_provider() as s:
        await sync_central_index(s)


@pytest.fixture
def test_app():
    with ExitStack():
        yield app_under_test


@pytest.fixture()
async def session_override(test_app, db_session):
    async def get_db_session_override():
        yield db_session

    test_app.dependency_overrides[get_db_session] = get_db_session_override


@pytest.fixture(scope="module")
def mock_llm_invoke():
    with patch("app.chat.create_new_chat") as mock_invoke:
        mock_invoke.return_value = "Hi! how can I help you today?"
        yield mock_invoke


@pytest.fixture(autouse=True)
def set_env_vars():
    os.environ["DISABLE_BUGSNAG_LOGGING"] = "True"


@pytest.fixture(name="user_id")
def user_id():
    return str(uuid4())


@pytest.fixture(name="user_id_int", autouse=True)
def user_id_int():
    return 1


@pytest.fixture
async def chat(user_id, async_client, async_http_requester) -> ChatWithLatestMessage:
    """
    Creates a random chat for the user specified by the user_id fixture.
    Returns a ChatWithLatestMessage instance.
    """
    logger.debug(f"Creating chat for user ID: {user_id}")
    url = api.chats(user_uuid=user_id)
    response = await async_http_requester(
        "chat_endpoint",
        async_client.post,
        url,
        json={"query": "Say Hi"},
    )

    chat = ChatWithLatestMessage(**response)
    return chat


@pytest.fixture()
async def chat_item(async_client, user_id, async_http_requester):
    logger.debug(f"Creating chat for user ID: {user_id}")
    url = api.chats(user_uuid=user_id)
    response = await async_http_requester("chat_endpoint", async_client.post, url, json={"query": "hello"})

    return response


@pytest.fixture()
def user_prompt_payload():
    return {
        "title": "testing, testing, 1, 2, 3...",
        "content": "Test prompt body. You can be a tree, which tree would you like to be?",
    }


@pytest.fixture()
async def user_prompt_item(async_client, user_id, async_http_requester, user_prompt_payload):
    # create a new user prompt
    logger.debug(f"Creating user prompt for user int ID: {user_id}")

    post_url = api.create_user_prompt(user_uuid=user_id)
    logger.debug(f"POST URL for user ID: {user_id}, url: {post_url}")

    post_response = await async_http_requester(
        "POST to user_prompts_endpoint", async_client.post, post_url, json=user_prompt_payload
    )
    return post_response


@pytest.fixture(autouse=True)
def user_id_malformed():
    return str(uuid4())[:-3]


@pytest.fixture(autouse=True)
def session():
    return str(uuid4())


@pytest.fixture(autouse=True)
def session_malformed():
    return str(uuid4())[:-3]


@pytest.fixture(scope="session")
def theme():
    return 1


@pytest.fixture(scope="session")
def use_case():
    return 1


@pytest.fixture(name="auth_token", autouse=True)
def auth_token(user_id):
    return AUTH_TOKEN


@pytest.fixture
async def user(user_id) -> User:
    async with async_db_session() as session:
        user_obj = await verify_and_get_user_from_header(user_key_uuid=user_id, db_session=session)
        await session.commit()
    return user_obj


@pytest.fixture
async def auth_session(user) -> AuthSession:
    async with async_db_session() as session:
        auth_session_obj = await create_auth_session(session, user)
        await session.commit()
    return auth_session_obj


@pytest.fixture(name="another_user_auth_session")
async def another_user_auth_session(async_client, auth_token):
    async def _session(user_id):
        response = await async_client.post(
            api.get_sessions(),
            headers={USER_KEY_UUID_ALIAS: user_id, AUTH_TOKEN_ALIAS: auth_token},
        )
        assert response.status_code == 200, f"The status code {response.status_code} was incorrect; it should be 200."

        body = response.json()

        assert body, "The response was empty."
        assert body != "", "The response was empty."

        return body[SESSION_AUTH_ALIAS]

    return _session


@pytest.fixture(name="client", autouse=True)
def client(test_app):
    client = TestClient(test_app)
    yield client


@pytest.fixture()
async def async_client(test_app):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://testserver") as cl:
        yield cl


@pytest.fixture(name="default_headers", autouse=True)
def default_headers(user_id, auth_token, auth_session):
    return {
        USER_KEY_UUID_ALIAS: user_id,
        SESSION_AUTH_ALIAS: str(auth_session.uuid),
        AUTH_TOKEN_ALIAS: auth_token,
    }


@pytest.fixture(name="auth_token_only_headers", autouse=True)
def auth_token_only_headers(auth_token):
    return {AUTH_TOKEN_ALIAS: auth_token}


def _log_http_request(kwargs, test_name, url):
    logger.debug(f"Starting request {test_name}")
    logger.debug(f"ENDPOINT URL: {url}")
    logger.debug(f"HEADERS: {default_headers}")
    logger.debug(f"DATA: {kwargs}")


@pytest.fixture()
async def file_uploader(user_id, async_http_requester, async_client):
    api = ENDPOINTS()
    api_endpoint = api.user_documents(user_id)

    async def upload(file_path: str, scenario_name: str, expected_response_code: int = 200):
        file_name = Path(file_path).name
        async with await anyio.open_file(file_path, "rb") as f:
            response = await async_http_requester(
                scenario_name,
                async_client.post,
                api_endpoint,
                files={"file": (file_name, await f.read())},
                response_code=expected_response_code,
            )
            return response

    return upload


@pytest.fixture()
async def mock_db_session(test_app):
    mock_db_session = AsyncMock(name="db_session")

    async def get_db_session_override():
        yield mock_db_session

    test_app.dependency_overrides[get_db_session] = get_db_session_override
    return mock_db_session


@pytest.fixture
async def async_http_requester(default_headers):
    async def run(
        test_name,
        client_invoke,
        url,
        response_code=200,
        response_type: str = "json",
        response_content_type: str = None,
        headers: Dict = None,
        **kwargs,
    ) -> Dict:
        _log_http_request(kwargs, test_name, url)
        combined_headers = default_headers | headers if headers else default_headers
        response = await client_invoke(url, headers=combined_headers, **kwargs)
        return _assert_http_response(response, response_code, response_content_type, response_type)

    return run


def _assert_http_response(response, response_code, response_content_type, response_type):
    logger.debug(f"RESPONSE: {response.status_code} - {response.content}.")
    logger.debug(f"RESPONSE HEADERS: {response.headers}")

    assert response.status_code == response_code, (
        f"The status code {response.status_code} was incorrect; it should be {response_code}."
    )
    if response_content_type is not None:
        actual_content_type = response.headers["content-type"]
        assert response_content_type == actual_content_type, (
            f"The content type {response_content_type} was incorrect; it should be {actual_content_type}."
        )
    if response_type == "json":
        return response.json()
    return response.content


@pytest.fixture
def http_requester(default_headers):
    def run(
        test_name,
        client_invoke,
        url,
        response_code=200,
        response_type: str = "json",
        response_content_type: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        _log_http_request(kwargs, test_name, url)

        response = client_invoke(url, headers=default_headers, **kwargs)
        return _assert_http_response(response, response_code, response_content_type, response_type)

    return run


@pytest.fixture(scope="function")
def mock_message_table(mocker):
    mock_message_table_class = mocker.patch("app.database.table.MessageTable")
    mock_message_table_instance = mock_message_table_class.return_value

    mock_message = Mock()
    mock_message.id = 456
    mock_message.uuid = "some-uuid"
    mock_message.content = "some-content"
    mock_message_table_instance.create.return_value = mock_message
    mock_message_table_instance.update.return_value = mock_message
    mock_message_table_instance.get_by_chat.return_value = []

    return mock_message_table_instance


@pytest.fixture(scope="function")
def mock_bedrock_handler(mocker):
    mock_bedrock_handler_class = mocker.patch("app.bedrock.bedrock.BedrockHandler")
    mock_bedrock_handler_instance = mock_bedrock_handler_class.return_value

    mock_bedrock_handler_instance.format_messages.return_value = "Formatted Messages"
    mock_bedrock_handler_instance.format_response.return_value = "Formatted Response"
    mock_bedrock_handler_instance.invoke_async.return_value = AsyncMock(return_value="LLM Response")

    return mock_bedrock_handler_instance


@pytest.fixture(scope="module")
def opensearch_record():
    return OpenSearchRecord(
        document_name="Test document",
        document_url="www.testurl.com",
        chunk_name="Introduction",
        chunk_content="This is some test content.",
    )


@pytest.fixture(scope="module")
def opensearch_client():
    return create_client()


@pytest.fixture
async def async_opensearch_client():
    client = AsyncOpenSearchClient.get()
    yield client
    await AsyncOpenSearchClient.close()


@pytest.fixture(scope="session")
def prompts_for_upload():
    prompts = [
        {
            "theme_title": "Planning campaigns and communication",
            "theme_subtitle": "Create OASIS plans, design narratives, conduct COM-B" + "analysis",
            "theme_position": 1,
            "use_case_position": 1,
            "use_case_title": "I want to create an outline for an OASIS PLAN",
            "use_case_instruction": "Build a detailed OASIS communications plan using"
            + "the Government Communications Service OASIS framework."
            + " Use all of the information above to build the OASIS plan, filling in"
            + "gaps yourself."
            + " Give several options where there isn't one obvious approach to"
            + "recommend."
            + " For objectives, focus on behaviour change as primary objectives,"
            + "including secondary sub-objectives where needed."
            + " For audience, provide some key insights about the relevant audiences"
            + "and carry out a COM-B analysis of behaviour for each relevant audience."
            + " For strategy, develop options for an overarching communications"
            + "narrative that will underpin the campaign."
            + " For implementation, explore the practicalities of delivering the"
            + "campaign, including timing and channels."
            + " For scoring, use the GCS Evaluation Framework to suggest appropriate"
            + "evaluation methods and KPIs."
            + " Introduce each section with a summary paragraph."
            + " At the end, review the plan and list any weaknesses, risks, gaps, or"
            + "assumptions.",
            "use_case_user_input_form": "Background information: [USER PROMPT]\n"
            + ""
            + "Objectives: [USER PROMPT]\n"
            + ""
            + "Audience: [USER PROMPT]\n"
            + "Strategy: [USER PROMPT]\n"
            + "Implementation: [USER PROMPT]\n"
            + "Scoring: [USER PROMPT]",
        },
        {
            "theme_title": "Planning campaigns and communication",
            "theme_subtitle": "Create OASIS plans, design narratives, conduct COM-B" + "analysis",
            "theme_position": 1,
            "use_case_position": 2,
            "use_case_title": "I want feedback on my OASIS plan ",
            "use_case_instruction": "Provide detailed feedback on this plan."
            + " Highlight any gaps, weaknesses, assumptions, or risks in the plan.",
            "use_case_user_input_form": "Plan: [USER PROMPT]",
        },
        {
            "theme_title": "Planning campaigns and communication",
            "theme_subtitle": "Create OASIS plans, design narratives, conduct COM-B" + "analysis",
            "theme_position": 1,
            "use_case_position": 3,
            "use_case_title": "I want to create some audience personas",
            "use_case_instruction": "Develop a set of personas for the following"
            + "audience insight."
            + " Identify the topic from the insight."
            + " Each persona should be representative of a different segment of the"
            + "audience."
            + " They should include demographic and situational information about the"
            + "person including factors relevant to the topic, their attitude towards the"
            + "relevant topic, and a short vignette recounting a specific recent experience"
            + "related to the topic."
            + " Review your answer and list any weaknesses, gaps, or assumptions in"
            + "your personas.",
            "use_case_user_input_form": "Audience insight: [USER PROMPT]",
        },
        {
            "theme_title": "Planning campaigns and communication",
            "theme_subtitle": "Create OASIS plans, design narratives, conduct COM-B" + "analysis",
            "theme_position": 1,
            "use_case_position": 4,
            "use_case_title": "I want to conduct COM-B analysis",
            "use_case_instruction": "Produce a COM-B analysis for this behaviour, using"
            + "any background information provided."
            + " Write a very short introduction to COM-B, and give a list of 5 possible"
            + "barriers for each of the three elements."
            + " Include barriers that might only be relevant to a subset of the"
            + "audience, but specify where this is the case."
            + " At the end, make a list of any gaps, weaknesses, or assumptions in the"
            + "analysis.",
            "use_case_user_input_form": "Behaviour (WHO does WHAT?): [USER PROMPT]*"
            + "Other information: [USER PROMPT]",
        },
        {
            "theme_title": "Planning campaigns and communication",
            "theme_subtitle": "Create OASIS plans, design narratives, conduct COM-B" + "analysis",
            "theme_position": 1,
            "use_case_position": 5,
            "use_case_title": "I want to design a narrative for a campaign",
            "use_case_instruction": "Develop 3 contrasting options for an overarching"
            + "campaign narrative, taking into account all of the information above."
            + " Ensure at least 1 of the ideas is highly creative and unexpected."
            + " Write a short paragraph outlining each overarching narrative including"
            + "the tone, approach, and strategy."
            + " For each option, label them, and provide 3 cohesive messaging ideas for"
            + "each overarching narrative."
            + " For each option, set out any important considerations for"
            + "communicators, and a list of benefits, risks, and unintended consequences for"
            + "each option.",
            "use_case_user_input_form": "Campaign background information: [USER"
            + "PROMPT]\n"
            + "Target audience: [USER PROMPT]",
        },
        {
            "theme_title": "Campaign evaluation",
            "theme_subtitle": "Build theories of change, define outcomes, produce" + "KPIs",
            "theme_position": 2,
            "use_case_position": 1,
            "use_case_title": "I want suggestions for some Key Performance Indicators" + "(KPIs)",
            "use_case_instruction": "Identify a wide range of options for key"
            + "performance indicators (KPIs) for the following campaign."
            + " Use all the information provided below."
            + " KPIs should represent a combination of desired outcomes (Changes in"
            + "behaviour, changes in attitude and contribution to policy objectives) and"
            + "outtakes (Reception, Perception and reaction of stakeholders."
            + " Campaign efficiency metrics for communicators)."
            + " Categorise them into outtakes and outcomes, with further subcategories"
            + "as needed."
            + " Provide a suggestion for how each KPI could be measured (in brackets),"
            + "and provide the unit."
            + " At the end, review your suggestions and identify any weaknesses or"
            + "gaps, or likely challenges with measuring these KPIs.",
            "use_case_user_input_form": "Background information: [USER PROMPT]\n"
            + "Campaign objective: [USER PROMPT]\n"
            + "Target audience: [USER PROMPT]\n"
            + "Budget and timeline: [USER PROMPT]\n"
            + "Channels to be used: [USER PROMPT]",
        },
        {
            "theme_title": "Campaign evaluation",
            "theme_subtitle": "Build theories of change, define outcomes, produce" + "KPIs",
            "theme_position": 2,
            "use_case_position": 2,
            "use_case_title": "I want to define outtakes and outcomes for a project",
            "use_case_instruction": "Identify a wide range of options for key"
            + "performance indicators (KPIs) for the following campaign."
            + " Use all the information provided below."
            + " KPIs should represent a combination of desired outcomes (Changes in"
            + "behaviour, changes in attitude and contribution to policy objectives) and"
            + "outtakes (Reception, Perception and reaction of stakeholders."
            + " Campaign efficiency metrics for communicators)."
            + " Categorise them into outtakes and outcomes, with further subcategories"
            + "as needed."
            + " Provide a suggestion for how each KPI could be measured (in brackets),"
            + "and provide the unit."
            + " At the end, review your suggestions and identify any weaknesses or"
            + "gaps, or likely challenges with measuring these KPIs.",
            "use_case_user_input_form": "Background information: [USER PROMPT]\n"
            + "Campaign objective: [USER PROMPT]\n"
            + "Target audience: [USER PROMPT]\n"
            + "Budget and timeline: [USER PROMPT]\n"
            + "Channels to be used: [USER PROMPT]",
        },
        {
            "theme_title": "Campaign evaluation",
            "theme_subtitle": "Build theories of change, define outcomes, produce" + "KPIs",
            "theme_position": 2,
            "use_case_position": 3,
            "use_case_title": "I want to build a theory of change for a project",
            "use_case_instruction": "You are aiming to build a theory of change for"
            + "this campaign."
            + " If a list of activities, outputs, outtakes, outcomes, and impact is"
            + "provided above then check through and improve this list."
            + " If it says \u201c[USER PROMPT]\u201d above then you must identify the"
            + "information yourself based on background information about the campaign."
            + " Connect together the different activities, outputs, outtakes, and"
            + "outcomes in a coherent logic model."
            + " Explain the assumptions that connect each step to the next."
            + " At the end, review your suggestions and identify any weaknesses or"
            + "gaps, or likely challenges with measuring these KPIs.",
            "use_case_user_input_form": "Background information: [USER PROMPT]\n"
            + "Desired impact: [USER PROMPT]\n"
            + "Desired outcomes: [USER PROMPT]\n"
            + "Desired outtakes: [USER PROMPT]\n"
            + "Expected outputs (campaign components, channels): [USER PROMPT]\n"
            + "Expected activities (budget, assets, initiatives): [USER PROMPT] \n"
            + "Target audience: [USER PROMPT]\n"
            + "Budget and timeline: [USER PROMPT]\n"
            + "Channels to be used: [USER PROMPT]",
        },
        {
            "theme_title": "Developing marketing and communication content",
            "theme_subtitle": "Design messaging, conduct message testing, summarise" + "text",
            "theme_position": 3,
            "use_case_position": 1,
            "use_case_title": "I want to conduct message testing with my target" + "audience ",
            "use_case_instruction": "The goal is to understand how target audiences"
            + "might receive a message so that we can prevent unintended consequences and"
            + "tailor the message appropriately."
            + " Analyse the messages, narrative, or communications assets above."
            + " Identify who the target audience(s) are and what their characteristics"
            + "are."
            + " Analyse how the target audiences might perceive the communication,"
            + "using the following steps: Analyse potential negative or unintended consequences"
            + "Explain how you could optimise or tailor the message most effectively for each"
            + "target audience."
            + " Identify subsets of target audiences who might respond or perceive the"
            + "message differently from others, and explain how the message might be adapted or"
            + "targeted.",
            "use_case_user_input_form": "Communication message, narrative, or asset:"
            + "[USER PROMPT]\n"
            + "Target audience: [USER PROMPT]",
        },
        {
            "theme_title": "Developing marketing and communication content",
            "theme_subtitle": "Design messaging, conduct message testing, summarise" + "text",
            "theme_position": 3,
            "use_case_position": 2,
            "use_case_title": "I want to design some effective marketing messages",
            "use_case_instruction": "The goal is to come up with a wide range of ideas"
            + "for potential marketing messages based on the information above about a"
            + "campaign.\n"
            + "1) Develop 6 contrasting \u201ctop line\u201d messages, including ideas"
            + "that are conventional, unconventional, action-oriented, authoritative,"
            + "aspirational, and creative. \n"
            + "2) For each message, explain how it could be adapted for press, social"
            + "media, TV, radio, billboards.\n"
            + "3) For each message, list strengths and weaknesses. \n"
            + "4) After, check through the messages to make sure that there are 6"
            + "options, and that each option is distinct. ",
            "use_case_user_input_form": "Campaign background information: [USER"
            + "PROMPT]\n"
            + "Campaign narrative or plan: [USER PROMPT]\n"
            + "Target audience:  [USER PROMPT]",
        },
        {
            "theme_title": "Developing marketing and communication content",
            "theme_subtitle": "Design messaging, conduct message testing, summarise" + "text",
            "theme_position": 3,
            "use_case_position": 3,
            "use_case_title": "I want to create some content for social media",
            "use_case_instruction": "The goal is to draft the text for 5 posts that can"
            + "be used."
            + " 1) Identify who the target audience(s) are and what their"
            + "characteristics are."
            + " 2) Draft 5 social media posts that use the campaign narrative and"
            + "campaign background information to appeal to the target audience."
            + " 3) For each post, explain how it could be adapted for different social"
            + "media channels."
            + " 4) For each post, list strengths and weaknesses."
            + " 5) After, check through the messages to make sure that there are 5"
            + "options, and that each option is distinct.",
            "use_case_user_input_form": "Campaign background information: [USER PROMPT]"
            + "Campaign narrative: [USER PROMPT] Target audience: [USER PROMPT] Social media"
            + "channel: [USER PROMPT]",
        },
        {
            "theme_title": "Developing marketing and communication content",
            "theme_subtitle": "Design messaging, conduct message testing, summarise" + "text",
            "theme_position": 3,
            "use_case_position": 4,
            "use_case_title": "I want to adapt my messaging for different" + "publications",
            "use_case_instruction": "The goal is to adapt the campaign messaging to"
            + "appeal to the target audience based on the target audiences' persona.\n"
            + "1) Identify who the target audience(s) are and what their"
            + "characteristics are.\n"
            + "2) Analyse the campaign messaging to see whether it appeals to the"
            + "target audience based on their persona.\n"
            + "3) Adapt the messaging where it has been identified that it does not"
            + "appeal to the target audience.\n"
            + "4) Explain why changes made to the messaging have made it more appealing"
            + "to the target audience. \n"
            + "5) List the messages strengths, weaknesses and limitations. \n"
            + ""
            + "",
            "use_case_user_input_form": "Campaign messaging: [USER PROMPT] Target" + "audience persona: [USER PROMPT]",
        },
        {
            "theme_title": "Developing marketing and communication content",
            "theme_subtitle": "Design messaging, conduct message testing, summarise" + "text",
            "theme_position": 3,
            "use_case_position": 5,
            "use_case_title": "I want a draft of a speech or talking points ",
            "use_case_instruction": "Generate a speech, matching the tone based, taking"
            + "into account specific information.\n"
            + "Guidance for use:\n"
            + "Ensure the press release is clear, factual and to the point.\n"
            + "Structure it clearly and systematically.\n"
            + "Avoid jargon and ensure the language is accessible and key terms are"
            + "defined.",
            "use_case_user_input_form": "Subject: [USER PROMPT Specify topic of the"
            + "speech]\n"
            + ""
            + "\n"
            + "Background information: [USER PROMPT provide as much information about"
            + "the topic of the speech, the key points that the speech needs to cover and the"
            + "context in which the speech is being made]\n"
            + ""
            + "\n"
            + "Tone of voice: [USER PROMPT: specify the tone of voice]\n"
            + ""
            + "\n"
            + "Key objectives: [Describe the broader context and relevance of the topic"
            + "in current governmental policies or initiatives]",
        },
        {
            "theme_title": "Developing marketing and communication content",
            "theme_subtitle": "Design messaging, conduct message testing, summarise" + "text",
            "theme_position": 3,
            "use_case_position": 6,
            "use_case_title": "I want to simplify, paraphrase or summarise text ",
            "use_case_instruction": "The goal is to summarise and simplify the text to"
            + "correspond with the number of words."
            + " 1) Identify how many words long the summarisation of the text should"
            + "be."
            + " 2) Analyse the text to identify key points."
            + " 3) Draft the summarisation to focus on the key points."
            + " 4) Check that the summarisation does not exceed the maximum number of"
            + "words. ",
            "use_case_user_input_form": "Text: [USER PROMPT] Number of words: [USER" + "PROMPT]",
        },
        {
            "theme_title": "Developing marketing and communication content",
            "theme_subtitle": "Design messaging, conduct message testing, summarise" + "text",
            "theme_position": 3,
            "use_case_position": 7,
            "use_case_title": "I want to draft a toolkit for a campaign",
            "use_case_instruction": "The goal is to create a document with instructions"
            + "for stakeholders on how to use campaign assets and messaging as part of OASIS"
            + "campaign."
            + " 1) Analyse the OASIS plan and campaign narrative."
            + " Identify the campaign objectives, target audiences and key campaign"
            + "milestones from the OASIS plan."
            + " 2) Create instructions on how to use messaging and campaign assets to"
            + "reach the target audience."
            + " These instructions should include recommended channels, messaging and"
            + "dates\\/times to put out communications content."
            + " 3) Check that the instructions take into account the stakeholders"
            + "characteristics.",
            "use_case_user_input_form": "OASIS plan: [USER PROMPT] Campaign narrative:"
            + "[USER PROMPT] Messaging: [USER PROMPT] Campaign assets: [USER PROMPT]"
            + "Stakeholder persona:",
        },
        {
            "theme_title": "Developing marketing and communication content",
            "theme_subtitle": "Design messaging, conduct message testing, summarise" + "text",
            "theme_position": 3,
            "use_case_position": 8,
            "use_case_title": "I want to adapt messaging for my different audiences",
            "use_case_instruction": "Adapt the provided government message to suit"
            + "different audience segments."
            + " Ensure that the core information remains consistent, but tailor the"
            + "language, tone, and examples to resonate with each distinct group."
            + " Consider cultural, demographic, and psychographic factors that might"
            + "influence how each audience receives and interprets the message.\n"
            + ""
            + "\n"
            + "Adaptation Guidelines:\n"
            + "Language and Tone: Adjust the language and tone to suit each audience."
            + " For example, use more formal language for older audiences and"
            + "conversational tone for younger ones.\n"
            + "Examples and Analogies: Tailor examples and analogies to be relevant and"
            + "relatable to each audience segment.\n"
            + ""
            + "\n"
            + "Evaluation:\n"
            + "After adapting the message for each segment, evaluate the effectiveness"
            + "of your adaptations."
            + " Consider feedback from representatives of each group, if available, and"
            + "refine the message accordingly.\n"
            + ""
            + "\n"
            + "Format:\n"
            + "Present the adapted messages in a clear format, indicating the audience"
            + "segment each version is intended for.",
            "use_case_user_input_form": "Government Message:\n"
            + ""
            + "[USER PROMPT: Insert the primary message or information to be"
            + "communicated]\n"
            + ""
            + "\n"
            + "Audience details:\n"
            + ""
            + "[USER PROMPT: Describe the audience, including relevant demographic,"
            + "cultural, or psychographic characteristics]",
        },
        {
            "theme_title": "Developing marketing and communication content",
            "theme_subtitle": "Design messaging, conduct message testing, summarise" + "text",
            "theme_position": 3,
            "use_case_position": 9,
            "use_case_title": "I want to adapt messaging for some different channels",
            "use_case_instruction": "Develop a strategy to adapt the provided"
            + "government message for different communication channels."
            + " The goal is to ensure the message is effectively conveyed across each"
            + "platform, taking into account the unique characteristics and user expectations"
            + "of each channel.\n"
            + ""
            + "\n"
            + ""
            + "\n"
            + "Adaptation Guidelines:\n"
            + "Social Media: Focus on concise, engaging content with ideas for relevant"
            + "visuals."
            + " Adapt the message to be shareable and interactive."
            + " Consider using hashtags to increase visibility.\n"
            + "GOV.UK Website: Provide detailed and comprehensive information."
            + " Use plain English and ensure accessibility for a diverse audience.\n"
            + "Press Release: Maintain a formal and informative tone."
            + " Include quotes from relevant officials and detailed background"
            + "information.\n"
            + ""
            + "\n"
            + "Channel-Specific Considerations:\n"
            + "Consider the limitations and strengths of each channel (e.g., character"
            + "limits on Twitter, visual capabilities on social media, format restrictions in"
            + "emails)."
            + "\n"
            + "Tailor the call-to-action for each channel based on user behavior and"
            + "channel capabilities.\n"
            + ""
            + "\n"
            + "Format:\n"
            + "Present the adapted messages in a clear format, indicating the specific"
            + "channel each version is intended for.",
            "use_case_user_input_form": "Government Message:\n"
            + ""
            + "[USER PROMPT: Insert the primary message or information to be"
            + "communicated]\n"
            + ""
            + "\n"
            + "Communication Channels:\n"
            + "Channel 1: Social Media (e.g., Twitter, Facebook)\n"
            + "Channel 2: Official GOV.UK Website\n"
            + "Channel 3: Press Release\n"
            + ""
            + "[Add or delete channels as needed]",
        },
        {
            "theme_title": "Media handling and press releases",
            "theme_subtitle": "Brainstorm media questions, write press releases, draft" + "briefings",
            "theme_position": 4,
            "use_case_position": 1,
            "use_case_title": "I want to write a press release",
            "use_case_instruction": "Generate a formal, impartial and plain English"
            + "press release on the provided topic, taking into account specific information."
            + "\n"
            + "Guidance for use:\n"
            + "Ensure the press release is clear, factual and to the point.\n"
            + "Structure it clearly and systematically.\n"
            + "Avoid jargon and ensure the language is accessible and key terms are"
            + "defined.",
            "use_case_user_input_form": "Subject: [USER PROMPT Specify the topic of the"
            + "press release]\n"
            + ""
            + "\n"
            + "Background information: [USER PROMPT provide as much information as"
            + "possible about the topic of the briefing]\n"
            + ""
            + "\n"
            + "Key objectives: [Describe the broader context and relevance of the topic"
            + "in current governmental policies or initiatives]",
        },
        {
            "theme_title": "Media handling and press releases",
            "theme_subtitle": "Brainstorm media questions, write press releases, draft" + "briefings",
            "theme_position": 4,
            "use_case_position": 2,
            "use_case_title": "I want to brainstorm possible media questions or" + "concerns",
            "use_case_instruction": "The goal is to anticipate possible questions from"
            + "journalists or the general public based on the information provided above."
            + " Identify 5 sincere questions and 5 hostile questions that might be"
            + "asked.",
            "use_case_user_input_form": "Press release, speech text, announcement, or" + "campaign plan: [USER PROMPT]",
        },
        {
            "theme_title": "Media handling and press releases",
            "theme_subtitle": "Brainstorm media questions, write press releases, draft" + "briefings",
            "theme_position": 4,
            "use_case_position": 3,
            "use_case_title": "I want to prepare some responses for media questions ",
            "use_case_instruction": "The goal is to generate 3 different responses to"
            + "the anticipated questions based on the organisational position and target"
            + "audience."
            + " 1) Identify the target audience and their characteristics."
            + " 2) Analyse the question and the organisational position to see how the"
            + "organisation should respond honestly."
            + " 3) Draft 3 answers to each anticipated question which is consistent"
            + "with the organisations position and appeals to the target audience."
            + " 4) For each answer list its strengths, weaknesses and limitations.",
            "use_case_user_input_form": "Anticipated questions: [USER PROMPT]"
            + "Organisational position: [USER PROMPT] Target audiences: [USER PROMPT]",
        },
        {
            "theme_title": "Media handling and press releases",
            "theme_subtitle": "Brainstorm media questions, write press releases, draft" + "briefings",
            "theme_position": 4,
            "use_case_position": 4,
            "use_case_title": "I want to draft a briefing for a minister",
            "use_case_instruction": "Generate a formal, impartial, and plain English"
            + "ministerial briefing on the provided topic, taking into account specific"
            + "information provided.\n"
            + "Guidance for Use:\n"
            + "Ensure the briefing is clear, factual, and to the point.\n"
            + "Structure it clearly and systematically.\n"
            + "Avoid jargon and ensure the language is accessible and key terms are"
            + "defined.\n"
            + "Adapt the content to the minister\u2019s portfolio.",
            "use_case_user_input_form": "Subject:\n"
            + ""
            + "[USER PROMPT: Specify the topic of the briefing].\n"
            + ""
            + "\n"
            + "Audience:\n"
            + ""
            + "[USER PROMPT: Enter information about the minister].\n"
            + ""
            + "\n"
            + "Background information:\n"
            + ""
            + "[USER PROMPT: Provide as much information as possible about the topic"
            + "and context of the briefing].\n"
            + ""
            + "\n"
            + "Key Objectives:\n"
            + ""
            + "[USER PROMPT: Describe the broader context and relevance of the topic in"
            + "current governmental policies or initiatives].",
        },
        {
            "theme_title": "Exploring strategy risks",
            "theme_subtitle": "Analyse risks, explore consequences, test adversarial" + "strategies",
            "theme_position": 5,
            "use_case_position": 1,
            "use_case_title": "I want to analyse possible risks and consequences",
            "use_case_instruction": "The goal is to carry out a risk and unintended"
            + "consequence analysis for the campaign detailed above."
            + " Identify several relevant audience(s) who might be affected, and for"
            + "each audience, write a list of 3 risks and potential unintended consequences for"
            + "the campaign, ordered from most to least likely.",
            "use_case_user_input_form": "Campaign plan: [USER PROMPT]",
        },
        {
            "theme_title": "Exploring strategy risks",
            "theme_subtitle": "Analyse risks, explore consequences, test adversarial" + "strategies",
            "theme_position": 5,
            "use_case_position": 2,
            "use_case_title": "I want to test strategies from an adversarial" + "perspective ",
            "use_case_instruction": "You are carrying out a red teaming exercise, to"
            + "identify how a campaign could go wrong. \n"
            + "Assume that the campaign above was a failure, resulting in wasted money"
            + "and resources and a negative public response.\n"
            + "Produce 6 contrasting hypothetical explanations for the campaign"
            + "failure."
            + " Explain the causal chain of events that led to failure in each"
            + "scenario, and identify the assumptions present.",
            "use_case_user_input_form": "Campaign plan: [USER PROMPT]",
        },
        {
            "theme_title": "Crisis Communications",
            "theme_subtitle": "Develop response strategies, draft crisis messages",
            "theme_position": 6,
            "use_case_position": 1,
            "use_case_title": "I want to create a STOP crisis communications plan",
            "use_case_instruction": "A STOP plan covers the key aspects of a crisis"
            + "comms plan - that is strategy (objectives and audience), tactics (key actions in"
            + "first hour, day and week), organisation structure (team) and people requirements"
            + "(how many people and training). \n"
            + "On the cover page: provide detail on the name of the crisis, lead"
            + "government department, summary of reasonable worst-case scenario, details of any"
            + "previous events in which a crisis has occurred and/or a plan has been"
            + "implemented. \n"
            + "In the strategy section: provide detail on policy objective, comms"
            + "objective (short mission statement setting out the big picture of what"
            + "successful communication looks like), key audiences (including those who are"
            + "hard to reach or vulnerable, identification of who might be especially impacted"
            + "by the crisis and what their needs are), message principles (the core principles"
            + "guiding the response such as empathy, transparency and accuracy of information"
            + "to impacted groups), risks and barriers to communications and specific guidance"
            + "on how the public can meet their essential needs and protect themselves from"
            + "harm."
            + " Their essential needs are for: shelter, hydration, food, warmth, light,"
            + "hygiene, health, safety from hazards, news about the crisis, help from others,"
            + "earn a living, contact loved ones, volunteer/help others, entertainment, get"
            + "home from work and/or pick up children from school/childcare. \n"
            + "There are four steps involved in providing this guidance: STEP 1:"
            + "Identify the barriers that the crisis will cause for the public meeting each of"
            + "the essential needs (e.g."
            + " a flood could cause a power outage that means people can't use their"
            + "phones to check in on elderly/vulnerable relatives) STEP 2: Identify what"
            + "actions the public might take to meet their needs, despite the crisis (e.g."
            + " someone might leave their home to visit a nearby family member or"
            + "friend to use their phone to contact their relative) STEP 3: Identify what"
            + "positive or negative consequences could arise as a result of the actions people"
            + "may take to meet their needs (e.g."
            + " a person leaving their home during a flood could get swept away) STEP"
            + "4: Develop communications and messages to help remove the barriers and protect"
            + "people from experiencing the harms identified (e.g."
            + " tell people not to leave their home as the flood waters could put them"
            + "at risk of injury or death and issue messaging that lets people know that anyone"
            + "on the Priority Services Register will be visited by a member of their local"
            + "resilience team to check on them - and therefore that family members should not"
            + "leave their homes to do this themselves)."
            + " Please provide output for each step for each essential need, ensuring"
            + "that messages are specific (e.g."
            + " they don't just tell people what not to do, they say what they should"
            + "do instead). \n"
            + ""
            + "                    In tactics section: provide a quick reference"
            + "handbook of key comms actions pre-crisis, over the first hour, day and week, key"
            + "other departments to contact, draft holding line and messages in the Krebs"
            + "format (outlining what is known, what is not known, what the public should do,"
            + "what government is doing, and when more information will be made available)."
            + " All information should aim to be risk specific."
            + " Include other government departments that may be linked to the risk."
            + " \n"
            + ""
            + "                    In organisation section: who will be the Comms"
            + "Strategic level leader (GOLD), key comms team members & disciplines, and policy"
            + "/ operations / political colleagues / stakeholders Other contacts and teams,"
            + "physical location and logistics (e.g."
            + " logistics, IT access for surge team, would in person access to"
            + "buildings be required for those who don\u2019t normally access), business"
            + "continuity plan in case of a cyber attack, how to escalate and cascade"
            + "information outside of business hours In people section: outline an estimate how"
            + "many people will be needed, and their location / clearance requirements."
            + " Success criteria: Performance and accuracy of information, up to date"
            + "information, consideration of hard-to-reach audiences, audiences essential"
            + "needs, anticipation of crisis behaviour from a behavioural science perspective,"
            + "following the Krebs method.\n"
            + ""
            + "            ",
            "use_case_user_input_form": "Please create a STOP crisis comms plan for"
            + "[USER INPUT RISK NAME]."
            + " Assume that the lead government department is [USER INPUT LEAD"
            + "GOVERNMENT DEPARTMENT].",
        },
        {
            "theme_title": "Crisis Communications",
            "theme_subtitle": "Develop response strategies, draft crisis messages",
            "theme_position": 6,
            "use_case_position": 2,
            "use_case_title": "I want to draft messages for a crisis",
            "use_case_instruction": 'Design a crisis communications message about the" \
    + "information provided above, directed at the public." \
                + " The information should be laid out clearly and directly, using" \
    + "plain English, without a salutation or sign off." \
                + " It should be suitable for members of the public to read directly," \
    + "and also for media organisations, industry, and NGOs." \
                + " Use "The Krebs method" of crisis communication, giving each part a" \
    + "suitable label." \
                + " If any key information is missing, leave placeholders." \
                + " The Krebs method of crisis communications sets out that" \
    + "communicators need to get across 5 key messages during a crisis:\n" \
                            + f"1. What is known about the crisis\n" \
                            + f"2. What is currently unknown\n" \
                            + f"3. What the Government is doing about the crisis\n" \
                            + f"4. What the public should do\n" \
                            + f"5.  When and where the public can find out more" \
    + "information.\n" \
                            + f"At the end, review your message and add any extra" \
    + "considerations or weaknesses.',
            "use_case_user_input_form": "What is currently known about the crisis:"
            + "[USER PROMPT]\n"
            + "What actions the Government are taking: [USER PROMPT]\n"
            + "What you are instructing the public to do: [USER PROMPT]\n"
            + "When and where the public can access further information: [USER"
            + "PROMPT]",
        },
        {
            "theme_title": "Crisis Communications",
            "theme_subtitle": "Develop response strategies, draft crisis messages",
            "theme_position": 6,
            "use_case_position": 3,
            "use_case_title": "I want to develop a response strategy in a crisis",
            "use_case_instruction": " The goal is to create a crisis communications"
            + "plan with the following sections: 1) Title."
            + " 2) Background and aim."
            + " 3) Objectives."
            + " 4) Target audience."
            + " 5) Key messages."
            + " 6) Channel."
            + " 7) Spokesperson."
            + " 8) Managing co-ordination."
            + " 9) Timescales."
            + " 10) Practical applications."
            + " 11) Resources and budget."
            + " 12) Research."
            + " 13) Review, evaluation and exercising."
            + " 14) Gaps and risks."
            + " 1) Analyse the background and aims, summarise this into a series of"
            + "bullet points."
            + " 2) Suggest between 3-5 objectives based on the background and aims."
            + " The objectives should be specific, measurable, achievable, realistic"
            + "and time-bound 3) Identify the target audience and summarise their capabilities,"
            + "opportunities, motivations and behaviours."
            + " 4) Using this understanding of the audience select relevant"
            + "communications channels to reach that audience and identify spokespersons"
            + "trusted by that audience."
            + " 5) Identify any research that may be relevant to the target audience."
            + " 6) Conduct a GAP analysis and risk assessment.",
            "use_case_user_input_form": "Title: [USER INPUT] Background and aim: [USER"
            + "INPUT] Target audience: [USER INPUT] Spokesperson: [USER INPUT] Key messages:"
            + "[USER INPUT]",
        },
        {
            "theme_title": "Stakeholder identification and management",
            "theme_subtitle": "Identify stakeholders, understand key requirements",
            "theme_position": 7,
            "use_case_position": 1,
            "use_case_title": "I want to identify and understand key stakeholders",
            "use_case_instruction": "I want to identify a wide range of possible"
            + "stakeholders for this project, including functions of UK Government departments,"
            + "public interest groups, subsets of the general public, organisations, and"
            + "businesses."
            + " Make a list of at least 10 possible stakeholders, grouped in"
            + "categories, including some unconventional ideas.",
            "use_case_user_input_form": "Background information: [USER PROMPT]",
        },
        {
            "theme_title": "Internal communications",
            "theme_subtitle": "Brainstorm ideas, develop strategies, produce" + "materials",
            "theme_position": 8,
            "use_case_position": 1,
            "use_case_title": "I want to develop a strategy for internal" + "communication",
            "use_case_instruction": "Build a detailed OASIS internal communications"
            + "plan using the Government Communications Service OASIS framework."
            + " Use all of the information above to build the OASIS plan, filling in"
            + "gaps yourself."
            + " Give several options where there isn't one obvious approach to"
            + "recommend."
            + " For objectives, focus on behaviour change as primary objectives,"
            + "including secondary sub-objectives where needed."
            + " For audience, provide some key insights about the relevant audiences"
            + "and carry out a COM-B analysis of behaviour for each relevant audience."
            + " For strategy, develop options for an overarching communications"
            + "narrative that will underpin the campaign."
            + " For implementation, explore the practicalities of delivering the"
            + "campaign, including timing and channels."
            + " For scoring, use the GCS Evaluation Framework to suggest appropriate"
            + "evaluation methods and KPIs."
            + " Introduce each section with a summary paragraph."
            + " At the end, review the plan and list any weaknesses, risks, gaps, or"
            + "assumptions.",
            "use_case_user_input_form": "Background information: [USER PROMPT]\n"
            + "Objectives: [USER PROMPT]\n"
            + "Audience: [USER PROMPT]\n"
            + "Strategy: [USER PROMPT]\n"
            + "Implementation: [USER PROMPT]\n"
            + "Scoring: [USER PROMPT]",
        },
        {
            "theme_title": "Internal communications",
            "theme_subtitle": "Brainstorm ideas, develop strategies, produce" + "materials",
            "theme_position": 8,
            "use_case_position": 2,
            "use_case_title": "I want to produce emails, posters or other internal" + "materials ",
            "use_case_instruction": 'Build an internal communications product using the" \
    + "information above." \
                + " It should be engaging, using a friendly yet professional tone, and" \
    + "suitable for a UK audience." \
                + " Make sure that any "calls to action" are clear, and that the most" \
    + "important information is prominent.',
            "use_case_user_input_form": "Objective: [USER PROMPT]\n"
            + "Information to include: [USER PROMPT]\n"
            + "Tone and style: [USER PROMPT]\n"
            + "Type of product: [USER PROMPT]",
        },
        {
            "theme_title": "Internal communications",
            "theme_subtitle": "Brainstorm ideas, develop strategies, produce" + "materials",
            "theme_position": 8,
            "use_case_position": 3,
            "use_case_title": "I want to create some engaging headlines\\/event" + "titles\\/articles",
            "use_case_instruction": "Aim: Brainstorm Headlines\\/Titles.\n"
            + "Using the information provided, create 8-10 potential headlines or"
            + "titles.\n"
            + "Ensure they are concise, engaging, and relevant to the topic and"
            + "audience.\n"
            + "Consider using action verbs, thought-provoking questions, or relevant"
            + "puns\\/phrases.\n"
            + ""
            + "\n"
            + "Final Recommendation:\n"
            + "Suggest the 2-3 most compelling headline\\/titles that effectively"
            + "represents the content and resonates with the target audience.\n"
            + ""
            + "\n"
            + "Guidance for Use:\n"
            + "Strive for clarity and impact while remaining true to the content's"
            + "essence.\n"
            + "Keep the language accessible and avoid technical jargon unless it's"
            + "audience-appropriate.\n"
            + "Test the effectiveness of the headline\\/title by considering its appeal"
            + "and clarity. ",
            "use_case_user_input_form": "Content Type & Topic:\n"
            + ""
            + "[USER PROMPT: Identify if it's for an event or an article, and provide a"
            + "brief topic summary].\n"
            + ""
            + "\n"
            + "Target Audience:\n"
            + ""
            + "[USER PROMPT: Who is the intended audience? (e.g."
            + ", public, professionals, policymakers)].\n"
            + ""
            + "\n"
            + "Core Message:\n"
            + ""
            + "[USER PROMPT: What is the key message or main focus?].",
        },
        {
            "theme_title": "Internal communications",
            "theme_subtitle": "Brainstorm ideas, develop strategies, produce" + "materials",
            "theme_position": 8,
            "use_case_position": 4,
            "use_case_title": "I want to brainstorm some interview questions for Q&A",
            "use_case_instruction": "Question Development:\n"
            + "Based on the material overview and speaker profile, create a list of"
            + "potential questions.\n"
            + "Focus on questions that:\n"
            + "Dive deeper into the material's subject matter.\n"
            + "Explore the speaker's unique perspective or expertise.\n"
            + "Encourage discussion on practical implications, future outlooks, or"
            + "challenges.\n"
            + "Question Types:\n"
            + ""
            + "\n"
            + "Ensure a mix of question app_types:\n"
            + "Open-ended questions for detailed responses.\n"
            + "Specific questions for clarification on particular points.\n"
            + "Hypothetical or scenario-based questions for perspective.\n"
            + ""
            + "\n"
            + "Question Review:\n"
            + "Evaluate the questions for relevance, clarity, and their ability to"
            + "engage both the speaker and the audience.\n"
            + ""
            + "\n"
            + "Guidance for Use:\n"
            + "Aim for questions that stimulate discussion, rather than simple yes\\/no"
            + "answers.\n"
            + "Consider the audience's interest and knowledge level when framing"
            + "questions.\n"
            + "Balance between topical relevance and exploring new insights or"
            + "angles.\n"
            + "Be mindful of the speaker's expertise; align questions to their area of"
            + "knowledge.\n"
            + "Ensure questions are respectful and in line with government"
            + "communication standards. ",
            "use_case_user_input_form": "Material Overview:\n"
            + ""
            + "[USER PROMPT: Summarize the key points or main themes of the material"
            + "provided].\n"
            + ""
            + "\n"
            + "Speaker Profile:\n"
            + ""
            + "[USER PROMPT: Briefly describe the background of the speaker (e.g."
            + ", expertise, role, previous contributions)].",
        },
        {
            "theme_title": "Skills and training",
            "theme_subtitle": "Develop training content, design workshops, optimise" + "text",
            "theme_position": 9,
            "use_case_position": 1,
            "use_case_title": "I want to develop training content for my staff",
            "use_case_instruction": "Develop a training plan using information above."
            + " Provide a list of topics and principles that should be taught."
            + " Include detailed learning objectives."
            + " The training plan should be logically structured, and should build"
            + "progressively from simple to more complex information that brings different"
            + "elements together."
            + " Give ideas for exercises, case studies, and discussions to enhance"
            + "learning."
            + " At the end, list any weaknesses, gaps, or limitations in the training"
            + "plan.",
            "use_case_user_input_form": "Learning objectives: [USER PROMPT]\n"
            + "Information about the audience for the training: [USER PROMPT]\n"
            + "Format of the training: [USER PROMPT]",
        },
        {
            "theme_title": "Skills and training",
            "theme_subtitle": "Develop training content, design workshops, optimise" + "text",
            "theme_position": 9,
            "use_case_position": 2,
            "use_case_title": "I want to create a quiz for assessment",
            "use_case_instruction": "Generate a quiz about a topic based on specific"
            + "information provided."
            + " Each question should have four answer options of which only one is"
            + "true."
            + " Highlight which are the correct and incorrect answers\n"
            + ""
            + "\n"
            + "User instruction:\n"
            + "Check that the correct answer options are accurate.\n"
            + ""
            + "\n"
            + "Review and update the content of the quiz accordingly",
            "use_case_user_input_form": "Subject: [USER PROMPT - provide the subject of"
            + "the quiz]\n"
            + "Topic: [USER PROMPT - provide as much information as possible about the"
            + "topic including content that the participants of the quiz need to have"
            + "learnt.\n"
            + ""
            + "\n"
            + ""
            + "",
        },
        {
            "theme_title": "Skills and training",
            "theme_subtitle": "Develop training content, design workshops, optimise" + "text",
            "theme_position": 9,
            "use_case_position": 3,
            "use_case_title": "I want to design workshop activities",
            "use_case_instruction": "Suggest contrasting ideas for workshop exercises"
            + "to aid in reaching the objective above."
            + " The exercises should all be different, and may aim to ideate,"
            + "prioritise, analyse, or spark conversation, depending on the aims of the"
            + "workshop."
            + " List the benefits and weaknesses of each idea, and provide a few"
            + "variations of each idea.",
            "use_case_user_input_form": "Objective of the workshop: [USER PROMPT]\n"
            + "Information about workshop participants: [USER PROMPT]\n"
            + "Workshop format: [ONLINE\\/IN PERSON]",
        },
        {
            "theme_title": "Skills and training",
            "theme_subtitle": "Develop training content, design workshops, optimise" + "text",
            "theme_position": 9,
            "use_case_position": 4,
            "use_case_title": "I want to simplify, paraphrase or summarise text ",
            "use_case_instruction": " The goal is to summarise and simplify the text to"
            + "correspond with the number of words."
            + " 1) Identify how many words long the summarisation of the text should"
            + "be."
            + " 2) Analyse the text to identify key points."
            + " 3) Draft the summarisation to focus on the key points."
            + " 4) Check that the summarisation does not exceed the maximum number of"
            + "words.",
            "use_case_user_input_form": "Text: [USER PROMPT] Number of words: [USER" + "PROMPT]",
        },
        {
            "theme_title": "Skills and training",
            "theme_subtitle": "Develop training content, design workshops, optimise" + "text",
            "theme_position": 9,
            "use_case_position": 5,
            "use_case_title": "I want to present ideas in a way that is optimised for" + "learning",
            "use_case_instruction": "Create Learning Materials from Provided"
            + "Information\n"
            + ""
            + "\n"
            + "Objective:\n"
            + "Condense and present information in a format that enhances learning and"
            + "understanding for a specific audience.\n"
            + ""
            + "\n"
            + "Development of Learning Content:\n"
            + "Create content tailored to the chosen format, focusing on key points.\n"
            + "Use visuals and examples to clarify complex ideas.\n"
            + "Incorporate interactive or reflective elements if applicable.\n"
            + ""
            + "\n"
            + "Guidance for Use:\n"
            + "Strive for clear, accessible language and a logical flow of"
            + "information.\n"
            + "Prioritize the most important or impactful information to maintain"
            + "engagement.\n"
            + "Ensure all content is accurate, up-to-date, and relevant to the"
            + "audience. ",
            "use_case_user_input_form": "Content Overview:\n"
            + ""
            + "[USER PROMPT: Provide a brief summary of the content to be used for"
            + "educational purposes].\n"
            + ""
            + "\n"
            + "Target Audience and Learning Objectives:\n"
            + ""
            + "[USER PROMPT: Identify the audience and state the primary learning"
            + "objective].",
        },
        {
            "theme_title": "Inclusive and accessible communications",
            "theme_subtitle": "Considerations, test material for accessibility",
            "theme_position": 10,
            "use_case_position": 1,
            "use_case_title": "I want to account for cultural considerations in my" + "communication",
            "use_case_instruction": "Generate 5 ways in which a communications campaign"
            + "could be adapted to account for cultural considerations."
            + " This could be in terms of the messaging (i.e."
            + " tone of voice, messenger and themes) and implementation (i.e."
            + " communications channels, type of content, format of content).",
            "use_case_user_input_form": "Subject: [USER PROMPT - provide the subject of"
            + "the communications campaign]\n"
            + "Target audience: [USER PROMPT provide as much detail about the target"
            + "audience and factual information about cultural preferences]",
        },
        {
            "theme_title": "Inclusive and accessible communications",
            "theme_subtitle": "Considerations, test material for accessibility",
            "theme_position": 10,
            "use_case_position": 2,
            "use_case_title": "I want to test communications material for" + "accessibility",
            "use_case_instruction": "Review Materials for Accessibility\n"
            + ""
            + "\n"
            + "Objective:\n"
            + "Evaluate provided materials (such as articles, speeches, announcements)"
            + "for accessibility, ensuring they are suitable for a diverse audience, including"
            + "individuals with disabilities.\n"
            + ""
            + "\n"
            + "Accessibility Checklist:\n"
            + ""
            + "\n"
            + "Language Clarity: Check for clear, jargon-free language appropriate to a"
            + "diverse UK audience.\n"
            + "Readability: Assess sentence length and complexity, and review for"
            + "potential misunderstandings.\n"
            + "Format: Ensure the material is available in multiple formats (e.g."
            + ", text, audio).\n"
            + "Structure: Evaluate the organization for a logical flow and ease of"
            + "understanding.\n"
            + ""
            + "\n"
            + "Recommendations:\n"
            + "Provide suggestions for improving accessibility in areas where the"
            + "material falls short.\n"
            + ""
            + "\n"
            + "Guidance for Use:\n"
            + "Focus on inclusivity to ensure the material is comprehensible and usable"
            + "by people with various needs.\n"
            + "Be mindful of cultural sensitivity and avoid potential biases.\n"
            + "Regularly update guidelines and practices to align with the latest"
            + "accessibility standards.",
            "use_case_user_input_form": "Material Overview:\n"
            + ""
            + "[USER PROMPT: Identify the type of material to be reviewed (e.g."
            + ", article, speech, announcement)].\n"
            + ""
            + "\n"
            + "Specific Areas of Focus:\n"
            + ""
            + "[USER PROMPT: Highlight any particular areas that require careful review"
            + "based on the content or audience].",
        },
        {
            "theme_title": "Inclusive and accessible communications",
            "theme_subtitle": "Considerations, test material for accessibility",
            "theme_position": 10,
            "use_case_position": 3,
            "use_case_title": "I want to apply alt-text in my digital communications",
            "use_case_instruction": "Please outline how I would create alt-text for an"
            + "image being used on social media, in line with the Government Communication"
            + "Service digital accessibility standards?",
            "use_case_user_input_form": "",
        },
        {
            "theme_title": "Inclusive and accessible communications",
            "theme_subtitle": "Considerations, test material for accessibility",
            "theme_position": 10,
            "use_case_position": 4,
            "use_case_title": "I want to improve my provision for British Sign Language translations",
            "use_case_instruction": "Please outline what the expectation is for the use of British Sign Language in UK "
            + "Government communications, in line with the Government Communication Service accessibility standards.",
            "use_case_user_input_form": "",
        },
        {
            "theme_title": "Inclusive and accessible communications",
            "theme_subtitle": "Considerations, test material for accessibility",
            "theme_position": 10,
            "use_case_position": 5,
            "use_case_title": "I want to consider the colour contrast of my digital communications content",
            "use_case_instruction": "Please outline why colour contrast is important in government communications in "
            + "line with the Government Communication Service accessibility standards. Could you also recommend a free "
            + "online tool which can help me check the contrast of my digital communications content.",
            "use_case_user_input_form": "",
        },
        {
            "theme_title": "Inclusive and accessible communications",
            "theme_subtitle": "Considerations, test material for accessibility",
            "theme_position": 10,
            "use_case_position": 6,
            "use_case_title": "I want to use emojis and hashtags on social media in an accessible way",
            "use_case_instruction": "Please outline the best practices when using emojis and hashtags on social media "
            + "communications."
            + " Please give me recommendations on how to best apply these features taking into consideration "
            + "accessibility needs of users and the government communication service accessibility standards."
            + " Please ensure the response includes relation to the platforms: Twitter/X, LinkedIn, Instagram, "
            + "Facebook.",
            "use_case_user_input_form": "",
        },
        {
            "theme_title": "Brainstorming and ideation",
            "theme_subtitle": "Consider problems, brainstorm, understand communications impact",
            "theme_position": 11,
            "use_case_position": 1,
            "use_case_title": "I want to explore a problem from different points of view",
            "use_case_instruction": "I want to think about this problem from several different perspectives."
            + " Give perspectives from 8 different disciplines."
            + " Include the following disciplines: Economist, Service Designer, Behavioural Scientist, Social "
            + "Researcher, Innovator\\/Entrepreneur."
            + " Choose the other disciplines yourself, depending on the problem."
            + " Include their view on the problem, questions to explore, and potential solutions.",
            "use_case_user_input_form": 'Problem: "[USER PROMPT - description of problem]"',
        },
        {
            "theme_title": "Brainstorming and ideation",
            "theme_subtitle": "Consider problems, brainstorm, understand communications impact",
            "theme_position": 11,
            "use_case_position": 2,
            "use_case_title": "I want to understand how communications can help solve a problem",
            "use_case_instruction": "I am exploring a problem, and considering the role of Government communications "
            + "and campaigns in addressing this problem."
            + " Set out a list of ways that communications and campaigns could help solve this problem, and set out a "
            + "list of limitations of what communications and campaigns could achieve."
            + " Finally, set out any additional considerations for Government communicators when attempted to tackle "
            + "this problem using communication.",
            "use_case_user_input_form": 'Problem: "[USER PROMPT - description of problem]"',
        },
        {
            "theme_title": "Brainstorming and ideation",
            "theme_subtitle": "Consider problems, brainstorm, understand communications impact",
            "theme_position": 11,
            "use_case_position": 3,
            "use_case_title": "I want to brainstorm ideas for communication content",
            "use_case_instruction": "Brainstorm Creative Ideas for Communications Content\n"
            + ""
            + "\n"
            + "Objective:\n"
            + "Generate diverse and engaging content ideas suitable for communication purposes, based on provided "
            + "background information and understanding of the target audience.\n"
            + ""
            + "\n"
            + "Idea Generation:\n"
            + "Develop a list of potential content ideas that align with the background information and appeal to the "
            + "target audience.\n"
            + "Consider various formats like articles, videos, infographics, podcasts, or social media posts.\n"
            + "Creativity and Engagement:\n"
            + "Focus on ideas that are informative, engaging, and likely to capture the attention of the specified "
            + "audience.\n"
            + "Explore innovative ways to present the information.\n"
            + ""
            + "\n"
            + "Idea Evaluation:\n"
            + "Assess the feasibility, potential impact, and alignment with communication goals for each idea, "
            + "considering the target audience's preferences and needs.\n"
            + ""
            + "\n"
            + "Guidance for Use:\n"
            + "Balance creativity with the preferences and interests of the target audience.\n"
            + "Stay informed about current trends and successful communication strategies within the audience's "
            + "domain.\n"
            + "Aim to create content that not only informs but also engages and resonates with the audience.",
            "use_case_user_input_form": "Background Information:\n"
            + ""
            + "[USER PROMPT: Summarize the key information or themes that the content should address].\n"
            + ""
            + "\n"
            + "Target Audience:\n"
            + ""
            + "[USER PROMPT: Define the target audience for the content].",
        },
        {
            "theme_title": "Brainstorming and ideation",
            "theme_subtitle": "Consider problems, brainstorm, understand communications impact",
            "theme_position": 11,
            "use_case_position": 4,
            "use_case_title": "I want to develop a presentation to engage an audience",
            "use_case_instruction": "You have to give a presentation on a topic."
            + " The goal is to develop an agenda for the presentation."
            + " The presentation should emphasise several key takeaways."
            + " For each item on the agenda there should be one engaging activation.",
            "use_case_user_input_form": "Topic: [USER INPUT]\n"
            + ""
            + "\n"
            + "Key takeaways: [USER INPUT]\n"
            + ""
            + "\n"
            + "Further information: [USER INPUT]",
        },
        {
            "theme_title": "Research",
            "theme_subtitle": "Write surveys, create guides, design activities",
            "theme_position": 12,
            "use_case_position": 1,
            "use_case_title": "I want to design workshop activities",
            "use_case_instruction": "Suggest 5 contrasting ideas for workshop exercises to aid in reaching the "
            + "objective above."
            + " The exercises should all be different, and may aim to ideate, prioritise, analyse, or spark "
            + "conversation, depending on the aims of the workshop."
            + " List the benefits and weaknesses of each idea, and provide a few variations of each idea.",
            "use_case_user_input_form": "Objective of the workshop: [USER PROMPT]\n"
            + "Information about workshop participants: [USER PROMPT]\n"
            + "Workshop format: [ONLINE\\/IN PERSON]",
        },
        {
            "theme_title": "Research",
            "theme_subtitle": "Write surveys, create guides, design activities",
            "theme_position": 12,
            "use_case_position": 2,
            "use_case_title": "I want to conduct COM-B analysis",
            "use_case_instruction": "Produce a COM-B analysis for this behaviour, using any background information "
            + "provided."
            + " Write a very short introduction to COM-B, and give a list of 5 possible barriers for each of the three"
            + "elements."
            + " Include barriers that might only be relevant to a subset of the audience,"
            + " but specify where this is the case."
            + " At the end, make a list of any gaps, weaknesses, or assumptions in the analysis.",
            "use_case_user_input_form": "Behaviour (WHO does WHAT?): [USER PROMPT]* Other information: [USER PROMPT]",
        },
        {
            "theme_title": "Research",
            "theme_subtitle": "Write surveys, create guides, design activities",
            "theme_position": 12,
            "use_case_position": 3,
            "use_case_title": "I want to create a discussion guide for a focus group",
            "use_case_instruction": "Produce a focus group discussion guide for this topic and audience. "
            + "Explore a range of relevant research questions. "
            + "The guide should be divided into categories with clear headings. "
            + "Each question should have 2-3 optional follow up topics. "
            + "Include a template for a facilitator introduction, leaving gaps for them to explain the purpose "
            + "and context of the research."
            + " At the end, review the guide "
            + "and make a list of any missing topics or considerations for the facilitator.",
            "use_case_user_input_form": "Topic and background: [USER PROMPT] \n"
            + "Audience: [USER PROMPT]\n"
            + "Number of questions: [USER PROMPT]",
        },
        {
            "theme_title": "Research",
            "theme_subtitle": "Write surveys, create guides, design activities",
            "theme_position": 12,
            "use_case_position": 4,
            "use_case_title": "I want to write a survey",
            "use_case_instruction": "Produce a survey for this topic and audience. \n"
            + ""
            + "\n"
            + "Explore a range of relevant research questions. "
            + "The survey should be divided into categories with clear headings."
            + " Similar questions should be grouped together. "
            + "At the end, review the survey and make a list of any missing topics "
            + "or potential biases in the questions. "
            + "Do not include questions about demographics.",
            "use_case_user_input_form": "Topic and background: [USER PROMPT]\n"
            + "Audience: [USER PROMPT]\n"
            + "Maximum number of questions: [USER PROMPT]\n"
            + "Type of answers: [Multiple choice only\\/Open questions only\\/Mix "
            + "of multiple choice and open questions]",
        },
        {
            "theme_title": "Research",
            "theme_subtitle": "Write surveys, create guides, design activities",
            "theme_position": 12,
            "use_case_position": 5,
            "use_case_title": "I want to create a survey for staff feedback",
            "use_case_instruction": " Objective:\n"
            + "Create an effective survey to gather staff feedback on a particular topic, "
            + "ensuring the questions are relevant, engaging, and provide valuable insights.\n"
            + ""
            + "\n"
            + ""
            + "\n"
            + "Question Formulation:\n"
            + "Based on the topic and survey purpose, draft a series of questions.\n"
            + "Include a mix of question app_types:\n"
            + "Rating scale questions for quantifiable feedback.\n"
            + "Open-ended questions for qualitative insights.\n"
            + "Multiple-choice questions for specific preferences or choices.\n"
            + "Key Focus Areas:\n"
            + ""
            + "\n"
            + "Ensure questions cover different aspects of the topic, such as:\n"
            + "General perceptions and attitudes.\n"
            + "Specific experiences or examples.\n"
            + "Suggestions for improvements or changes.\n"
            + "Survey Length and Structure:\n"
            + ""
            + "\n"
            + "Keep the survey concise; ideally no more than 10-15 questions.\n"
            + "Organize questions logically, starting from broad questions and moving to more specific ones.\n"
            + ""
            + "\n"
            + "Guidance for Use:\n"
            + ""
            + "\n"
            + "Ensure questions are unbiased and phrased clearly.\n"
            + "Maintain confidentiality and anonymity to encourage honest responses.\n"
            + "Avoid leading questions to get unbiased feedback.\n"
            + "Consider including an option for open feedback or comments at the end.\n"
            + "Be respectful of respondents' time; ensure the survey is not overly long. ",
            "use_case_user_input_form": "Survey Topic:\n"
            + ""
            + "[USER PROMPT: Define the specific topic or area for which feedback is being sought].\n"
            + ""
            + "\n"
            + "Purpose of the Survey:\n"
            + ""
            + "[USER PROMPT: Clarify the objectives of the survey and how the feedback will be used].",
        },
        {
            "theme_title": "Recruitment",
            "theme_subtitle": "Test, adapt job descriptions for audiences",
            "theme_position": 13,
            "use_case_position": 1,
            "use_case_title": "I want to test a job description against an audience persona",
            "use_case_instruction": "Please review the following job description"
            + " and provide feedback on whether it effectively appeals to the target audience."
            + " Specifically assess whether the language, requirements and overall tone"
            + " of the job description would be attractive and engaging to this target"
            + " audience. Please present your response clearly using subheadings and bullet"
            + " points. Thank you!",
            "use_case_user_input_form": "Target audience: Age: [USER INPUT]"
            + " Gender: [USER INPUT] "
            + "Ethnicity: [USER INPUT]"
            + " Location: [USER INPUT] "
            + "Educational background: [USER INPUT]"
            + " Level of professional experience: [USER INPUT] "
            + "Further information: [USER INPUT]",
        },
        {
            "theme_title": "Recruitment",
            "theme_subtitle": "Test, adapt job descriptions for audiences",
            "theme_position": 13,
            "use_case_position": 2,
            "use_case_title": "I want to adapt a job description for different audiences",
            "use_case_instruction": "Please update the job description to better resonate "
            + "with the Target Audience. Focus on using language that resonates with "
            + "the audience and showcasing that might appeal to that target audience. "
            + "Also can you please identify any changes you made and why.",
            "use_case_user_input_form": "Target audience: \n"
            + "Age: [USER INPUT]\n"
            + "Gender: [USER INPUT]\n"
            + "Ethnicity: [USER INPUT]\n"
            + "Location: [USER INPUT]\n"
            + "Educational background: [USER INPUT]\n"
            + "Level of professional experience: [USER INPUT] \n"
            + "Further information:\n"
            + "[USER INPUT]\n",
        },
        {
            "theme_title": "Recruitment",
            "theme_subtitle": "Test, adapt job descriptions for audiences",
            "theme_position": 13,
            "use_case_position": 3,
            "use_case_title": "I want to identify stakeholders to help with recruitment",
            "use_case_instruction": "I want to identify a wide range of possible stakeholders"
            + " that could be used to help recruit people from the target audience."
            + " This could includefunctions of UK Government departments,"
            + " public interest groups, subsets of the general public, "
            + "organisations, and businesses. Make a list of and group "
            + "in categories, including some unconventional ideas.",
            "use_case_user_input_form": "Target audience:"
            + " \n"
            + "Age: [USER INPUT]"
            + "\n"
            + "Gender: [USER INPUT]"
            + "\n"
            + "Ethnicity: [USER INPUT]"
            + "\n"
            + "Location: [USER INPUT]"
            + "\n"
            + "Educational background: [USER INPUT]"
            + "\n"
            + "Level of professional experience: [USER INPUT]"
            + " \n"
            + "Further information:\n"
            + "[USER INPUT]\n",
        },
    ]
    return prompts


@pytest.fixture(scope="session")
def user_prompt_response():
    response_dict = {
        "id": 1,
        "uuid": "ac33c76f-146f-4e57-b448-498679ad7559",
        "user_id": 1,
        "title": "testing, testing, 1, 2, 3...",
        "content": "Test prompt body. You can be a tree, which tree would you like to be?",
        "created_at": "2024-11-13T19:27:50.798573",
        "updated_at": "2024-11-13T19:27:50.798573",
        "deleted_at": None,
    }
    return f"{response_dict}"
