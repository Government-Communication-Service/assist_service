from unittest.mock import Mock

import pytest

from app.bedrock.bedrock import ContentToolUse


@pytest.fixture(autouse=True)
def mock_llm_table(mocker):
    mock_llm = Mock()
    mock_llm.max_tokens = 4096
    mocker.patch("app.smart_targets.service.LLMTable")
    mocker.patch("app.smart_targets.service.LLMTable.return_value.get_by_model", return_value=mock_llm)
    return mock_llm


@pytest.fixture
def mock_get_available_metrics(mocker):
    mock_response = {"no-metric-selected": {"id": "0", "name": "no-metric-selected", "uuid": None}}
    return mocker.patch(
        "app.smart_targets.service.SmartTargetsService.get_available_metrics", return_value=mock_response
    )


@pytest.fixture
def mock_llm_smart_targets_choice(mocker):
    mock_response = Mock()
    mock_response.content = [
        ContentToolUse(
            input={
                "selected_metrics": [
                    {
                        "metric_name": "no-metric-selected",
                        "context_for_filters": "Test context",
                    }
                ]
            },
            type="tool_use",
        )
    ]
    return mocker.patch("app.bedrock.bedrock.BedrockHandler.invoke_async", return_value=mock_response)


@pytest.fixture
def mock_messages():
    mock_message = Mock()
    mock_message.content = "Test content"
    mock_message.content_enhanced_with_rag = "Test content enhanced with rag"
    mock_message.summary = "Test summary"
    return [mock_message]
