# Style Guide Tests

Comprehensive test suite for the GOV.UK Style Guide checking feature.

## Test Structure

### Unit Tests

#### `test_unit_service.py`
Tests for service-level functions:
- **Conversation Context Wrapping**: Tests `_wrap_conversation_context()` with various message configurations
- **Router Agent**: Tests `should_run_full_style_guide_analysis()` decision logic
  - Initial messages always trigger full analysis
  - Follow-ups with modification requests route to simple responses
  - Follow-ups with new check requests route to full analysis
  - Error handling and fallback behavior
- **Simple Response Handler**: Tests `_handle_simple_response()` conversational responses

#### `test_unit_checker.py`
Tests for core checker functions:
- **Sentence Extraction**: Tests helper functions for extracting sentences from documents
- **Rule Loading**: Tests loading and filtering of rule definitions
- **Case-Insensitive Checking**: Tests deterministic rule detection (case-insensitive)
- **Case-Sensitive Checking**: Tests deterministic rule detection (case-sensitive)
- **Prompt Creation**: Tests prompt generation for LLM calls

### Integration Tests

#### `test_integration_service.py`
Tests for the complete service flow:
- **Initial Call Flow**: Full analysis with deterministic + LLM checks
- **Follow-up Routing**: Router directing to simple response vs full analysis
- **No Violations**: Proper handling of clean content
- **Error Handling**: Graceful degradation on errors
- **Conversation Context**: Context propagation through the flow
- **End-to-End Detection**: Real deterministic violation detection

### End-to-End Tests

#### `test_e2e_chat.py`
Tests for complete chat integration:
- **Initial Check (Streaming)**: First message with style guide violations
- **Follow-up Modification (Streaming)**: Conversational edits after initial check
- **Clean Content**: Content with no violations
- **Combined Features**: Style guide + RAG working together
- **Error Handling**: Edge cases and graceful failures
- **Non-Streaming**: Traditional request/response flow

## Running Tests

### Run all style guide tests:
```bash
pytest tests/style_guide/ -v
```

### Run by category:
```bash
# Unit tests only
pytest tests/style_guide/ -v -m unit

# Integration tests only
pytest tests/style_guide/ -v -m integration

# E2E tests only
pytest tests/style_guide/ -v -m e2e
```

### Run specific test file:
```bash
pytest tests/style_guide/test_unit_service.py -v
pytest tests/style_guide/test_integration_service.py -v
pytest tests/style_guide/test_e2e_chat.py -v
```

### Run with coverage:
```bash
pytest tests/style_guide/ --cov=app.style_guide --cov-report=html
```

## Test Coverage

The test suite covers:

1. **Router Logic** (90%+)
   - Decision making for follow-up messages
   - LLM prompt construction
   - Error fallback behavior

2. **Conversation Context** (100%)
   - Message wrapping for LLM context
   - Multi-turn conversation handling

3. **Simple Response Handler** (90%+)
   - Conversational modification responses
   - Context-aware prompting

4. **Deterministic Checking** (85%+)
   - Case-sensitive rule matching
   - Case-insensitive rule matching
   - Sentence extraction

5. **Full Flow Integration** (80%+)
   - Initial analysis path
   - Follow-up routing
   - Error handling

6. **Chat Integration** (75%+)
   - Streaming responses
   - Non-streaming responses
   - Multi-feature interaction

## Key Test Patterns

### Mocking LLM Responses
```python
from anthropic.types import Message as AnthropicMessage, TextBlock, Usage

def create_mock_llm_response(text: str) -> AnthropicMessage:
    return AnthropicMessage(
        id="msg_test",
        content=[TextBlock(text=text, type="text")],
        model="test-model",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage=Usage(input_tokens=100, output_tokens=50),
    )
```

### Mocking Message Objects
```python
from unittest.mock import Mock
from app.database.models import Message

def create_mock_message(role: str, content: str, message_id: int = 1) -> Message:
    message = Mock(spec=Message)
    message.id = message_id
    message.role = role
    message.content = content
    message.content_enhanced_with_rag = content
    return message
```

### Testing Router Decisions
```python
@patch("app.style_guide.service.LLMTable")
@patch("app.style_guide.service.BedrockHandler")
async def test_router_decides_simple_response(mock_bedrock_handler, mock_llm_table):
    mock_handler_instance.invoke_async = AsyncMock(
        return_value=create_mock_llm_response("SIMPLE_RESPONSE")
    )
    # ... test logic
```

## Fixtures Used

From `conftest.py`:
- `async_client`: Async HTTP client for API calls
- `user_id`: Test user UUID
- `async_http_requester`: Request headers and auth
- `db_session`: Database session
- `sync_central_rag_index`: Synced RAG index for integration tests

## Notes

- Tests use mocking extensively to avoid external LLM calls
- E2E tests may require database and OpenSearch setup
- Some tests validate the router's decision-making, not just outcomes
- Streaming response parsing is handled by utility functions
- Tests follow the same patterns as other features (smart_targets, gov_uk_search)
