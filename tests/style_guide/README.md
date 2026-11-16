# Style guide tests

Tests for the GOV.UK style guide check behind the chat model's `check_style_guide` tool. None of them call Bedrock: the engine tests use a fake model, and the service tests mock the engine.

| File | What it covers |
|---|---|
| `test_rules.py` | `rule_mapping.json`: every rule has a priority of 0, 100, 1000 or 2000 and a valid `check`; how checks are loaded. |
| `test_matching.py` | Pattern and correct-form matching, match order tags, the built-in matchers (long sentences, American spelling), chunking and quote location. |
| `test_engine.py` | `run_check` with a fake model: how calls are planned and batched, structured output and thinking settings, quote checks, deduplication, failed and timed-out calls. |
| `test_report.py` | The tool result: grouping by rule, ordering by priority and source, the size limit and what is left out, warnings and errors. |
| `test_unit_service.py` | Reading an attached document, and the tool's arguments. |
| `test_integration_service.py` | `check_style_guide_for_tool`: text or attached document, size and empty-text errors, engine failures. |
| `test_e2e_chat.py` | The tool called from a chat request, with the check mocked. |

Run them in the api container against the test database:

```bash
docker exec -e POSTGRES_DB=testcopilot api python -m pytest tests/style_guide
```

The checker's accuracy is measured with real model calls by the evaluation scripts in `scripts/style_guide/evaluation/` (see the main README) and the `style_guide` cases in `scripts/evaluation/`.
