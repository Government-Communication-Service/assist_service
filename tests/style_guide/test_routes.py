from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_check_failure_returns_fixed_detail(async_client, auth_token_only_headers):
    with patch(
        "app.style_guide.routes.check_llm_validation_rules",
        new_callable=AsyncMock,
        side_effect=RuntimeError("secret internals"),
    ):
        response = await async_client.post(
            "/v1/style-guide/check", headers=auth_token_only_headers, json={"content": "Some text"}
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "Error checking style guide"}
