import random

from config import STYLE_GUIDE_USE_CASE_UUID

from tasks import UserProtocol

# Text snippets containing known style guide violations for realistic testing
_CONTENT_SNIPPETS = [
    (
        "The utilisation of this new digital service will facilitate the implementation "
        "of streamlined processes going forward. We are committed to delivering a robust "
        "and fit for purpose solution that meets the needs of all stakeholders. "
        "Please be advised that the aforementioned measures will be actioned accordingly."
    ),
    (
        "The Department has recently announced a new initiative to improve service delivery "
        "across all government departments. This will help to ensure that citizens can "
        "access the information they need in a timely manner. The project is currently "
        "in the process of being developed and will be rolled out in due course. "
        "Going forward, we will be looking to leverage synergies across the organisation."
    ),
    (
        "We are pleased to announce the launch of our new digital transformation programme. "
        "The programme will enable us to provide a more efficient and effective service "
        "to members of the public. We have worked closely with a number of key stakeholders "
        "to develop this solution and are confident it will deliver significant benefits. "
        "Please note that all queries should be directed to the relevant department."
    ),
    (
        "The Government has today published its new strategy for economic growth. "
        "The strategy sets out a number of key priorities for the coming years, "
        "including investment in infrastructure, skills and innovation. "
        "We are committed to working in partnership with businesses, local authorities "
        "and other organisations to deliver on these priorities. "
        "The strategy has been developed following an extensive period of consultation."
    ),
    (
        "Following the recent review of our communications strategy, we have identified "
        "a number of areas where improvements can be made going forward. "
        "We will be looking to utilise digital channels more effectively in order to "
        "reach a wider audience and ensure our key messages are communicated clearly. "
        "The implementation of these changes will be managed by the communications team."
    ),
]


def check_style_guide_via_chat(user: UserProtocol) -> None:
    """Check text against the GOV.UK style guide via the chat endpoint, as the frontend does."""
    content = random.choice(_CONTENT_SNIPPETS)

    assert STYLE_GUIDE_USE_CASE_UUID is not None and len(STYLE_GUIDE_USE_CASE_UUID) > 1, (
        "Style Guide UUID has not been set in .env/.env.dev"
    )

    with user.client.post(
        f"/v1/chats/users/{user.user_uuid}",
        headers={**user.auth_headers, "Content-Type": "application/json"},
        json={
            "query": content,
            "use_case_id": STYLE_GUIDE_USE_CASE_UUID,
            "use_rag": False,
            "use_gov_uk_search_api": False,
        },
        catch_response=True,
        name="POST /v1/chats/users/{uuid} (style guide)",
    ) as resp:
        if resp.status_code == 200:
            chat_uuid = resp.json().get("uuid")
            if chat_uuid:
                user.chat_uuids.append(chat_uuid)
            resp.success()
        else:
            resp.failure(f"Unexpected {resp.status_code}: {resp.text[:200]}")
