import random

from tasks import UserProtocol

_QUERIES = [
    "What are the key principles of effective government communication?",
    "Summarise the GCS evaluation framework.",
    "What is MCOM and how does it apply to campaigns?",
    "How should I approach an integrated communications campaign?",
    "What are the GOV.UK content design principles?",
    "Explain the role of behavioural insights in public communications.",
    "What metrics should I use to measure campaign effectiveness?",
    "How do I write an effective press release for a government announcement?",
    "What are the best practices for social media in government comms?",
    "How should I plan a communications strategy for a policy launch?",
]


def create_chat_no_rag(user: UserProtocol) -> None:
    query = random.choice(_QUERIES)
    with user.client.post(
        f"/v1/chats/users/{user.user_uuid}",
        headers={**user.auth_headers, "Content-Type": "application/json"},
        json={"query": query, "use_rag": False, "use_gov_uk_search_api": False},
        catch_response=True,
        name="POST /v1/chats/users/{uuid} (no RAG)",
    ) as resp:
        if resp.status_code == 200:
            chat_uuid = resp.json().get("uuid")
            if chat_uuid:
                user.chat_uuids.append(chat_uuid)
            resp.success()
        else:
            resp.failure(f"Unexpected {resp.status_code}: {resp.text[:200]}")


def create_chat_with_rag(user: UserProtocol) -> None:
    query = random.choice(_QUERIES)
    with user.client.post(
        f"/v1/chats/users/{user.user_uuid}",
        headers={**user.auth_headers, "Content-Type": "application/json"},
        json={"query": query, "use_rag": True, "use_gov_uk_search_api": False},
        catch_response=True,
        name="POST /v1/chats/users/{uuid} (RAG)",
    ) as resp:
        if resp.status_code == 200:
            chat_uuid = resp.json().get("uuid")
            if chat_uuid:
                user.chat_uuids.append(chat_uuid)
            resp.success()
        else:
            resp.failure(f"Unexpected {resp.status_code}: {resp.text[:200]}")
