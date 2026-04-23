from tasks import UserProtocol


def healthcheck(user: UserProtocol) -> None:
    user.client.get("/healthcheck", headers=user.auth_headers, name="GET /healthcheck")


def list_chats(user: UserProtocol) -> None:
    user.client.get(
        f"/v1/chats/users/{user.user_uuid}/chats",
        headers=user.auth_headers,
        name="GET /v1/chats/users/{uuid}/chats",
    )


def list_documents(user: UserProtocol) -> None:
    user.client.get(
        f"/v1/users/{user.user_uuid}/documents",
        headers=user.auth_headers,
        name="GET /v1/users/{uuid}/documents",
    )
