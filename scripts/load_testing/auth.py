import uuid

from config import AUTH_SECRET_KEY, FRONTEND_URL, USER_AGENT


class AuthMixin:
    """
    Mixin for Locust HttpUser classes.
    Handles user + session setup in on_start so every virtual user
    gets its own UUID and Session-Auth token.
    """

    def on_start(self):
        self.user_uuid = str(uuid.uuid4())
        self.auth_headers = {
            "Auth-Token": AUTH_SECRET_KEY,
            "User-Key-UUID": self.user_uuid,
            # Browser identity headers — required to pass AWS WAF AWSManagedRulesCommonRuleSet.
            # Without these the requests look like bare HTTP tool traffic and trip the bot rules.
            "User-Agent": USER_AGENT,
            "Origin": FRONTEND_URL,
            "Referer": f"{FRONTEND_URL}/",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-GB,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Dest": "empty",
            "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"macOS"',
        }

        # Step 1: Create user — UUID must exist in the DB before any other calls
        with self.client.post(
            "/v1/users",
            headers=self.auth_headers,
            json={"uuid": self.user_uuid},
            catch_response=True,
            name="setup: POST /v1/users",
        ) as resp:
            if resp.status_code in (200, 201):
                resp.success()
            else:
                resp.failure(f"User creation failed: {resp.status_code} {resp.text[:200]}")
                return

        # Step 2: Create auth session
        with self.client.post(
            "/v1/auth-sessions",
            headers=self.auth_headers,
            catch_response=True,
            name="setup: POST /v1/auth-sessions",
        ) as resp:
            if resp.status_code == 200:
                session_auth = resp.json().get("Session-Auth")
                self.auth_headers["Session-Auth"] = session_auth
                resp.success()
            else:
                resp.failure(f"Auth session creation failed: {resp.status_code} {resp.text[:200]}")

        # Per-user state for tracking uploaded documents and created chats
        self.uploaded_document_uuids: list[str] = []
        self.chat_uuids: list[str] = []
        self.audience_document_uuids: list[str] = []
        self.last_analysis_result: dict | None = None
        self.last_audience_filename: str | None = None

    def on_stop(self):
        """Delete documents and archive chats created during this user's session."""
        all_doc_uuids = self.uploaded_document_uuids + self.audience_document_uuids
        for doc_uuid in all_doc_uuids:
            self.client.delete(
                f"/v1/users/{self.user_uuid}/documents/{doc_uuid}",
                headers=self.auth_headers,
                name="cleanup: DELETE /v1/users/{uuid}/documents/{doc_uuid}",
            )

        for chat_uuid in self.chat_uuids:
            self.client.patch(
                f"/v1/chats/users/{self.user_uuid}/chats/{chat_uuid}/archive",
                headers=self.auth_headers,
                name="cleanup: PATCH /v1/chats/users/{uuid}/chats/{uuid}/archive",
            )
