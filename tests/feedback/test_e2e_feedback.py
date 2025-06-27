import logging

import pytest

from app.api import ENDPOINTS
from app.chat.schemas import FeedbackRequest
from app.database.table import FeedbackLabelTable, FeedbackTable

api = ENDPOINTS()


logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.streaming,
]


class TestUserChatsV1:
    @pytest.mark.asyncio
    async def test_get_feedback_labels(self, async_client, async_http_requester):
        get_feedback_labels_url = api.get_feedback_labels()
        response = await async_http_requester(
            "test_add_feedback_to_chat_messages",
            async_client.get,
            get_feedback_labels_url,
        )

        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_add_negative_feedback_with_label_to_chat_messages(
        self, async_client, user_id, chat, async_http_requester
    ):
        create_feedback_url = api.add_message_feedback(user_uuid=user_id, message_uuid=chat.message.uuid)
        label_model = FeedbackLabelTable().get_one_by("label", "Not factually correct")

        feedback_request_json = FeedbackRequest(
            score=-1, freetext="Needs improvement", label=str(label_model.uuid)
        ).to_dict()

        response = await async_http_requester(
            "test_add_feedback_to_chat_messages",
            async_client.put,
            create_feedback_url,
            json=feedback_request_json,
        )

        feedback_uuid = response["uuid"]
        feedback = FeedbackTable().get_by_uuid(feedback_uuid)
        # check freetext is same as the one sent
        assert feedback.freetext == feedback_request_json["freetext"]
        # check negative score stored in the table, negative score is stored as 1 in the database table
        assert feedback.feedback_score_id == 1
        # check label is same as the one sent
        assert feedback.feedback_label_id == label_model.id

    @pytest.mark.asyncio
    async def test_add_negative_feedback_to_chat_messages_without_label(
        self, async_client, user_id, chat, async_http_requester
    ):
        create_feedback_url = api.add_message_feedback(user_uuid=user_id, message_uuid=chat.message.uuid)

        feedback_request_json = FeedbackRequest(score=-1, freetext="Needs improvement").to_dict()

        response = await async_http_requester(
            "test_add_feedback_to_chat_messages",
            async_client.put,
            create_feedback_url,
            json=feedback_request_json,
        )

        feedback_uuid = response["uuid"]
        feedback = FeedbackTable().get_by_uuid(feedback_uuid)
        # check freetext is same as the one sent
        assert feedback.freetext == feedback_request_json["freetext"]

        # check negative score stored in the table, negative score is stored as 1 in the database table
        assert feedback.feedback_score_id == 1

        assert feedback.feedback_label_id is None

    @pytest.mark.asyncio
    async def test_add_positive_feedback_to_chat_messages(self, async_client, user_id, chat, async_http_requester):
        create_feedback_url = api.add_message_feedback(user_uuid=user_id, message_uuid=chat.message.uuid)

        # check labels are ignored for positive feedbacks
        label_model = FeedbackLabelTable().get_one_by("label", "Not factually correct")

        feedback_request_json = FeedbackRequest(
            score=1, label=str(label_model.uuid), freetext="Needs improvement"
        ).to_dict()

        response = await async_http_requester(
            "test_add_feedback_to_chat_messages",
            async_client.put,
            create_feedback_url,
            json=feedback_request_json,
        )

        feedback_uuid = response["uuid"]
        feedback = FeedbackTable().get_by_uuid(feedback_uuid)
        # check freetext is same as the one sent
        assert feedback.freetext == feedback_request_json["freetext"]

        # check positive score, which is stored as 1 in the database table
        assert feedback.feedback_score_id == 2

        # check labels are ignored for positive feedbacks
        assert feedback.feedback_label_id is None

    @pytest.mark.asyncio
    async def test_add_negative_feedback_to_chat_messages_without_freetext(
        self, async_client, user_id, chat, async_http_requester
    ):
        create_feedback_url = api.add_message_feedback(user_uuid=user_id, message_uuid=chat.message.uuid)
        label_model = FeedbackLabelTable().get_one_by("label", "Not factually correct")

        feedback_request_json = FeedbackRequest(score=-1, label=str(label_model.uuid)).to_dict()

        response = await async_http_requester(
            "test_add_feedback_to_chat_messages",
            async_client.put,
            create_feedback_url,
            json=feedback_request_json,
        )

        feedback_uuid = response["uuid"]
        feedback = FeedbackTable().get_by_uuid(feedback_uuid)
        # check freetext is same as the one sent
        assert feedback.freetext == feedback_request_json["freetext"]
        # check negative score, which is stored as 1 in the database table
        assert feedback.feedback_score_id == 1
        # check label is same as the one sent
        assert feedback.feedback_label_id == label_model.id

    @pytest.mark.asyncio
    async def test_add_negative_feedback_without_label_to_chat_messages_without_freetext(
        self, async_client, user_id, chat, async_http_requester
    ):
        create_feedback_url = api.add_message_feedback(user_uuid=user_id, message_uuid=chat.message.uuid)

        feedback_request_json = FeedbackRequest(score=-1).to_dict()

        response = await async_http_requester(
            "test_add_feedback_to_chat_messages",
            async_client.put,
            create_feedback_url,
            json=feedback_request_json,
        )

        feedback_uuid = response["uuid"]
        feedback = FeedbackTable().get_by_uuid(feedback_uuid)
        # check freetext is same as the one sent
        assert feedback.freetext == feedback_request_json["freetext"]
        # check negative score stored in the table, negative score is stored as 1 in the database table
        assert feedback.feedback_score_id == 1

        assert feedback.feedback_label_id is None

    @pytest.mark.asyncio
    async def test_add_positive_feedback_to_chat_messages_without_freetext(
        self, async_client, user_id, chat, async_http_requester
    ):
        create_feedback_url = api.add_message_feedback(user_uuid=user_id, message_uuid=chat.message.uuid)

        feedback_request_json = FeedbackRequest(score=1).to_dict()

        response = await async_http_requester(
            "test_add_feedback_to_chat_messages",
            async_client.put,
            create_feedback_url,
            json=feedback_request_json,
        )

        feedback_uuid = response["uuid"]
        feedback = FeedbackTable().get_by_uuid(feedback_uuid)
        # check freetext is same as the one sent
        assert feedback.freetext == feedback_request_json["freetext"]
        # check score is same as the one sent
        # check positive score , which  is stored as 2 in the database table
        assert feedback.feedback_score_id == 2

        assert feedback.feedback_label_id is None

    @pytest.mark.asyncio
    async def test_remove_negative_feedback_then_update_as_positive_feedback(
        self, async_client, user_id, chat, async_http_requester
    ):
        create_feedback_url = api.add_message_feedback(user_uuid=user_id, message_uuid=chat.message.uuid)
        label_model = FeedbackLabelTable().get_one_by("label", "Not factually correct")

        # Step 1: send negative feedback
        feedback_request_json = FeedbackRequest(
            score=-1, label=str(label_model.uuid), freetext="Needs improvement"
        ).to_dict()

        response = await async_http_requester(
            "test_add_feedback_to_chat_messages",
            async_client.put,
            create_feedback_url,
            json=feedback_request_json,
        )

        feedback_uuid = response["uuid"]
        feedback = FeedbackTable().get_by_uuid(feedback_uuid)

        # check freetext is same as the one sent
        assert feedback.freetext == feedback_request_json["freetext"]
        # check score is same as the one sent
        # check positive score , which  is stored as 2 in the database table
        assert feedback.feedback_score_id == 1

        # check label is same as the one sent
        assert feedback.feedback_label_id == label_model.id

        # Step 2: remove feedback
        feedback_request_json = FeedbackRequest(score=0, freetext="Needs improvement").to_dict()

        response = await async_http_requester(
            "test_add_feedback_to_chat_messages",
            async_client.put,
            create_feedback_url,
            json=feedback_request_json,
        )

        feedback_uuid = response["uuid"]
        feedback = FeedbackTable().get_by_uuid(feedback_uuid)

        # check freetext is same as the one sent
        assert feedback.deleted_at is not None

        # Step 3: send positive feedback
        feedback_request_json = FeedbackRequest(score=1, freetext="Needs improvement").to_dict()
        await async_http_requester(
            "test_add_feedback_to_chat_messages",
            async_client.put,
            create_feedback_url,
            json=feedback_request_json,
        )
        feedback = FeedbackTable().get_by_uuid(feedback_uuid)

        # check freetext is same as the one sent
        assert feedback.freetext == feedback_request_json["freetext"]
        # check score is same as the one sent
        # check positive score , which  is stored as 2 in the database table
        assert feedback.feedback_score_id == 2

        # check label is removed as the feedback is positive
        assert feedback.feedback_label_id is None
        assert feedback.deleted_at is None
