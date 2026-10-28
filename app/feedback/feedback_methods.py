import json

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.schemas import FeedbackRequest, MessageFeedbackEnum, MessageFeedbackResponse
from app.database.db_operations import DbOperations
from app.database.models import Message
from app.logs.logs_handler import logger

__FEEDBACK_SCORE_LOOKUP = {score.value: score for score in MessageFeedbackEnum}


async def get_message_feedback_labels(db_session: AsyncSession):
    return await DbOperations.get_message_feedback_labels_list(db_session)


async def process_message_feedback(
    db_session: AsyncSession, message: Message, feedback_request: FeedbackRequest
) -> MessageFeedbackResponse:
    """
    Processes user feedback for a message, including positive, negative, and removed feedback.
    It updates or creates a feedback record based on the provided score and
    user actions. It handles special cases for removed feedback

    Parameters:
        db_session (AsyncSession): sqlalchemy async db session
        message (Message): The message object containing the message ID.
        feedback_request (FeedbackRequest): The feedback request object containing:
            - score (FeedbackScoreEnum): The score indicating the type of feedback (positive, negative, removed).
            - freetext (str): Optional text providing additional context for the feedback.
            - label (str): Optional UUID for a feedback label (applicable for negative feedback).

    Returns:
        MessageFeedbackResponse: A response object containing the details of the processed feedback.

    Notes:
        - If the feedback is marked as removed, the existing feedback is updated and marked as deleted.
        - Only Negative feedback can have an associated label.

    """

    message = await DbOperations.get_message_by_uuid(db_session=db_session, message_uuid=message.uuid)

    user_feedback = __FEEDBACK_SCORE_LOOKUP.get(feedback_request.score)
    user_removed_feedback = user_feedback == user_feedback.removed

    is_negative = user_feedback == MessageFeedbackEnum.negative

    # special case for removed feedback
    if user_feedback == MessageFeedbackEnum.removed:
        # update existing feedback and mark as deleted
        feedback = await DbOperations.get_feedback(db_session, message.id)
        if feedback:
            feedback = await DbOperations.delete_feedback(db_session, feedback=feedback)
            return MessageFeedbackResponse(**feedback.client_response())

    # Retrieve the corresponding score name e.g. positive, negative
    feedback_score = await DbOperations.get_feedback_score_by_name(db_session=db_session, score=user_feedback.name)
    score_id = feedback_score.id

    # initial feedback parameters, which should reset when a feedback is updated from negative to positive or vice versa
    params = {
        "message_id": message.id,
        "feedback_score_id": score_id,
        "freetext": feedback_request.freetext,
        "deleted_at": None,
        "feedback_label_id": None,
    }

    logger.debug(
        "process_message_feedback:"
        + json.dumps(
            {
                "original_data": feedback_request.__dict__,
                "user_removed_feedback": user_removed_feedback,
                "is_negative": is_negative,
                "params": params,
            },
            indent=2,
        ),
    )

    feedback_label_id = None
    if is_negative and feedback_request.label:
        label = await DbOperations.get_feedback_label_by_name(
            db_session=db_session, feedback_label=feedback_request.label
        )
        if label:
            feedback_label_id = label.id

    feedback = await DbOperations.upsert_feedback(
        db_session,
        message_id=message.id,
        feedback_score_id=score_id,
        freetext=feedback_request.freetext,
        feedback_label_id=feedback_label_id,
    )
    return MessageFeedbackResponse(**feedback.client_response())
