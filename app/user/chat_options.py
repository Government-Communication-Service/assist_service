from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import set_committed_value

from app.chat.schemas import ChatOptions
from app.config import CHAT_OPTIONS_RESET_AT, DEFAULT_CHAT_OPTIONS, RECORD_USER_CHAT_OPTIONS
from app.database.models import Chat, UserChatOptions

CHAT_OPTION_FIELDS = ("use_rag", "use_gov_uk_search_api", "use_smart_targets")


async def _get_chosen_chat_options(db_session: AsyncSession, user_id: int) -> dict[str, bool | None]:
    """Return the options the user has chosen, with None for options they've never set.

    Reads the user's latest live `user_chat_options` row created at or after `CHAT_OPTIONS_RESET_AT`.

    Args:
        db_session: Database session.
        user_id: ID of the user.

    Returns:
        Each chat option field mapped to the user's chosen value, or None.
    """
    conditions = [UserChatOptions.user_id == user_id, UserChatOptions.deleted_at.is_(None)]
    if CHAT_OPTIONS_RESET_AT is not None:
        conditions.append(UserChatOptions.created_at >= CHAT_OPTIONS_RESET_AT)
    result = await db_session.execute(
        select(UserChatOptions)
        .where(*conditions)
        .order_by(UserChatOptions.created_at.desc(), UserChatOptions.id.desc())
        .limit(1)
    )
    latest = result.scalars().first()
    return {field: getattr(latest, field) if latest else None for field in CHAT_OPTION_FIELDS}


def _with_defaults(chosen: dict[str, bool | None]) -> dict[str, bool]:
    return {field: DEFAULT_CHAT_OPTIONS[field] if chosen[field] is None else chosen[field] for field in chosen}


async def get_current_chat_options(db_session: AsyncSession, user_id: int) -> dict[str, bool]:
    """Return the user's current chat options.

    Args:
        db_session: Database session.
        user_id: ID of the user.

    Returns:
        The user's chosen options, with `DEFAULT_CHAT_OPTIONS` for any they've never set.
    """
    return _with_defaults(await _get_chosen_chat_options(db_session, user_id))


async def resolve_chat_options(
    db_session: AsyncSession, user_id: int, input_data: ChatOptions, chat_options: dict[str, bool | None] | None = None
) -> None:
    """Fill any chat option left as None on `input_data`, in place.

    Uses the chat's options for a message in an existing chat, and the user's current options for a new
    chat or for any option the chat has no value for.

    Args:
        db_session: Database session.
        user_id: ID of the user sending the request.
        input_data: Request data carrying the chat option flags.
        chat_options: The existing chat's options, or None for a new chat.
    """
    for field in CHAT_OPTION_FIELDS:
        if getattr(input_data, field) is None and chat_options is not None:
            setattr(input_data, field, chat_options[field])
    if all(getattr(input_data, field) is not None for field in CHAT_OPTION_FIELDS):
        return
    current = await get_current_chat_options(db_session, user_id)
    for field in CHAT_OPTION_FIELDS:
        if getattr(input_data, field) is None:
            setattr(input_data, field, current[field])


async def record_chat_options(
    db_session: AsyncSession,
    user_id: int,
    options: dict[str, bool],
    previous_chat_options: dict[str, bool | None] | None = None,
    message_id: int | None = None,
) -> bool:
    """Append a `user_chat_options` row if the user changed an option.

    For a new chat, an option counts as changed if it differs from the user's current options. For a
    message in an existing chat, it counts only if it also differs from the chat's previous options, so
    sending a message in an older chat doesn't overwrite a choice made in another chat.

    The new row sets the changed options and carries forward the rest of the user's latest row, so
    options the user has never set stay None and follow `DEFAULT_CHAT_OPTIONS`. Does nothing while
    `RECORD_USER_CHAT_OPTIONS` is off.

    Args:
        db_session: Database session. The caller commits.
        user_id: ID of the user.
        options: The three chat option flags sent with the message.
        previous_chat_options: The existing chat's options before this message, or None for a new chat.
        message_id: ID of the message sent with these options, if any.

    Returns:
        True if a row was added.
    """
    if not RECORD_USER_CHAT_OPTIONS:
        return False
    toggled = [
        field
        for field in CHAT_OPTION_FIELDS
        if previous_chat_options is None or options[field] != previous_chat_options[field]
    ]
    if not toggled:
        return False
    chosen = await _get_chosen_chat_options(db_session, user_id)
    current = _with_defaults(chosen)
    changed = {field for field in toggled if options[field] != current[field]}
    if not changed:
        return False
    new_row = {field: options[field] if field in changed else chosen[field] for field in CHAT_OPTION_FIELDS}
    db_session.add(UserChatOptions(user_id=user_id, message_id=message_id, **new_row))
    await db_session.flush()
    return True


async def update_chat_options(db_session: AsyncSession, chat: Chat, options: dict[str, bool]) -> None:
    """Save `options` as the chat's current options, if they differ.

    Also updates the in-memory `chat`, which may belong to another session, without marking it changed.

    Args:
        db_session: Database session. The caller commits.
        chat: The chat.
        options: The three chat option flags.
    """
    if all(getattr(chat, field) == options[field] for field in CHAT_OPTION_FIELDS):
        return
    await db_session.execute(update(Chat).where(Chat.id == chat.id).values(**options))
    for field in CHAT_OPTION_FIELDS:
        set_committed_value(chat, field, options[field])
