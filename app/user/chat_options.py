from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import set_committed_value

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
    db_session: AsyncSession, user_id: int, options: dict[str, bool | None]
) -> dict[str, bool]:
    """Return the user's current chat options with `options` applied, without saving them.

    Args:
        db_session: Database session.
        user_id: ID of the user.
        options: Chat option flags, with None for any that follow the user's current option.

    Returns:
        Each chat option field mapped to the value in `options`, or the user's current value if None.
    """
    current = await get_current_chat_options(db_session, user_id)
    return {field: current[field] if options[field] is None else options[field] for field in CHAT_OPTION_FIELDS}


async def set_chat_options(db_session: AsyncSession, user_id: int, options: dict[str, bool | None]) -> None:
    """Set the user's chat options.

    Options left as None keep their current value. A `user_chat_options` row is appended only if an option
    changes. It sets the changed options and carries forward the rest of the user's latest row, so options the
    user has never set stay None and follow `DEFAULT_CHAT_OPTIONS`. Does nothing while
    `RECORD_USER_CHAT_OPTIONS` is off.

    Args:
        db_session: Database session. The caller commits.
        user_id: ID of the user.
        options: Chat option flags to set, with None for any to leave as they are.
    """
    if not RECORD_USER_CHAT_OPTIONS:
        return
    chosen = await _get_chosen_chat_options(db_session, user_id)
    current = _with_defaults(chosen)
    changed = {field for field in CHAT_OPTION_FIELDS if options[field] is not None and options[field] != current[field]}
    if not changed:
        return
    new_row = {field: options[field] if field in changed else chosen[field] for field in CHAT_OPTION_FIELDS}
    db_session.add(UserChatOptions(user_id=user_id, **new_row))
    await db_session.flush()


async def update_chat_options(db_session: AsyncSession, chat: Chat, options: dict[str, bool]) -> None:
    """Save `options` on the chat as the options its latest message used, if they differ.

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
