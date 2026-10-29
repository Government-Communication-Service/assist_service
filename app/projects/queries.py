from datetime import datetime
from typing import Any, Literal, Mapping, Optional, Sequence
from uuid import UUID

from sqlalchemy import desc, func, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Chat, Project
from app.projects.exceptions import ProjectNameConflictError


async def get_user_project(
    db_session: AsyncSession,
    user_id: int,
    project_uuid: UUID,
    lock: Optional[Literal["share", "update"]] = None,
) -> Optional[Project]:
    """
    Returns the user's live project, or None if it is missing, deleted or not theirs.

    Args:
        lock: Lock the project row until the transaction ends: "share" (FOR SHARE) blocks changes to it,
            "update" (FOR UPDATE) also blocks other share or update locks. A caller that waits on a lock
            re-checks deleted_at, so it gets None if the holder deleted the project.
    """
    stmt = select(Project).where(Project.uuid == project_uuid, Project.user_id == user_id, Project.deleted_at.is_(None))
    if lock:
        stmt = stmt.with_for_update(read=lock == "share")
    return await db_session.scalar(stmt)


async def list_user_projects(db_session: AsyncSession, user_id: int) -> Sequence[Project]:
    """Returns the user's live projects, alphabetically by name."""
    result = await db_session.scalars(
        select(Project)
        .where(Project.user_id == user_id, Project.deleted_at.is_(None))
        .order_by(func.lower(Project.name), Project.id)
    )
    return result.all()


async def create_project(db_session: AsyncSession, user_id: int, name: str, description: Optional[str]) -> Project:
    """Creates a project, raising ProjectNameConflictError if the user already has a live project with this name."""
    stmt = insert(Project).values(user_id=user_id, name=name, description=description).returning(Project)
    return await _with_name_conflict(db_session, stmt, name)


async def update_project(db_session: AsyncSession, project: Project, values: Mapping[str, Any]) -> Project:
    """Updates the given columns, raising ProjectNameConflictError if a rename clashes with another live project."""
    stmt = update(Project).where(Project.id == project.id).values(**values).returning(Project)
    return await _with_name_conflict(db_session, stmt, values.get("name", project.name))


async def _with_name_conflict(db_session: AsyncSession, stmt, name: str) -> Project:
    try:
        return await db_session.scalar(stmt)
    except IntegrityError as e:
        raise ProjectNameConflictError(f"A project named '{name}' already exists for this user") from e


async def delete_project(db_session: AsyncSession, project: Project) -> None:
    """Soft deletes a project and moves its chats back to plain chat history, keeping their updated_at."""
    await db_session.execute(
        update(Chat).where(Chat.project_id == project.id).values(project_id=None, updated_at=Chat.updated_at)
    )
    await db_session.execute(update(Project).where(Project.id == project.id).values(deleted_at=datetime.now()))


async def list_project_chats(db_session: AsyncSession, project: Project) -> Sequence[Chat]:
    """Returns the project's live chats, most recently updated first."""
    result = await db_session.scalars(
        select(Chat)
        .where(Chat.project_id == project.id, Chat.deleted_at.is_(None))
        .order_by(desc(Chat.updated_at), desc(Chat.id))
    )
    return result.all()


async def add_chat_to_project(
    db_session: AsyncSession, user_id: int, chat_uuid: UUID, project: Project
) -> Optional[Chat]:
    """
    Moves the user's live chat into the project, keeping its updated_at.

    Returns:
        The chat, or None if the user has no live chat with this UUID.
    """
    chat_id = await db_session.scalar(
        update(Chat)
        .where(Chat.uuid == chat_uuid, Chat.user_id == user_id, Chat.deleted_at.is_(None))
        .values(project_id=project.id, updated_at=Chat.updated_at)
        .returning(Chat.id)
    )
    if chat_id is None:
        return None
    # Selected rather than returned by the UPDATE, which doesn't load Chat.project_uuid.
    return await db_session.scalar(select(Chat).where(Chat.id == chat_id))


async def remove_chat_from_project(db_session: AsyncSession, chat_uuid: UUID, project: Project) -> bool:
    """
    Moves a chat out of the project, keeping its updated_at.

    Returns:
        False if the project has no chat with this UUID.
    """
    chat_id = await db_session.scalar(
        update(Chat)
        .where(Chat.uuid == chat_uuid, Chat.project_id == project.id)
        .values(project_id=None, updated_at=Chat.updated_at)
        .returning(Chat.id)
    )
    return chat_id is not None
