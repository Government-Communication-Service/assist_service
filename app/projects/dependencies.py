from typing import Annotated, Literal, Optional
from uuid import UUID

from fastapi import Depends, Path

from app.auth.verify_service import verify_and_get_user_from_path_and_header
from app.database.db_session import DbSession
from app.database.models import Project, User
from app.projects.exceptions import ProjectNotFoundError
from app.projects.queries import get_user_project


# All project routes resolve the project through these dependencies, so a later shared-projects phase
# only needs to swap the ownership check for a membership check in _get_project.
async def _get_project(
    db_session: DbSession, user: User, project_uuid: UUID, lock: Optional[Literal["share", "update"]]
) -> Project:
    project = await get_user_project(db_session=db_session, user_id=user.id, project_uuid=project_uuid, lock=lock)
    if not project:
        raise ProjectNotFoundError(f"No project for this user with project_uuid: {project_uuid}")
    return project


async def verify_and_get_project_from_path(
    db_session: DbSession,
    project_uuid: Annotated[UUID, Path()],
    user: User = Depends(verify_and_get_user_from_path_and_header),
) -> Project:
    """For routes that only read the project."""
    return await _get_project(db_session, user, project_uuid, lock=None)


async def verify_and_share_lock_project_from_path(
    db_session: DbSession,
    project_uuid: Annotated[UUID, Path()],
    user: User = Depends(verify_and_get_user_from_path_and_header),
) -> Project:
    """For routes that attach things to the project: the project can't be deleted until the request ends."""
    return await _get_project(db_session, user, project_uuid, lock="share")


async def verify_and_update_lock_project_from_path(
    db_session: DbSession,
    project_uuid: Annotated[UUID, Path()],
    user: User = Depends(verify_and_get_user_from_path_and_header),
) -> Project:
    """For routes that change or delete the project: concurrent changes, deletes and attaches wait for this one."""
    return await _get_project(db_session, user, project_uuid, lock="update")
