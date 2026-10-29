# ruff: noqa: B008
from uuid import UUID

from fastapi import APIRouter, Depends, Response, status

from app.api.endpoints import ENDPOINTS
from app.auth.verify_service import (
    verify_and_get_auth_session_from_header,
    verify_and_get_user_from_path_and_header,
    verify_auth_token,
)
from app.chat.schemas import ChatBasicResponse
from app.database.db_session import DbSession
from app.database.models import Project, User
from app.projects import queries
from app.projects.dependencies import (
    verify_and_get_project_from_path,
    verify_and_share_lock_project_from_path,
    verify_and_update_lock_project_from_path,
)
from app.projects.exceptions import ProjectChatNotFoundError
from app.projects.schemas import (
    ProjectChatsResponse,
    ProjectCreateRequest,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdateRequest,
)

router = APIRouter()

PROJECT_DEPENDENCIES = [
    Depends(verify_auth_token),
    Depends(verify_and_get_user_from_path_and_header),
    Depends(verify_and_get_auth_session_from_header),
]


@router.post(
    path=ENDPOINTS.USER_PROJECTS,
    dependencies=PROJECT_DEPENDENCIES,
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_user_project(
    db_session: DbSession,
    data: ProjectCreateRequest,
    user: User = Depends(verify_and_get_user_from_path_and_header),
) -> ProjectResponse:
    """
    Create a project for the user, with a name (unique per user, ignoring case) and an optional description.
    """
    project = await queries.create_project(
        db_session=db_session, user_id=user.id, name=data.name, description=data.description
    )
    return ProjectResponse.model_validate(project)


@router.get(path=ENDPOINTS.USER_PROJECTS, dependencies=PROJECT_DEPENDENCIES, response_model=ProjectListResponse)
async def get_user_projects(
    db_session: DbSession,
    user: User = Depends(verify_and_get_user_from_path_and_header),
) -> ProjectListResponse:
    """
    List the user's projects alphabetically by name.
    """
    projects = await queries.list_user_projects(db_session=db_session, user_id=user.id)
    return ProjectListResponse(projects=[ProjectResponse.model_validate(project) for project in projects])


@router.get(path=ENDPOINTS.USER_PROJECT, dependencies=PROJECT_DEPENDENCIES, response_model=ProjectResponse)
async def get_user_project(project: Project = Depends(verify_and_get_project_from_path)) -> ProjectResponse:
    """
    Get one of the user's projects.
    """
    return ProjectResponse.model_validate(project)


@router.patch(path=ENDPOINTS.USER_PROJECT, dependencies=PROJECT_DEPENDENCIES, response_model=ProjectResponse)
async def patch_user_project(
    db_session: DbSession,
    data: ProjectUpdateRequest,
    project: Project = Depends(verify_and_update_lock_project_from_path),
) -> ProjectResponse:
    """
    Update a project's name and/or description. Only the fields sent are changed.
    """
    project = await queries.update_project(
        db_session=db_session, project=project, values=data.model_dump(exclude_unset=True)
    )
    return ProjectResponse.model_validate(project)


@router.delete(
    path=ENDPOINTS.USER_PROJECT,
    dependencies=PROJECT_DEPENDENCIES,
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_user_project(
    db_session: DbSession,
    project: Project = Depends(verify_and_update_lock_project_from_path),
) -> Response:
    """
    Delete a project. Its chats stay in the user's chat history.
    """
    await queries.delete_project(db_session=db_session, project=project)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(path=ENDPOINTS.USER_PROJECT_CHATS, dependencies=PROJECT_DEPENDENCIES, response_model=ProjectChatsResponse)
async def get_user_project_chats(
    db_session: DbSession,
    project: Project = Depends(verify_and_get_project_from_path),
) -> ProjectChatsResponse:
    """
    List the chats in a project, most recently updated first. Archived chats are not included.
    """
    chats = await queries.list_project_chats(db_session=db_session, project=project)
    return ProjectChatsResponse(
        project_uuid=project.uuid, chats=[ChatBasicResponse.model_validate(chat) for chat in chats]
    )


@router.put(path=ENDPOINTS.USER_PROJECT_CHAT, dependencies=PROJECT_DEPENDENCIES, response_model=ChatBasicResponse)
async def put_user_project_chat(
    db_session: DbSession,
    chat_uuid: UUID,
    user: User = Depends(verify_and_get_user_from_path_and_header),
    project: Project = Depends(verify_and_share_lock_project_from_path),
) -> ChatBasicResponse:
    """
    Add one of the user's chats to a project. A chat belongs to at most one project, so this moves it out of any
    other project. Calling it again for the same project has no effect.
    """
    chat = await queries.add_chat_to_project(
        db_session=db_session, user_id=user.id, chat_uuid=chat_uuid, project=project
    )
    if not chat:
        raise ProjectChatNotFoundError(f"No live chat for this user with chat_uuid: {chat_uuid}")
    return ChatBasicResponse.model_validate(chat)


@router.delete(
    path=ENDPOINTS.USER_PROJECT_CHAT,
    dependencies=PROJECT_DEPENDENCIES,
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_user_project_chat(
    db_session: DbSession,
    chat_uuid: UUID,
    project: Project = Depends(verify_and_get_project_from_path),
) -> Response:
    """
    Remove a chat from a project. The chat stays in the user's chat history.
    """
    if not await queries.remove_chat_from_project(db_session=db_session, chat_uuid=chat_uuid, project=project):
        raise ProjectChatNotFoundError(f"Chat {chat_uuid} is not in project {project.uuid}")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
