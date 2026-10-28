from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import AuthSession, User


async def create_auth_session(
    db_session: AsyncSession,
    user: User,
) -> AuthSession:
    """
    Creates a new authentication session for a user with their profile information.

    Args:
        db_session (AsyncSession): The database session for executing queries.
        user (User): The user object containing profile information including:
            - id: User's unique identifier
            - job_title: User's job title
            - region: User's region
            - sector: User's sector
            - organisation: User's organisation
            - grade: User's grade
            - communicator_role: User's communicator role status

    Returns:
        AuthSession: The newly created authentication session object.

    Raises:
        Exception: If there's an error during execution, the original exception will be logged and re-raised.
    """
    stmt = (
        insert(AuthSession)
        .values(
            user_id=user.id,
            job_title=user.job_title,
            region=user.region,
            sector=user.sector,
            organisation=user.organisation,
            grade=user.grade,
            communicator_role=user.communicator_role,
        )
        .returning(AuthSession)
    )
    auth_session = await db_session.execute(stmt)
    return auth_session.scalars().first()
