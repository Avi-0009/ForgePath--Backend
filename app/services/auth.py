from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password, verify_password
from app.models.user import User
from app.repositories import user as user_repository
from app.schemas.auth import LoginRequest, SignupRequest


class AccountAlreadyExistsError(Exception):
    """An email or username is already registered."""


class InvalidCredentialsError(Exception):
    """The supplied login credentials are invalid."""


async def register_user(
    db: AsyncSession,
    payload: SignupRequest,
) -> User:
    existing_email = await user_repository.get_by_email(
        db,
        str(payload.email),
    )

    if existing_email is not None:
        raise AccountAlreadyExistsError()

    existing_username = await user_repository.get_by_username(
        db,
        payload.username,
    )

    if existing_username is not None:
        raise AccountAlreadyExistsError()

    user = User(
        full_name=payload.full_name,
        username=payload.username,
        email=str(payload.email).lower(),
        password_hash=hash_password(payload.password),
    )

    db.add(user)

    try:
        await db.commit()
        await db.refresh(user)
    except IntegrityError as exc:
        await db.rollback()
        raise AccountAlreadyExistsError() from exc

    return user


async def authenticate_user(
    db: AsyncSession,
    payload: LoginRequest,
) -> User:
    if "@" in payload.identifier:
        user = await user_repository.get_by_email(
            db,
            payload.identifier,
        )
    else:
        user = await user_repository.get_by_username(
            db,
            payload.identifier,
        )

    if user is None:
        raise InvalidCredentialsError()

    if not verify_password(payload.password, user.password_hash):
        raise InvalidCredentialsError()

    if not user.is_active:
        raise InvalidCredentialsError()

    return user