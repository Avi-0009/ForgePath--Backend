from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password, verify_password
from app.models.user import User
from app.repositories import user as user_repository
from app.schemas.auth import LoginRequest, ProfileUpdateRequest, SignupRequest


class AccountAlreadyExistsError(Exception):
    """An email or username is already registered."""

    def __init__(self, field: str = "account") -> None:
        self.field = field
        super().__init__(f"Duplicate {field}")


class InvalidCredentialsError(Exception):
    """The supplied login credentials are invalid."""


def _integrity_conflict_field(exc: IntegrityError) -> str | None:
    constraint_name = getattr(
        getattr(exc.orig, "diag", None),
        "constraint_name",
        None,
    )
    message = str(exc.orig).casefold()
    if constraint_name == "uq_users_email" or "users.email" in message:
        return "email"
    if constraint_name == "uq_users_username" or "users.username" in message:
        return "username"
    return None


async def register_user(
    db: AsyncSession,
    payload: SignupRequest,
) -> User:
    existing_email = await user_repository.get_by_email(
        db,
        str(payload.email),
    )

    if existing_email is not None:
        raise AccountAlreadyExistsError("email")

    existing_username = await user_repository.get_by_username(
        db,
        payload.username,
    )

    if existing_username is not None:
        raise AccountAlreadyExistsError("username")

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

        conflict_field = _integrity_conflict_field(exc)
        if conflict_field is not None:
            raise AccountAlreadyExistsError(conflict_field) from exc

        raise

    return user


async def update_user_profile(
    db: AsyncSession,
    user: User,
    payload: ProfileUpdateRequest,
) -> User:
    if payload.username is not None and payload.username != user.username:
        existing = await user_repository.get_by_username(db, payload.username)
        if existing is not None and existing.id != user.id:
            raise AccountAlreadyExistsError("username")

    if payload.full_name is not None:
        user.full_name = payload.full_name
    if payload.username is not None:
        user.username = payload.username

    try:
        await db.commit()
        await db.refresh(user)
    except IntegrityError as exc:
        await db.rollback()
        conflict_field = _integrity_conflict_field(exc)
        if conflict_field == "username":
            raise AccountAlreadyExistsError("username") from exc
        raise

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
