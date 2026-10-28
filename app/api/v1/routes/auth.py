from typing import Annotated
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Response, Cookie
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.core.security import create_access_token
from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import (
    LoginRequest,
    SignupRequest,
    TokenResponse,
    UserResponse,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    EmailVerificationOTPRequest,
)
from app.services.auth import (
    AccountAlreadyExistsError,
    InvalidCredentialsError,
    authenticate_user,
    register_user,
)
from app.repositories import user as user_repository
from app.core.config import settings
from app.services.auth_session import create_auth_session, rotate_auth_session, revoke_auth_session
from app.services.brevo_email import BrevoEmailError
from app.services.password_reset import request_password_reset_otp, reset_password_with_otp
from app.services.email_verification import (
    EmailAlreadyVerifiedError,
    EmailVerificationRateLimitedError,
    EmailVerificationUserError,
    request_email_verification_otp,
    verify_email_with_otp,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/signup",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def signup(
    payload: SignupRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    try:
        return await register_user(db, payload)
    except AccountAlreadyExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email or username already exists",
        ) from exc


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
)
async def login(
    payload: LoginRequest,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    try:
        user = await authenticate_user(db, payload)
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email/username or password",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    auth_session = await create_auth_session(db, user.id)

    access_token = create_access_token(
        user_id=user.id,
        session_id=auth_session.session.id,
    )

    response.set_cookie(
        key="forgepath_refresh",
        value=auth_session.refresh_token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=settings.refresh_token_expire_days * 24 * 60 * 60,
        path="/api/v1/auth",
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user),
    )

@router.post(
    "/logout", 
    status_code=204
)
async def logout(
    response: Response,
    db: AsyncSession = Depends(get_db),
    refresh_token: str | None = Cookie(
        default=None,
        alias="forgepath_refresh",
    ),
) -> None:
    await revoke_auth_session(db, refresh_token)

    # Remove the refresh-token cookie from the browser.
    response.delete_cookie(
        key="forgepath_refresh",
        path="/api/v1/auth",
        httponly=True,
        secure=False,
        samesite="lax",
    )


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
)
async def me(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    return current_user


@router.post(
    "/refresh", 
    response_model=TokenResponse
)
async def refresh_access_token(
    response: Response,
    db: AsyncSession = Depends(get_db),
    refresh_token: str | None = Cookie(
        default=None,
        alias="forgepath_refresh",
    ),
) -> TokenResponse:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not refresh_token:
        raise unauthorized

    rotated = await rotate_auth_session(db, refresh_token)

    if rotated is None:
        raise unauthorized

    user = await user_repository.get_by_id(
        db,
        rotated.session.user_id,
    )

    if user is None or not user.is_active:
        await revoke_auth_session(db, rotated.refresh_token)
        raise unauthorized

    access_token = create_access_token(
        user_id=user.id,
        session_id=rotated.session.id,
    )

    response.set_cookie(
        key="forgepath_refresh",
        value=rotated.refresh_token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=settings.refresh_token_expire_days * 24 * 60 * 60,
        path="/api/v1/auth",
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=user,
    )


@router.post(
    "/forgot-password",
    status_code=status.HTTP_202_ACCEPTED,
)
async def forgot_password(
    payload: ForgotPasswordRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, str]:
    try:
        await request_password_reset_otp(
            db=db,
            email=str(payload.email),
        )
    except BrevoEmailError:
        logger.exception(
            "Brevo failed to send a password-reset email"
        )

    return {
        "message": (
            "If an account exists for that email, "
            "a password reset code will be sent."
        )
    }


@router.post("/reset-password")
async def reset_password(
    payload: ResetPasswordRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, str]:
    success = await reset_password_with_otp(
        db=db,
        email=str(payload.email),
        otp=payload.otp,
        new_password=payload.new_password,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset code",
        )

    return {
        "message": (
            "Password reset successfully. "
            "Please log in with your new password."
        )
    }



@router.post(
    "/email-verification/request",
    status_code=status.HTTP_200_OK,
)
async def request_email_verification(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, str]:
    """Send an OTP to the authenticated user's registered email."""
    try:
        await request_email_verification_otp(
            db=db,
            user_id=current_user.id,
        )

        return {
            "message": "Verification code sent to your email address."
        }

    except EmailAlreadyVerifiedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except EmailVerificationRateLimitedError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
            headers={"Retry-After": "60"},
        ) from exc

    except EmailVerificationUserError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except BrevoEmailError as exc:
        logger.exception("Failed to send an email-verification OTP")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not send the verification email. Please try again later.",
        ) from exc


@router.post(
    "/email-verification/verify",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
)
async def verify_email(
    payload: EmailVerificationOTPRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Verify the authenticated user's email using a six-digit OTP."""
    try:
        success = await verify_email_with_otp(
            db=db,
            user_id=current_user.id,
            otp=payload.otp,
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired verification code.",
            )

        # Reload persisted fields so the response reflects PostgreSQL.
        await db.refresh(current_user)

        return current_user

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Failed to verify the authenticated user's email")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Email verification could not be completed.",
        ) from exc
