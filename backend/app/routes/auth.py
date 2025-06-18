"""Authentication endpoints – async SQLAlchemy version."""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response, BackgroundTasks
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_db
from ..core.schemas import (
    UserCreate,
    UserResponse,
    LoginRequest,
    LoginResponse,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    MessageResponse,
    RegistrationAvailableResponse,
)
from ..models.user import User
from ..models.password_reset import PasswordResetToken
from ..services.security import SecurityService
from ..services.email import EmailService
from ..dependencies.auth import get_security_service, get_email_service

from ..core.config import settings


router = APIRouter(prefix="/auth", tags=["authentication"])


# ---------------------------------------------------------------------------
# Utility helpers                                                            
# ---------------------------------------------------------------------------


async def _email_exists(db: AsyncSession, email: str) -> bool:
    stmt = select(User.id).where(User.email == email)
    return (await db.scalar(stmt)) is not None


async def _username_exists(db: AsyncSession, username: str) -> bool:
    stmt = select(User.id).where(User.username == username)
    return (await db.scalar(stmt)) is not None


# ---------------------------------------------------------------------------
# Routes                                                                     
# ---------------------------------------------------------------------------


@router.get("/registration-available", response_model=RegistrationAvailableResponse)
async def check_registration_available() -> RegistrationAvailableResponse:  # noqa: D401
    """Public helper for the frontend; always returns true for now."""

    return RegistrationAvailableResponse(available=True)


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
    security_service: SecurityService = Depends(get_security_service),
) -> UserResponse:
    """Create a new user after validating uniqueness constraints."""

    if await _email_exists(db, user_data.email):
        raise HTTPException(status_code=400, detail="Email already registered")

    if await _username_exists(db, user_data.username):
        raise HTTPException(status_code=400, detail="Username already taken")

    db_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=security_service.hash_password(user_data.password),
    )

    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)

    return db_user


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    response: Response,
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_db),
    security_service: SecurityService = Depends(get_security_service),
) -> LoginResponse:
    """Authenticate a user and set the cookie-based access token."""

    stmt_user = select(User).where(User.email == login_data.email)
    user = await db.scalar(stmt_user)

    if not user or not security_service.verify_password(login_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account is disabled")

    # Token ---------------------------------------------------------------
    token_data = {"sub": str(user.id), "email": user.email}
    access_token = security_service.create_access_token(token_data)

    # Update last login ---------------------------------------------------
    user.last_login_at = datetime.utcnow()
    await db.commit()

    # Set httpOnly cookie --------------------------------------------------
    scheme = request.url.scheme
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=(scheme == "https"),
        samesite="lax",
        max_age=24 * 60 * 60,
    )

    return LoginResponse(access_token=access_token, token_type="bearer", user=UserResponse.model_validate(user))


@router.post("/logout", response_model=MessageResponse)
async def logout(response: Response) -> MessageResponse:  # noqa: D401
    """Clear the auth cookie."""

    response.delete_cookie("access_token")
    return MessageResponse(message="Successfully logged out")


@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(
    background_tasks: BackgroundTasks,
    request_data: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
    security_service: SecurityService = Depends(get_security_service),
    email_service: EmailService = Depends(get_email_service),
) -> MessageResponse:
    """Generate a password reset token and send the email (if the user exists)."""

    stmt_user = select(User).where(User.email == request_data.email)
    user = await db.scalar(stmt_user)

    if user:
        reset_token = security_service.create_password_reset_token(user.email)

        token_record = PasswordResetToken(
            user_id=user.id,
            token=reset_token,
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        db.add(token_record)
        await db.commit()

        background_tasks.add_task(
            email_service.send_password_reset_email,
            user.email,
            reset_token,
            settings.frontend_base_url,
        )

    # Always succeed to avoid user enumeration -----------------------------
    return MessageResponse(message="If the email exists, a reset link has been sent")


@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(
    request_data: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
    security_service: SecurityService = Depends(get_security_service),
) -> MessageResponse:
    """Verify a password reset token and set the new password."""

    try:
        payload = security_service.decode_token(request_data.token)
        if payload.get("type") != "password_reset":
            raise ValueError
        email = payload.get("email")
        if not email:
            raise ValueError
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    stmt_user = select(User).where(User.email == email)
    user = await db.scalar(stmt_user)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    # Token record ---------------------------------------------------------
    stmt_token = (
        select(PasswordResetToken)
        .where(
            PasswordResetToken.token == request_data.token,
            PasswordResetToken.user_id == user.id,
            PasswordResetToken.used.is_(False),
            PasswordResetToken.expires_at > datetime.utcnow(),
        )
    )
    token_record = await db.scalar(stmt_token)
    if not token_record:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    # Update password ------------------------------------------------------
    user.hashed_password = security_service.hash_password(request_data.new_password)
    token_record.used = True
    await db.commit()

    return MessageResponse(message="Password successfully reset")
