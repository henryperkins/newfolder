"""User account management routes – async version."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_db
from ..core.schemas import (
    UserResponse,
    UserUpdate,
    ChangePasswordRequest,
    MessageResponse,
)
from ..models.user import User
from ..services.security import SecurityService
from ..dependencies.auth import get_current_user, get_security_service


router = APIRouter(prefix="/users", tags=["users"])


# ---------------------------------------------------------------------------
# Helpers                                                                    
# ---------------------------------------------------------------------------


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user: User = Depends(get_current_user)) -> User:
    """Return the authenticated user's own profile."""

    return current_user


@router.patch("/me", response_model=UserResponse)
async def update_user_profile(
    user_update: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> User:
    """Update editable profile fields (email, username, etc.)."""

    update_data = user_update.model_dump(exclude_unset=True)

    # Conflict checks -------------------------------------------------------
    if "email" in update_data:
        stmt_email = select(User).where(User.email == update_data["email"], User.id != current_user.id)
        if await db.scalar(stmt_email):
            raise HTTPException(status_code=400, detail="Email already registered")

    if "username" in update_data:
        stmt_username = select(User).where(
            User.username == update_data["username"],
            User.id != current_user.id,
        )
        if await db.scalar(stmt_username):
            raise HTTPException(status_code=400, detail="Username already taken")

    # Apply updates ---------------------------------------------------------
    for field, value in update_data.items():
        setattr(current_user, field, value)

    await db.commit()
    await db.refresh(current_user)

    return current_user


@router.post("/me/change-password", response_model=MessageResponse)
async def change_password(
    password_data: ChangePasswordRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    security_service: SecurityService = Depends(get_security_service),
) -> MessageResponse:
    """Change the account's password after verifying the current one."""

    if not security_service.verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    current_user.hashed_password = security_service.hash_password(password_data.new_password)
    await db.commit()

    return MessageResponse(message="Password successfully changed")
