from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..core.config import settings
from ..models.user import User
from ..services.security import SecurityService
from ..services.email import EmailService


def get_security_service() -> SecurityService:
    return SecurityService(settings.secret_key, settings.algorithm)


def get_email_service() -> EmailService:
    return EmailService(
        settings.smtp_host,
        settings.smtp_port,
        settings.smtp_username,
        settings.smtp_password
    )


async def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    security_service: SecurityService = Depends(get_security_service)
) -> User:
    """Extract and validate user from JWT token"""
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = security_service.decode_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    return user