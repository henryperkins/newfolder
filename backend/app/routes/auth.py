from fastapi import APIRouter, Depends, HTTPException, Response, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from ..core.database import get_db
from ..core.schemas import (
    UserCreate, UserResponse, LoginRequest, LoginResponse, 
    ForgotPasswordRequest, ResetPasswordRequest, MessageResponse,
    RegistrationAvailableResponse
)
from ..models.user import User
from ..models.password_reset import PasswordResetToken
from ..services.security import SecurityService
from ..services.email import EmailService
from ..dependencies.auth import get_security_service, get_email_service
from ..core.config import settings

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.get("/registration-available", response_model=RegistrationAvailableResponse)
async def check_registration_available(db: Session = Depends(get_db)):
    """Check if registration is open (always available)"""
    return RegistrationAvailableResponse(available=True)


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    security_service: SecurityService = Depends(get_security_service)
):
    """Create a new user account"""
    from ..utils.concurrency import run_in_thread
    
    # Check if email already exists
    if await run_in_thread(lambda: db.query(User).filter(User.email == user_data.email).first()):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Check if username already exists
    if await run_in_thread(lambda: db.query(User).filter(User.username == user_data.username).first()):
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # Create user
    hashed_password = security_service.hash_password(user_data.password)
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password
    )
    
    db.add(db_user)
    await run_in_thread(db.commit)
    await run_in_thread(lambda: db.refresh(db_user))
    
    return db_user


@router.post("/login", response_model=LoginResponse)
async def login(
    response: Response,
    login_data: LoginRequest,
    db: Session = Depends(get_db),
    security_service: SecurityService = Depends(get_security_service)
):
    """Authenticate user and create session"""
    from ..utils.concurrency import run_in_thread

    user = await run_in_thread(lambda: db.query(User).filter(User.email == login_data.email).first())
    
    if not user or not security_service.verify_password(login_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account is disabled")
    
    # Create access token
    token_data = {"sub": str(user.id), "email": user.email}
    access_token = security_service.create_access_token(token_data)
    
    # Update last login
    user.last_login_at = datetime.utcnow()
    await run_in_thread(db.commit)
    
    # Set httpOnly cookie
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=24 * 60 * 60  # 24 hours
    )
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user)
    )


@router.post("/logout", response_model=MessageResponse)
async def logout(response: Response):
    """Invalidate current session"""
    response.delete_cookie("access_token")
    return MessageResponse(message="Successfully logged out")


@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(
    background_tasks: BackgroundTasks,
    request_data: ForgotPasswordRequest,
    db: Session = Depends(get_db),
    security_service: SecurityService = Depends(get_security_service),
    email_service: EmailService = Depends(get_email_service)
):
    """Initiate password reset flow"""
    from ..utils.concurrency import run_in_thread

    user = await run_in_thread(lambda: db.query(User).filter(User.email == request_data.email).first())
    
    if user:
        # Create reset token
        reset_token = security_service.create_password_reset_token(user.email)
        
        # Store token in database
        db_token = PasswordResetToken(
            user_id=user.id,
            token=reset_token,
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        db.add(db_token)
        await run_in_thread(db.commit)
        
        # Send email in background
        background_tasks.add_task(
            email_service.send_password_reset_email,
            user.email,
            reset_token,
            settings.frontend_base_url
        )
    
    # Always return success to prevent email enumeration
    return MessageResponse(message="If the email exists, a reset link has been sent")


@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(
    request_data: ResetPasswordRequest,
    db: Session = Depends(get_db),
    security_service: SecurityService = Depends(get_security_service)
):
    """Complete password reset with token"""
    try:
        # Decode token
        payload = security_service.decode_token(request_data.token)
        if payload.get("type") != "password_reset":
            raise ValueError("Invalid token type")
        
        email = payload.get("email")
        if not email:
            raise ValueError("Invalid token payload")
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    
    # Find user and token
    from ..utils.concurrency import run_in_thread

    user = await run_in_thread(lambda: db.query(User).filter(User.email == email).first())
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    
    token_record = await run_in_thread(
        lambda: db.query(PasswordResetToken).filter(
            PasswordResetToken.token == request_data.token,
            PasswordResetToken.user_id == user.id,
            PasswordResetToken.used == False,
            PasswordResetToken.expires_at > datetime.utcnow(),
        ).first()
    )
    
    if not token_record:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    
    # Update password
    user.hashed_password = security_service.hash_password(request_data.new_password)
    token_record.used = True
    
    await run_in_thread(db.commit)
    
    return MessageResponse(message="Password successfully reset")