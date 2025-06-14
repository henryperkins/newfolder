from pydantic import BaseModel, EmailStr, Field, ConfigDict, field_validator
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid
import re


# User schemas
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=30)
    email: EmailStr


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)


class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=30)
    email: Optional[EmailStr] = None


class UserResponse(UserBase):
    id: uuid.UUID
    is_active: bool
    created_at: datetime
    last_login_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# Auth schemas
class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)


class MessageResponse(BaseModel):
    message: str


class RegistrationAvailableResponse(BaseModel):
    available: bool
    message: Optional[str] = None


# Tag schemas
class TagBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=30)
    color: Optional[str] = Field(None, pattern="^#[0-9A-Fa-f]{6}$")


class TagCreate(TagBase):
    user_id: Optional[uuid.UUID] = None


class TagResponse(TagBase):
    id: uuid.UUID
    usage_count: Optional[int] = 0
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Project schemas
class ProjectBase(BaseModel):
    name: str = Field(..., min_length=3, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    color: str = Field(..., pattern="^#[0-9A-Fa-f]{6}$")
    template_id: Optional[str] = Field(None, max_length=50)

    @field_validator('name')
    @classmethod
    def name_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()


class ProjectCreate(ProjectBase):
    tags: List[str] = Field(default_factory=list, max_items=10)


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=3, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    color: Optional[str] = Field(None, pattern="^#[0-9A-Fa-f]{6}$")
    tags: Optional[List[str]] = Field(None, max_items=10)
    is_archived: Optional[bool] = None


class ProjectStats(BaseModel):
    chat_count: int = 0
    document_count: int = 0


class ProjectResponse(ProjectBase):
    id: uuid.UUID
    is_archived: bool
    tags: List[TagResponse] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime
    stats: Optional[ProjectStats] = None

    model_config = ConfigDict(from_attributes=True)


class ProjectListResponse(BaseModel):
    projects: List[ProjectResponse]
    total: int


# Template schemas
class ProjectTemplateResponse(BaseModel):
    id: str
    name: str
    description: str
    icon: str
    suggested_tags: List[str]
    starter_prompts: List[str]
    color: str
    category: str


class TemplateListResponse(BaseModel):
    templates: List[ProjectTemplateResponse]


# Activity schemas
class ActivityResponse(BaseModel):
    id: uuid.UUID
    activity_type: str
    project_id: Optional[uuid.UUID]
    project_name: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ActivityListResponse(BaseModel):
    activities: List[ActivityResponse]
    total: int
    has_more: bool


class ActivitySummaryResponse(BaseModel):
    total_activities: int
    projects_active: int
    most_active_project: Optional[Dict[str, Any]]
    activity_by_type: Dict[str, int]
    daily_breakdown: Dict[str, int]


class TagListResponse(BaseModel):
    tags: List[TagResponse]