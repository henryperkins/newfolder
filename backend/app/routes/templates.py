from fastapi import APIRouter, Depends, HTTPException
from typing import Optional

from ..dependencies.auth import get_current_user
from ..models import User
from ..core.schemas import ProjectTemplateResponse, TemplateListResponse
from ..services.template_service import TemplateService


router = APIRouter(prefix="/project-templates", tags=["templates"])


@router.get("", response_model=TemplateListResponse)
async def get_project_templates(
    category: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get available project templates"""
    template_service = TemplateService()
    
    if category:
        templates = await template_service.get_templates_by_category(category)
    else:
        templates = await template_service.get_all_templates()

    template_responses = [
        ProjectTemplateResponse(
            id=template.id,
            name=template.name,
            description=template.description,
            icon=template.icon,
            suggested_tags=template.suggested_tags,
            starter_prompts=template.starter_prompts,
            color=template.color,
            category=template.category
        )
        for template in templates
    ]

    return TemplateListResponse(templates=template_responses)


@router.get("/{template_id}", response_model=ProjectTemplateResponse)
async def get_project_template(
    template_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get specific template details"""
    template_service = TemplateService()
    template = await template_service.get_template_by_id(template_id)
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    return ProjectTemplateResponse(
        id=template.id,
        name=template.name,
        description=template.description,
        icon=template.icon,
        suggested_tags=template.suggested_tags,
        starter_prompts=template.starter_prompts,
        color=template.color,
        category=template.category
    )