from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class ProjectTemplate:
    """Project template data structure"""
    id: str
    name: str
    description: str
    icon: str  # Lucide icon name
    suggested_tags: List[str] = field(default_factory=list)
    starter_prompts: List[str] = field(default_factory=list)
    color: str = "#6366f1"  # Default indigo
    category: str = "general"


class TemplateService:
    """Service for managing project templates"""

    def __init__(self):
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> List[ProjectTemplate]:
        """Initialize hardcoded templates"""
        return [
            ProjectTemplate(
                id="research-project",
                name="Research Project",
                description="Organize research papers, notes, and findings",
                icon="Microscope",
                suggested_tags=["research", "academic", "literature-review"],
                starter_prompts=[
                    "Summarize the key findings from these papers",
                    "What are the main research gaps identified?",
                    "Create a literature review outline"
                ],
                color="#10b981",  # Green
                category="academic"
            ),
            ProjectTemplate(
                id="product-launch",
                name="Product Launch",
                description="Plan and track your product launch activities",
                icon="Rocket",
                suggested_tags=["product", "launch", "marketing", "timeline"],
                starter_prompts=[
                    "Create a launch timeline for the next 3 months",
                    "Draft a press release for our product",
                    "What are the key metrics to track post-launch?"
                ],
                color="#f59e0b",  # Amber
                category="business"
            ),
            ProjectTemplate(
                id="content-creation",
                name="Content Creation",
                description="Manage blog posts, videos, and creative content",
                icon="PenTool",
                suggested_tags=["content", "writing", "creative", "publishing"],
                starter_prompts=[
                    "Generate blog post ideas about AI productivity",
                    "Create an editorial calendar for next month",
                    "Write an engaging introduction for this topic"
                ],
                color="#8b5cf6",  # Purple
                category="creative"
            ),
            ProjectTemplate(
                id="code-development",
                name="Software Development",
                description="Track features, bugs, and code documentation",
                icon="Code2",
                suggested_tags=["development", "coding", "software", "bugs"],
                starter_prompts=[
                    "Review this code for potential improvements",
                    "Generate unit tests for this function",
                    "Explain this error message and suggest fixes"
                ],
                color="#3b82f6",  # Blue
                category="technical"
            ),
            ProjectTemplate(
                id="personal-goals",
                name="Personal Goals",
                description="Track personal development and life goals",
                icon="Target",
                suggested_tags=["personal", "goals", "habits", "growth"],
                starter_prompts=[
                    "Create a 30-day habit tracking plan",
                    "Break down this goal into actionable steps",
                    "Suggest books for professional development"
                ],
                color="#ec4899",  # Pink
                category="personal"
            ),
            ProjectTemplate(
                id="blank-project",
                name="Blank Project",
                description="Start with a clean slate",
                icon="FileText",
                suggested_tags=["general"],
                starter_prompts=[],
                color="#6b7280",  # Gray
                category="general"
            )
        ]

    async def get_all_templates(self) -> List[ProjectTemplate]:
        """Retrieve all available templates"""
        return self.templates

    async def get_template_by_id(self, template_id: str) -> Optional[ProjectTemplate]:
        """Get specific template by ID"""
        return next((t for t in self.templates if t.id == template_id), None)

    async def get_templates_by_category(self, category: str) -> List[ProjectTemplate]:
        """Filter templates by category"""
        return [t for t in self.templates if t.category == category]

    def get_starter_prompts(self, template_id: str) -> List[str]:
        """Get starter prompts for a specific template"""
        template = next((t for t in self.templates if t.id == template_id), None)
        return template.starter_prompts if template else []