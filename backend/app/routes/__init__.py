from .auth import router as auth_router
from .users import router as users_router
from .projects import router as projects_router
from .templates import router as templates_router
from .tags import router as tags_router
from .activities import router as activities_router
from .documents import router as documents_router
from .search import router as search_router

__all__ = [
    "auth_router",
    "users_router",
    "projects_router",
    "templates_router",
    "tags_router",
    "activities_router",
    "documents_router",
    "search_router",
]
