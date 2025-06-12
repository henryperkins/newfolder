from .user import User
from .password_reset import PasswordResetToken
from .project import Project
from .tag import Tag
from .project_tag import ProjectTag
# Existing exports
from .activity import ActivityLog, ActivityType
# Phase-4 document models ----------------------------------------------------
from .document import Document, DocumentVersion, ChatDocumentReference  # noqa: WPS433 – re-export

# Phase-3 chat models -------------------------------------------------------
from .chat import ChatThread, ChatMessage, ChatSummary  # noqa: WPS433 – re-export

__all__ = [
    "User",
    "PasswordResetToken",
    "Project",
    "Tag",
    "ProjectTag",
    "ActivityLog",
    "ActivityType",
    "ChatThread",
    "ChatMessage",
    "ChatSummary",
    "Document",
    "DocumentVersion",
    "ChatDocumentReference",
]

# ---------------------------------------------------------------------------
# Import alias so that external code can simply ``import models.chat``    
# ---------------------------------------------------------------------------

import sys as _sys
import types as _types

# Register the current package under the short name "models" when it is not
# already present.  This mirrors the convenience alias created for
# ``services`` inside :pymod:`backend.app.services.ai_provider` so that the
# public import paths used throughout the Phase-3 specification work without
# restructuring the repository.

if "models" not in _sys.modules:
    _sys.modules["models"] = _types.ModuleType("models")

# Expose the sub-module so that ``import models.chat`` succeeds.
_sys.modules.setdefault("models.chat", _sys.modules[__name__ + ".chat"])