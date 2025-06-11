"""Package initialisation – registers compatibility shims & public aliases."""

# Ensure optional heavy dependencies are stubbed when missing so that the
# code base remains importable in minimal execution environments (CI sandboxes
# without database/fastapi packages).

from . import _compat  # noqa: F401  – side-effect only

# Expose internal *services* package under the short name that appears in the
# Phase-3 specification so that imports like ``from services.chat_service
# import ChatService`` keep working without adjusting project layout.

import sys as _sys

# Resolve package and then register alias.

import importlib as _importlib

_services_pkg = "backend.app.services"

try:
    _services_ref = _importlib.import_module(_services_pkg)
    _sys.modules.setdefault("services", _services_ref)
except ModuleNotFoundError:  # pragma: no cover – happens only in minimal builds
    pass
