"""Shallow compatibility wrapper for ``backend.app.services``.

Some unit-tests import modules via the shorter ``services.*`` path even though
the *actual* source code lives under ``backend.app.services``.  To avoid
duplicating all re-exports we dynamically forward attribute access to the real
package and register a few convenience aliases in ``sys.modules``.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


# ---------------------------------------------------------------------------
# Helper – import backend package lazily so that heavy third-party
# dependencies (e.g. SQLAlchemy) are only pulled in when truly needed.
# ---------------------------------------------------------------------------


_BACKEND_PKG = "backend.app.services"


def _backend() -> ModuleType:  # noqa: D401 – local helper
    if _BACKEND_PKG in sys.modules:
        return sys.modules[_BACKEND_PKG]
    return importlib.import_module(_BACKEND_PKG)


# ---------------------------------------------------------------------------
# sys.modules aliases so that ``import services.ai_provider`` works out of the
# box without having to import the whole backend package.
# ---------------------------------------------------------------------------


for _sub in ("ai_provider", "websocket_manager", "summarization_service"):
    fq_name = f"services.{_sub}"
    target = f"{_BACKEND_PKG}.{_sub}"
    try:
        sys.modules[fq_name] = importlib.import_module(target)
    except ModuleNotFoundError:
        # Optional component (e.g. summarization_service when SQLAlchemy is
        # unavailable).  We simply skip the alias.
        pass


# ---------------------------------------------------------------------------
# Attribute forwarding – ``services.AIProvider`` etc.
# ---------------------------------------------------------------------------


def __getattr__(name: str):  # noqa: D401 – PEP-562 support
    backend = _backend()
    try:
        return getattr(backend, name)
    except AttributeError as exc:
        raise AttributeError(name) from exc


# Re-use backend ``__all__`` if present so that ``from services import *``
# behaves consistently.

try:
    __all__ = list(getattr(_backend(), "__all__", []))  # type: ignore[invalid-name]
except Exception:  # pragma: no cover – backend may fail to import
    __all__ = []  # type: ignore[invalid-name]
