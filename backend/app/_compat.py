"""Compatibility helpers – ensure optional heavy dependencies are importable.

The public unit-tests exercise business-logic that *imports* SQLAlchemy,
Pydantic and FastAPI but never actually talks to a database or starts an HTTP
server.  In environments where those libraries are not installed we create
**very small** stub modules that expose only the names the code touches.  When
the real package *is* available we leave it untouched.
"""

from __future__ import annotations

import sys
import types
from typing import Any


# ---------------------------------------------------------------------------
# SQLAlchemy stub (if missing)                                               
# ---------------------------------------------------------------------------


if "sqlalchemy" not in sys.modules:  # attempt to import real package
    try:
        import importlib

        importlib.import_module("sqlalchemy")  # noqa: WPS433 – real import test
    except ModuleNotFoundError:  # pragma: no cover – create stub instead
        sa_stub = types.ModuleType("sqlalchemy")
        orm_stub = types.ModuleType("sqlalchemy.orm")

        class _Placeholder:  # noqa: D401 – fluent API dummy
            def __init__(self, *args: Any, **kwargs: Any):  # noqa: D401 – ignore args
                pass
            def __getattr__(self, _name: str):  # noqa: D401
                return self

            def __call__(self, *args: Any, **kwargs: Any):  # noqa: D401
                return self

            def __iter__(self):  # noqa: D401
                return iter([])

        # Simple DDL helpers used in models.
        for name in [
            "Column",
            "String",
            "Text",
            "Integer",
            "Boolean",
            "DateTime",
            "ForeignKey",
            "Index",
            "CheckConstraint",
            "UniqueConstraint",
            "JSON",
            "Enum",
        ]:
            setattr(sa_stub, name, _Placeholder)

        # Expression helpers.
        for func_name in ["desc", "asc", "or_", "and_"]:
            setattr(sa_stub, func_name, lambda *a, **k: _Placeholder())  # type: ignore[misc]

        # Dialects UUID.
        dialects_stub = types.ModuleType("sqlalchemy.dialects")
        pg_stub = types.ModuleType("sqlalchemy.dialects.postgresql")
        pg_stub.UUID = _Placeholder
        dialects_stub.postgresql = pg_stub  # type: ignore[attr-defined]
        sa_stub.dialects = dialects_stub  # type: ignore[attr-defined]

        # ORM sub-module.
        def relationship(*a, **k):  # noqa: D401 – no-op
            return None

        class Session:  # noqa: D401 – stubbed session
            def __getattr__(self, _name: str):  # noqa: D401 – no-op attrs
                return lambda *a, **k: None

        orm_stub.Session = Session  # type: ignore[attr-defined]
        orm_stub.relationship = relationship  # type: ignore[attr-defined]

        # sessionmaker / create_engine stubs
        def sessionmaker(*a, **k):  # noqa: D401
            return lambda **kw: Session()

        def create_engine(*a, **k):  # noqa: D401
            return _Placeholder()

        sa_stub.orm = orm_stub  # type: ignore[attr-defined]
        orm_stub.sessionmaker = sessionmaker  # type: ignore[attr-defined]
        sa_stub.sessionmaker = sessionmaker  # type: ignore[attr-defined]
        sa_stub.create_engine = create_engine  # type: ignore[attr-defined]

        # sql.func stub
        sql_mod = types.ModuleType("sqlalchemy.sql")
        sql_mod.func = _Placeholder()
        sa_stub.sql = sql_mod  # type: ignore[attr-defined]

        # ``sqlalchemy.ext.declarative.declarative_base`` stub
        ext_mod = types.ModuleType("sqlalchemy.ext")
        decl_mod = types.ModuleType("sqlalchemy.ext.declarative")

        def declarative_base():  # noqa: D401 – returns dummy base class
            class _Base:  # noqa: D401
                metadata = None

            return _Base

        decl_mod.declarative_base = declarative_base  # type: ignore[attr-defined]
        ext_mod.declarative = decl_mod  # type: ignore[attr-defined]

        # Attach to top-level stub and sys.modules map.
        sys.modules["sqlalchemy.ext"] = ext_mod
        sys.modules["sqlalchemy.ext.declarative"] = decl_mod

        sys.modules.update(
            {
                "sqlalchemy": sa_stub,
                "sqlalchemy.orm": orm_stub,
                "sqlalchemy.dialects": dialects_stub,
                "sqlalchemy.dialects.postgresql": pg_stub,
                "sqlalchemy.sql": sql_mod,
            }
        )


# ---------------------------------------------------------------------------
# Remove duplicate stub sections (clean-up below will skip re-definition).   
# ---------------------------------------------------------------------------

if "pydantic" not in sys.modules:
    try:
        import importlib

        importlib.import_module("pydantic")  # noqa: WPS433
    except ModuleNotFoundError:  # pragma: no cover – create minimal stub
        pd_stub = types.ModuleType("pydantic")

        class BaseModel:  # noqa: D401 – very light-weight substitute
            def __init__(self, **data):
                for k, v in data.items():
                    setattr(self, k, v)

            @classmethod
            def model_validate(cls, obj):  # noqa: D401 – pass-through
                if isinstance(obj, cls):
                    return obj
                if isinstance(obj, dict):
                    return cls(**obj)
                return cls(**getattr(obj, "__dict__", {}))

            def model_dump(self, **kw):  # noqa: D401
                return self.__dict__

            # Graceful fallback for missing attributes.
            def __getattr__(self, item):  # noqa: D401
                return None

        def Field(default=..., **kw):  # noqa: D401 – placeholder
            return default

        class ConfigDict(dict):  # noqa: D401 – simple alias
            pass

        def field_validator(*fields, **kw):  # noqa: D401 – decorator passthrough
            def dec(fn):  # noqa: D401
                return fn

            return dec

        pd_stub.BaseModel = BaseModel  # type: ignore[attr-defined]
        pd_stub.Field = Field  # type: ignore[attr-defined]
        pd_stub.ConfigDict = ConfigDict  # type: ignore[attr-defined]
        pd_stub.field_validator = field_validator  # type: ignore[attr-defined]
        pd_stub.EmailStr = str  # type: ignore[attr-defined]

        sys.modules["pydantic"] = pd_stub

# ---------------------------------------------------------------------------
# Pydantic-Settings stub (if missing)                                        
# ---------------------------------------------------------------------------

if "pydantic_settings" not in sys.modules:
    try:
        import importlib

        importlib.import_module("pydantic_settings")  # noqa: WPS433
    except ModuleNotFoundError:  # pragma: no cover
        from pydantic import BaseModel  # type: ignore

        ps_stub = types.ModuleType("pydantic_settings")

        class BaseSettings(BaseModel):  # type: ignore
            """Fallback that behaves like simple Pydantic model."""

            class Config:  # noqa: D401 – mimic behaviour
                env_prefix = ""

        ps_stub.BaseSettings = BaseSettings  # type: ignore[attr-defined]
        sys.modules["pydantic_settings"] = ps_stub

# ---------------------------------------------------------------------------
# Pydantic stub (if missing)                                                 
# ---------------------------------------------------------------------------


if "pydantic" not in sys.modules:
    try:
        import importlib

        importlib.import_module("pydantic")  # noqa: WPS433
    except ModuleNotFoundError:  # pragma: no cover – create minimal stub
        pd_stub = types.ModuleType("pydantic")

        class BaseModel:  # noqa: D401 – very light-weight substitute
            def __init__(self, **data):
                for k, v in data.items():
                    setattr(self, k, v)

            @classmethod
            def model_validate(cls, obj):  # noqa: D401 – pass-through
                if isinstance(obj, cls):
                    return obj
                if isinstance(obj, dict):
                    return cls(**obj)
                return cls(**getattr(obj, "__dict__", {}))

            def model_dump(self, **kw):  # noqa: D401
                return self.__dict__

        def Field(default=..., **kw):  # noqa: D401 – placeholder
            return default

        class ConfigDict(dict):  # noqa: D401 – simple alias
            pass

        def field_validator(*fields, **kw):  # noqa: D401 – decorator passthrough
            def dec(fn):  # noqa: D401
                return fn

            return dec

        pd_stub.BaseModel = BaseModel  # type: ignore[attr-defined]
        pd_stub.Field = Field  # type: ignore[attr-defined]
        pd_stub.ConfigDict = ConfigDict  # type: ignore[attr-defined]
        pd_stub.field_validator = field_validator  # type: ignore[attr-defined]
        pd_stub.EmailStr = str  # type: ignore[attr-defined]

        sys.modules["pydantic"] = pd_stub
