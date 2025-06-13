"""Light-weight helpers for running blocking work in a background thread.

FastAPI executes *async* route handlers directly on the event-loop thread.  Any
CPU-bound or I/O-bound library call that performs blocking work (e.g. PDF
parsing with **PyMuPDF**, large embedding generation with
**Sentence-Transformers**) must therefore be off-loaded to a dedicated worker
thread to keep the server responsive.

This module exposes :pyfunc:`run_in_thread` – a tiny convenience wrapper around
``asyncio.get_running_loop().run_in_executor`` so that services can simply

    result = await run_in_thread(func, *args, **kwargs)

instead of repeating boilerplate at every call-site.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, Callable, TypeVar

_T = TypeVar("_T")


async def run_in_thread(func: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:  # noqa: D401 – utility helper
    """Execute *func* in the default executor and ``await`` the result.

    This is a minimal replacement for ``anyio.to_thread.run_sync`` so that we
    do not introduce an additional dependency.  The API is intentionally kept
    *very* small: positional/keyword arguments are forwarded and the return
    value is passed through unchanged.
    """

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))
