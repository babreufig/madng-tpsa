"""cffi ABI binding to ``madng_tpsa`` (the ``mad_*`` engine).

ABI mode (``ffi.dlopen``) against the prebuilt shared object, so nothing is compiled
here. Loading is lazy: importing ``xgtpsa`` never fails, the first call does.

The public header is the engine ABI. Add a declaration there when a new ``mad_*``
function is needed.
TPSA handles stay opaque ``void*``.
(MAD-NG) C conventions: ``setvar`` variable indices start at 1,
and ``mad_tpsa_dflt = 255`` is the full descriptor order.
"""

from __future__ import annotations

from importlib import resources
from typing import Any

import cffi

from .paths import core_library


def _read_cdef() -> str:
    header = resources.files(__package__).joinpath("include", "madng_tpsa.h")
    lines = []
    for line in header.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped in {'extern "C" {', "}"}:
            continue
        lines.append(line)
    return "\n".join(lines)


CDEF = _read_cdef()

_ffi = cffi.FFI()
_ffi.cdef(CDEF)
_lib: Any = None  # the dlopened core; its mad_* members come from the cdef


def lib() -> Any:
    """Dlopen the GTPSA core, keeping one handle for the process.

    The core owns mad's global descriptor state, so consumers that link it (the xtrack
    bridge modules) share the descriptors created here.
    """
    global _lib
    if _lib is None:
        _lib = _ffi.dlopen(core_library())
    return _lib


def ffi() -> cffi.FFI:
    return _ffi
