"""Where the built GTPSA core lives.

The CMake build installs its artifacts into ``xgtpsa`` (``lib/madng_tpsa.*`` +
``include/madng_tpsa.h``), so editable installs and wheels have the same package shape.
The paths are the contract consumers compile against: ``xtrack.tpsa`` builds its bridge with
``-I include_dir()`` and links ``core_library()``.

``XGTPSA_LIB`` (a file or a directory) overrides the packaged location.
"""

from __future__ import annotations

import os
import sysconfig
import warnings

CORE_BASENAME = "madng_tpsa"
_PKG = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.join(_PKG, "lib")
_INCLUDE = os.path.join(_PKG, "include")


def _library_names() -> list[str]:
    suffix = sysconfig.get_config_var("SHLIB_SUFFIX")
    candidates = []
    if suffix:
        candidates.append(f"{CORE_BASENAME}{suffix}")
    candidates.extend([f"{CORE_BASENAME}.so", f"{CORE_BASENAME}.dylib"])
    return list(dict.fromkeys(candidates))


def _base() -> str | None:
    """The directory an env override points at (a file's dirname is fine), or None."""
    p = os.environ.get("XGTPSA_LIB")
    if not p:
        p = os.environ.get("XTRACK_GTPSA_LIB")
        if p:
            warnings.warn(
                "XTRACK_GTPSA_LIB is deprecated; set XGTPSA_LIB (or rely on "
                "the packaged xgtpsa/lib) instead",
                DeprecationWarning,
                stacklevel=3,
            )
    if not p:
        return None
    return p if os.path.isdir(p) else os.path.dirname(os.path.abspath(p))


def _pick(candidates: list[str], markers: list[str], what: str) -> str:
    for d in candidates:
        for marker in markers:
            if os.path.exists(os.path.join(d, marker)):
                return os.path.join(d, marker)
    raise RuntimeError(
        f"{what} not found (looked in {candidates}); build it with pip install -e ."
    )


def core_library() -> str:
    """Absolute path to ``madng_tpsa.so`` or ``madng_tpsa.dylib``."""
    b = _base()
    if b:
        cands = [
            b,
            os.path.join(b, "lib"),
            os.path.join(b, "xgtpsa", "lib"),
            os.path.join(b, "src", "xgtpsa", "lib"),
            os.path.join(b, "build"),
        ]
    else:
        cands = [_LIB]
    return _pick(cands, _library_names(), CORE_BASENAME)


def include_dir() -> str:
    """Directory holding the public xgtpsa header (``madng_tpsa.h``)."""
    b = _base()
    if b:
        cands = [
            os.path.join(b, "include"),
            os.path.join(b, "xgtpsa", "include"),
            os.path.join(b, "src", "xgtpsa", "include"),
            b,
        ]
    else:
        cands = [_INCLUDE]
    return os.path.dirname(_pick(cands, ["madng_tpsa.h"], "madng_tpsa.h"))


def have_core() -> bool:
    """Whether the core is built (for skipping tests)."""
    try:
        core_library()
        return True
    except RuntimeError:
        return False
