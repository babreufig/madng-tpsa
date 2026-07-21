"""Where the built GTPSA core lives.

``build.sh`` installs its artifacts into ``xgtpsa/_core`` (``lib/libgtpsa_core.so`` +
``include/mad_*.h``), so a dev checkout on PYTHONPATH is self-describing.
The paths are the contract consumers compile against: ``xtrack.tpsa`` builds its bridge with
``-I include_dir()`` and links ``core_library()``.

``XGTPSA_LIB`` (a file or a directory) overrides the packaged location.
"""

from __future__ import annotations

import os
import warnings

CORE_SO = "libgtpsa_core.so"
_PKG = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(_PKG, "_core")


def _base() -> str | None:
    """The directory an env override points at (a file's dirname is fine), or None."""
    p = os.environ.get("XGTPSA_LIB")
    if not p:
        p = os.environ.get("XTRACK_GTPSA_LIB")
        if p:
            warnings.warn(
                "XTRACK_GTPSA_LIB is deprecated; set XGTPSA_LIB (or rely on the "
                "packaged xgtpsa/_core) instead",
                DeprecationWarning,
                stacklevel=3,
            )
    if not p:
        return None
    return p if os.path.isdir(p) else os.path.dirname(os.path.abspath(p))


def _pick(candidates: list[str], marker: str, what: str) -> str:
    for d in candidates:
        if os.path.exists(os.path.join(d, marker)):
            return d
    raise RuntimeError(
        f"{what} not found (looked in {candidates}); build it with gtpsa_lib/build.sh"
    )


def core_library() -> str:
    """Absolute path to ``libgtpsa_core.so``."""
    b = _base()
    # An override may name the installed _core, the repo root, or a raw build dir.
    cands = (
        [b, os.path.join(b, "xgtpsa", "_core", "lib"), os.path.join(b, "build")]
        if b
        else [os.path.join(_CORE, "lib")]
    )
    return os.path.join(_pick(cands, CORE_SO, CORE_SO), CORE_SO)


def include_dir() -> str:
    """Directory holding the MAD-NG headers (``mad_tpsa.hpp``, ``mad_*.h``)."""
    b = _base()
    cands = (
        [
            os.path.join(b, "include"),
            os.path.join(b, "xgtpsa", "_core", "include"),
            os.path.join(b, "build"),
            b,
        ]
        if b
        else [os.path.join(_CORE, "include")]
    )
    return _pick(cands, "mad_tpsa.hpp", "mad_tpsa.hpp")


def have_core() -> bool:
    """Whether the core is built (for skipping tests)."""
    try:
        core_library()
        return True
    except RuntimeError:
        return False
