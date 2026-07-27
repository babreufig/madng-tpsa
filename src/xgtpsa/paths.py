"""Where the built GTPSA core lives.

The CMake build installs its artifacts into ``xgtpsa`` (``lib/libmadng_tpsa.*`` +
``include/mad_tpsa.hpp``), so editable installs and wheels have the same package shape.
The paths are the contract consumers compile against: ``xtrack.tpsa`` builds its
bridge with ``-I include_dir()`` and links ``core_library()``.

``XGTPSA_LIB`` (a file or a directory) overrides the packaged location.
"""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

CORE_BASENAME = "madng_tpsa"
_PKG = Path(__file__).resolve().parent
_LIB = _PKG / "lib"
_INCLUDE = _PKG / "include"


def _library_names() -> list[str]:
    suffix = sysconfig.get_config_var("SHLIB_SUFFIX")
    candidates = []
    if suffix:
        candidates.append(f"lib{CORE_BASENAME}{suffix}")
    candidates.extend(
        [
            f"lib{CORE_BASENAME}.so",
            f"lib{CORE_BASENAME}.dylib",
        ]
    )
    return list(dict.fromkeys(candidates))


def _base() -> str | None:
    """The directory an env override points at (a file's dirname is fine), or None."""
    p = os.environ.get("XGTPSA_LIB")
    if not p:
        return None
    path = Path(p)
    return str(path if path.is_dir() else path.resolve().parent)


def _pick(candidates: Sequence[Path | str], markers: Sequence[str], what: str) -> str:
    for d in candidates:
        for marker in markers:
            path = Path(d) / marker
            if path.exists():
                return str(path)
    raise RuntimeError(f"{what} not found (looked in {candidates}); build it with pip install -e .")


def core_library() -> str:
    """Absolute path to ``libmadng_tpsa.so`` or ``libmadng_tpsa.dylib``."""
    b = _base()
    if b:
        base = Path(b)
        cands = [
            base,
            base / "lib",
            base / "xgtpsa" / "lib",
            base / "src" / "xgtpsa" / "lib",
            base / "build",
        ]
    else:
        cands = [_LIB]
    return _pick(cands, _library_names(), CORE_BASENAME)


def include_dir() -> str:
    """Directory holding the public MAD-NG GTPSA headers."""
    b = _base()
    if b:
        base = Path(b)
        cands = [
            base / "include",
            base / "xgtpsa" / "include",
            base / "src" / "xgtpsa" / "include",
            base,
        ]
    else:
        cands = [_INCLUDE]
    return str(Path(_pick(cands, ["mad_tpsa.hpp"], "mad_tpsa.hpp")).parent)


def have_core() -> bool:
    """Whether the core is built (for skipping tests)."""
    try:
        core_library()
        return True
    except RuntimeError:
        return False
