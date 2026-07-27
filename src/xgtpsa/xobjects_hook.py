"""Expose the Xgtpsa library and headers to Xobjects through an entrypoint hook."""

from xgtpsa.paths import LIB_BASENAME, include_dir, lib_dir


def get_build_info():
    """Return build metadata consumed by Xobjects."""
    return {
        "include_dirs": [include_dir()],
        "libraries": [LIB_BASENAME],
        "library_dirs": [lib_dir()],
    }
