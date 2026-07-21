"""xgtpsa: Python bindings to the GTPSA engine of MAD-NG.

Truncated Power Series Algebra: build a descriptor (how many variables, to which
order, plus optional parameters), seed identity series on it and perform
algebraic operations. The coefficients are the derivatives.

    import xgtpsa
    d = xgtpsa.Descriptor.new(2, 3)          # 2 variables, order 3
    x = xgtpsa.Tpsa.var(d, 1, 0.5)           # x, expanded around 0.5
    y = xgtpsa.Tpsa.var(d, 2)
    f = x * x + 2.0 * y
    f.const_part, f.grad(), f.monomial_coeffs()

This package can be used standalone: it needs only the built core (``build.sh``) and knows
nothing about tracking.
``xtrack.tpsa`` is a consumer, compiling its element bridge
against ``include_dir()`` and linking ``core_library()``.
"""

from ._cffi import CDEF, ffi, lib
from .descriptor import Descriptor, live_descriptors
from .paths import core_library, have_core, include_dir
from .tpsa import Tpsa

__version__ = "0.1.0"

__all__ = [
    "CDEF",
    "Descriptor",
    "Tpsa",
    "core_library",
    "ffi",
    "have_core",
    "include_dir",
    "lib",
    "live_descriptors",
    "__version__",
]
