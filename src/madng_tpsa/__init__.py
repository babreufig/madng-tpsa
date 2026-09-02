"""madng_tpsa: Python bindings to the GTPSA engine of MAD-NG.

Truncated Power Series Algebra: build a descriptor (how many variables, to which
order, plus optional parameters), seed identity series on it and perform
algebraic operations. The coefficients are the derivatives.

    import madng_tpsa
    d = madng_tpsa.Descriptor(2, 3)          # 2 variables, order 3
    x = d.var(1, 0.5)           # x, expanded around 0.5
    y = d.var(2)
    f = x * x + 2.0 * y
    f.const_part, f.grad(), f.monomial_coeffs()
"""

from ._cffi import CDEF, ffi, lib
from ._version import __version__
from .descriptor import Descriptor
from .errors import TpsaError
from .paths import core_library, include_dir
from .tpsa import Tpsa

__all__ = [
    'CDEF',
    'Descriptor',
    'Tpsa',
    'TpsaError',
    'core_library',
    'ffi',
    'include_dir',
    'lib',
    '__version__',
]
