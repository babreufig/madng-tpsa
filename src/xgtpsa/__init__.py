"""xgtpsa: Python bindings to the GTPSA engine of MAD-NG.

Truncated Power Series Algebra: build a descriptor (how many variables, to which
order, plus optional parameters), seed identity series on it and perform
algebraic operations. The coefficients are the derivatives.

    import xgtpsa
    d = xgtpsa.Descriptor(2, 3)          # 2 variables, order 3
    x = d.var(1, 0.5)           # x, expanded around 0.5
    y = d.var(2)
    f = x * x + 2.0 * y
    f.const_part, f.grad(), f.monomial_coeffs()
"""

from ._cffi import CDEF, ffi, lib
from .descriptor import Descriptor
from .paths import core_library, include_dir
from .tpsa import Tpsa

__version__ = '0.1.0'

__all__ = [
    'CDEF',
    'Descriptor',
    'Tpsa',
    'core_library',
    'ffi',
    'include_dir',
    'lib',
    '__version__',
]
