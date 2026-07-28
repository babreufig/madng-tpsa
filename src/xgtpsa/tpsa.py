"""Truncated power series objects."""

from __future__ import annotations

from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.special

from . import _cffi
from ._cffi import ffi, lib

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from .descriptor import Descriptor


class Tpsa:
    """A truncated power series in the algebraic space defined by a descriptor."""

    __array_priority__ = 1000
    __slots__ = ('_ptr',)  # pointer to the C tpsa_t* (the series itself)

    def __init__(self, descriptor: Descriptor) -> None:
        """Create a zero series on ``descriptor``."""
        self._ptr = lib().mad_tpsa_newd(descriptor.ptr, lib().mad_tpsa_dflt)

    @classmethod
    def from_ptr(cls, ptr: Any) -> Tpsa:
        """Wrap an existing low-level TPSA pointer."""
        t = cls.__new__(cls)
        t._ptr = ptr
        return t

    def __del__(self) -> None:
        # The underlying object, or the library might have already been released,
        # if not release the handle now.
        if getattr(self, '_ptr', None) is not None and _cffi._lib is not None:
            _cffi._lib.mad_tpsa_del(self._ptr)
            self._ptr = None

    @property
    def ptr(self) -> Any:
        """Low-level TPSA pointer for consumers marshalling across an ABI."""
        return self._ptr

    @property
    def descriptor(self) -> Descriptor:
        """Descriptor that defines this series' variables, parameters, and order."""
        from .descriptor import Descriptor

        return Descriptor.from_ptr(lib().mad_tpsa_desc(self._ptr))

    @property
    def order(self) -> int:
        """Maximum order stored by this series."""
        return lib().mad_tpsa_ord(self._ptr, False)  # noqa: FBT003

    @property
    def const_part(self) -> float:
        """Constant coefficient of the series."""
        return lib().mad_tpsa_geti(self._ptr, 0)

    def get(self, monomial: Iterable[int]) -> float:
        """Return the coefficient for ``monomial``."""
        monomial_orders = list(monomial)
        monomial_arr = ffi().new('unsigned char[]', monomial_orders)
        return lib().mad_tpsa_getm(self._ptr, len(monomial_orders), monomial_arr)

    def set_const_part(self, v: float) -> None:
        """Set the constant coefficient."""
        lib().mad_tpsa_seti(self._ptr, 0, 0.0, float(v))

    def set(self, monomial: Iterable[int], value: float) -> None:
        """Set the coefficient for ``monomial``."""
        monomial_orders = list(monomial)
        monomial_arr = ffi().new('unsigned char[]', monomial_orders)
        lib().mad_tpsa_setm(self._ptr, len(monomial_orders), monomial_arr, 0.0, float(value))

    def coefficient(
        self, monomials: Sequence[int] | Sequence[Sequence[int]] | np.ndarray
    ) -> float | np.ndarray:
        """Return coefficients for one monomial or a batch of monomials.

        A monomial gives the exponent for each variable and parameter in the
        descriptor. A one-dimensional input returns one float. A two-dimensional
        input returns a NumPy array with one coefficient per row.
        """
        monomial_arr = np.asarray(monomials, dtype=int)
        if monomial_arr.ndim == 1:
            return self.get(tuple(monomial_arr))
        if monomial_arr.ndim == 2:
            return np.array([self.get(tuple(row)) for row in monomial_arr])
        raise ValueError('monomials must be one monomial or a two-dimensional batch')

    def monomial_coeffs(self, tol: float = 1e-14) -> dict[tuple[int, ...], float]:
        """Return stored coefficients larger than ``tol`` in absolute value.

        Keys are full monomial tuples with one entry per variable and parameter.
        """
        monomial_len = self.descriptor.monomial_length
        monomial_arr = ffi().new('unsigned char[]', monomial_len)
        coeff_ptr = ffi().new('double*')
        coeffs: dict[tuple[int, ...], float] = {}

        i = -1
        while (i := lib().mad_tpsa_cycle(self._ptr, i, monomial_len, monomial_arr, coeff_ptr)) >= 0:
            coefficient = coeff_ptr[0]
            if abs(coefficient) <= tol:
                continue
            monomial = tuple(monomial_arr)
            coeffs[monomial] = coefficient

        return coeffs

    def to_dict(self, tol: float = 1e-14) -> dict[tuple[int, ...], float]:
        """Return this series as a monomial-to-coefficient dictionary."""
        return self.monomial_coeffs(tol=tol)

    def from_dict(self, coefficients: Mapping[tuple[int, ...], float]) -> None:
        """Replace this series with coefficients from ``coefficients``.

        Keys are full monomial exponent tuples with one entry per variable and
        parameter. Existing coefficients are cleared before the new ones are set.
        """
        lib().mad_tpsa_clear(self._ptr)
        for monomial, coefficient in coefficients.items():
            self.set(monomial, coefficient)

    def grad(self) -> list[float]:
        """First-order coefficients for the descriptor variables."""
        num_vars = self.descriptor.num_vars
        grad = []
        for var_idx in range(num_vars):
            monomial = [0] * num_vars
            monomial[var_idx] = 1
            grad.append(self.get(monomial))
        return grad

    def param_grad(self) -> list[float]:
        """First-order coefficients for the descriptor parameters."""
        attrs = self.descriptor._get_descriptor_attrs()
        monomial_len = attrs.num_vars + attrs.num_params
        param_grad = []
        for param_idx in range(attrs.num_params):
            monomial = [0] * monomial_len
            monomial[attrs.num_vars + param_idx] = 1
            param_grad.append(self.get(monomial))
        return param_grad

    def copy(self) -> Tpsa:
        """Return an independent copy of this series."""
        result = self.descriptor.zero()
        lib().mad_tpsa_copy(self._ptr, result._ptr)
        return result

    def __repr__(self):
        return f'Tpsa({self.to_dict()!r})'

    # --- arithmetic (fresh result on the same descriptor; scalars mix freely) --- #

    def _binop(self, other: Tpsa, fn: str) -> Tpsa:
        if not lib().xgtpsa_check_tpsa_compatibility(self._ptr, other._ptr):
            raise ValueError('Incompatible TPSA descriptors')

        result = self.descriptor.zero()
        getattr(lib(), fn)(self._ptr, other._ptr, result._ptr)
        return result

    def _coerce_operand(self, other: Tpsa | float) -> Tpsa:
        if isinstance(other, Tpsa):
            if not lib().xgtpsa_check_tpsa_compatibility(self._ptr, other._ptr):
                raise ValueError('Incompatible TPSA descriptors')
            return other
        return self.descriptor.constant(float(other))

    def _unary_op(self, fn: str) -> Tpsa:
        result = self.descriptor.zero()
        getattr(lib(), fn)(self._ptr, result._ptr)
        return result

    def _pair_op(self, fn: str) -> tuple[Tpsa, Tpsa]:
        first = self.descriptor.zero()
        second = self.descriptor.zero()
        getattr(lib(), fn)(self._ptr, first._ptr, second._ptr)
        return first, second

    def equals(self, other: Tpsa, tol: float = 0.0) -> bool:
        """Return whether this series and ``other`` have matching coefficients."""
        if not lib().xgtpsa_check_tpsa_compatibility(self._ptr, other._ptr):
            return False
        return bool(lib().mad_tpsa_equ(self._ptr, other._ptr, float(tol)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tpsa):
            return False
        return self.equals(other)

    def __add__(self, other: Tpsa | float) -> Tpsa:
        if isinstance(other, Tpsa):
            return self._binop(other, 'mad_tpsa_add')
        result = self.descriptor.zero()
        lib().mad_tpsa_axpb(1.0, self._ptr, float(other), result._ptr)
        return result

    __radd__ = __add__

    def __sub__(self, other: Tpsa | float) -> Tpsa:
        if isinstance(other, Tpsa):
            return self._binop(other, 'mad_tpsa_sub')
        return self.__add__(-float(other))

    def __rsub__(self, other: float) -> Tpsa:
        result = self.descriptor.zero()
        lib().mad_tpsa_axpb(-1.0, self._ptr, float(other), result._ptr)
        return result

    def __mul__(self, other: Tpsa | float) -> Tpsa:
        if isinstance(other, Tpsa):
            return self._binop(other, 'mad_tpsa_mul')
        result = self.descriptor.zero()
        lib().mad_tpsa_scl(self._ptr, float(other), result._ptr)
        return result

    __rmul__ = __mul__

    def __truediv__(self, other: Tpsa | float) -> Tpsa:
        if isinstance(other, Tpsa):
            return self._binop(other, 'mad_tpsa_div')
        result = self.descriptor.zero()
        lib().mad_tpsa_divn(self._ptr, float(other), result._ptr)
        return result

    def __rtruediv__(self, other: float) -> Tpsa:
        result = self.descriptor.zero()
        lib().mad_tpsa_inv(self._ptr, float(other), result._ptr)
        return result

    def __pow__(self, other: Tpsa | int | float) -> Tpsa:
        result = self.descriptor.zero()
        if isinstance(other, Tpsa):
            if not lib().xgtpsa_check_tpsa_compatibility(self._ptr, other._ptr):
                raise ValueError('Incompatible TPSA descriptors')
            lib().mad_tpsa_pow(self._ptr, other._ptr, result._ptr)
        elif isinstance(other, Integral):
            lib().mad_tpsa_powi(self._ptr, int(other), result._ptr)
        elif isinstance(other, Real):
            lib().mad_tpsa_pown(self._ptr, float(other), result._ptr)
        else:
            return NotImplemented
        return result

    def __rpow__(self, other: float) -> Tpsa:
        result = self * float(np.log(other))
        lib().mad_tpsa_exp(result._ptr, result._ptr)
        return result

    def __neg__(self) -> Tpsa:
        return self * -1.0

    def __pos__(self) -> Tpsa:
        return self.copy()

    def __abs__(self) -> Tpsa:
        return self.abs()

    def abs(self) -> Tpsa:
        """Return the TPSA absolute value determined by the constant coefficient."""
        return self._unary_op('mad_tpsa_abs')

    def norm(self) -> float:
        """Return the sum of absolute values of the stored coefficients."""
        return lib().mad_tpsa_nrm(self._ptr)

    def unit(self) -> Tpsa:
        """Return this series divided by the magnitude of its constant coefficient."""
        if self.const_part == 0.0:
            raise ZeroDivisionError('Cannot normalize a TPSA with zero constant part')
        return self._unary_op('mad_tpsa_unit')

    def sqrt(self) -> Tpsa:
        """Return the square root of this series."""
        return self._unary_op('mad_tpsa_sqrt')

    def exp(self) -> Tpsa:
        """Return the exponential of this series."""
        return self._unary_op('mad_tpsa_exp')

    def log(self) -> Tpsa:
        """Return the natural logarithm of this series."""
        return self._unary_op('mad_tpsa_log')

    def sin(self) -> Tpsa:
        """Return the sine of this series."""
        return self._unary_op('mad_tpsa_sin')

    def cos(self) -> Tpsa:
        """Return the cosine of this series."""
        return self._unary_op('mad_tpsa_cos')

    def tan(self) -> Tpsa:
        """Return the tangent of this series."""
        return self._unary_op('mad_tpsa_tan')

    def sinc(self) -> Tpsa:
        """Return ``sin(x) / x`` for this series, with MAD-NG's regularisation at zero."""
        return self._unary_op('mad_tpsa_sinc')

    def asin(self) -> Tpsa:
        """Return the inverse sine of this series."""
        return self._unary_op('mad_tpsa_asin')

    def acos(self) -> Tpsa:
        """Return the inverse cosine of this series."""
        return self._unary_op('mad_tpsa_acos')

    def atan(self) -> Tpsa:
        """Return the inverse tangent of this series."""
        return self._unary_op('mad_tpsa_atan')

    def sincos(self) -> tuple[Tpsa, Tpsa]:
        """Return ``(sin(x), cos(x))`` for this series."""
        return self._pair_op('mad_tpsa_sincos')

    def sincosq(self) -> tuple[Tpsa, Tpsa]:
        """Return ``(sinc(sqrt(x)), cos(sqrt(x)))`` for this series."""
        return self._pair_op('mad_tpsa_sincosq')

    def sincosmq(self) -> tuple[Tpsa, Tpsa]:
        """Return ``((sinc(sqrt(x)) - 1) / x, (cos(sqrt(x)) - 1) / x)``."""
        return self._pair_op('mad_tpsa_sincosmq')

    def sinh(self) -> Tpsa:
        """Return the hyperbolic sine of this series."""
        return self._unary_op('mad_tpsa_sinh')

    def cosh(self) -> Tpsa:
        """Return the hyperbolic cosine of this series."""
        return self._unary_op('mad_tpsa_cosh')

    def tanh(self) -> Tpsa:
        """Return the hyperbolic tangent of this series."""
        return self._unary_op('mad_tpsa_tanh')

    def sinhc(self) -> Tpsa:
        """Return ``sinh(x) / x`` for this series, with MAD-NG's regularisation at zero."""
        return self._unary_op('mad_tpsa_sinhc')

    def asinh(self) -> Tpsa:
        """Return the inverse hyperbolic sine of this series."""
        return self._unary_op('mad_tpsa_asinh')

    def acosh(self) -> Tpsa:
        """Return the inverse hyperbolic cosine of this series."""
        return self._unary_op('mad_tpsa_acosh')

    def atanh(self) -> Tpsa:
        """Return the inverse hyperbolic tangent of this series."""
        return self._unary_op('mad_tpsa_atanh')

    def sincosh(self) -> tuple[Tpsa, Tpsa]:
        """Return ``(sinh(x), cosh(x))`` for this series."""
        return self._pair_op('mad_tpsa_sincosh')

    def sincoshq(self) -> tuple[Tpsa, Tpsa]:
        """Return ``(sinhc(sqrt(x)), cosh(sqrt(x)))`` for this series."""
        return self._pair_op('mad_tpsa_sincoshq')

    def sincoshmq(self) -> tuple[Tpsa, Tpsa]:
        """Return ``((sinhc(sqrt(x)) - 1) / x, (cosh(sqrt(x)) - 1) / x)``."""
        return self._pair_op('mad_tpsa_sincoshmq')

    def erf(self) -> Tpsa:
        """Return the error function of this series."""
        return self._unary_op('mad_tpsa_erf')

    def erfc(self) -> Tpsa:
        """Return the complementary error function of this series."""
        return self._unary_op('mad_tpsa_erfc')

    def erfcx(self) -> Tpsa:
        """Return the scaled complementary error function of this series."""
        return self._unary_op('mad_tpsa_erfcx')

    def erfi(self) -> Tpsa:
        """Return the imaginary error function of this series."""
        return self._unary_op('mad_tpsa_erfi')

    def wofz(self) -> Tpsa:
        """Return the real-valued Faddeeva function for this series.

        MAD-NG names this operation ``wf``. For real TPSAs, it returns the real
        part of SciPy's complex-valued ``scipy.special.wofz``. The imaginary
        part is not represented by this API.
        """
        return self._unary_op('mad_tpsa_wf')

    def atan2(self, other: Tpsa | float) -> Tpsa:
        """Return ``atan2(self, other)``."""
        other = self._coerce_operand(other)
        result = self.descriptor.zero()
        lib().mad_tpsa_atan2(self._ptr, other._ptr, result._ptr)
        return result

    def hypot(self, other: Tpsa | float) -> Tpsa:
        """Return ``sqrt(self**2 + other**2)``."""
        other = self._coerce_operand(other)
        result = self.descriptor.zero()
        lib().mad_tpsa_hypot(self._ptr, other._ptr, result._ptr)
        return result

    def hypot3(self, other: Tpsa | float, third: Tpsa | float) -> Tpsa:
        """Return ``sqrt(self**2 + other**2 + third**2)``."""
        other = self._coerce_operand(other)
        third = self._coerce_operand(third)
        result = self.descriptor.zero()
        lib().mad_tpsa_hypot3(self._ptr, other._ptr, third._ptr, result._ptr)
        return result

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        """Map supported NumPy ufunc calls to the matching TPSA operations."""
        if method != '__call__' or kwargs:
            return NotImplemented

        if ufunc in _UFUNC_DISPATCH:
            return _UFUNC_DISPATCH[ufunc](*inputs)
        return NotImplemented

    def __array_function__(
        self, func: Any, types: tuple[type, ...], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Map supported NumPy functions to matching TPSA operations."""
        if func is np.sinc and args == (self,) and not kwargs:
            return (np.pi * self).sinc()
        return NotImplemented


_UFUNC_DISPATCH = {
    np.absolute: Tpsa.abs,
    np.sqrt: Tpsa.sqrt,
    np.exp: Tpsa.exp,
    np.log: Tpsa.log,
    np.sin: Tpsa.sin,
    np.cos: Tpsa.cos,
    np.tan: Tpsa.tan,
    np.sinh: Tpsa.sinh,
    np.cosh: Tpsa.cosh,
    np.tanh: Tpsa.tanh,
    np.arcsin: Tpsa.asin,
    np.arccos: Tpsa.acos,
    np.arctan: Tpsa.atan,
    np.arcsinh: Tpsa.asinh,
    np.arccosh: Tpsa.acosh,
    np.arctanh: Tpsa.atanh,
    np.negative: Tpsa.__neg__,
    np.positive: Tpsa.__pos__,
    np.add: lambda left, right: left + right,
    np.subtract: lambda left, right: left - right,
    np.multiply: lambda left, right: left * right,
    np.divide: lambda left, right: left / right,
    np.power: lambda left, right: left**right,
    np.atan2: lambda left, right: (
        left.atan2(right)
        if isinstance(left, Tpsa)
        else right.descriptor.constant(float(left)).atan2(right)
    ),
    np.hypot: lambda left, right: (
        left.hypot(right) if isinstance(left, Tpsa) else right.hypot(left)
    ),
    scipy.special.erf: Tpsa.erf,
    scipy.special.erfc: Tpsa.erfc,
    scipy.special.erfcx: Tpsa.erfcx,
    scipy.special.erfi: Tpsa.erfi,
    scipy.special.wofz: Tpsa.wofz,
}
