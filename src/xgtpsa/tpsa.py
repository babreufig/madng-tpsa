"""Truncated power series objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import _cffi
from ._cffi import ffi, lib

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .descriptor import Descriptor


class Tpsa:
    """A truncated power series in the algebraic space defined by a descriptor."""

    __slots__ = ("_ptr",)  # pointer to the C tpsa_t* (the series itself)

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
        if getattr(self, "_ptr", None) is not None and _cffi._lib is not None:
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
        monomial_arr = ffi().new("unsigned char[]", monomial_orders)
        return lib().mad_tpsa_getm(self._ptr, len(monomial_orders), monomial_arr)

    def set_const_part(self, v: float) -> None:
        """Set the constant coefficient."""
        lib().mad_tpsa_seti(self._ptr, 0, 0.0, float(v))

    def set(self, monomial: Iterable[int], value: float) -> None:
        """Set the coefficient for ``monomial``."""
        monomial_orders = list(monomial)
        monomial_arr = ffi().new("unsigned char[]", monomial_orders)
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
        raise ValueError("monomials must be one monomial or a two-dimensional batch")

    def monomial_coeffs(self, tol: float = 1e-14) -> dict[tuple[int, ...], float]:
        """Return stored coefficients larger than ``tol`` in absolute value.

        Keys are full monomial tuples with one entry per variable and parameter.
        """
        monomial_len = self.descriptor.monomial_length
        monomial_arr = ffi().new("unsigned char[]", monomial_len)
        coeff_ptr = ffi().new("double*")
        coeffs = {}

        i = -1
        while (i := lib().mad_tpsa_cycle(self._ptr, i, monomial_len, monomial_arr, coeff_ptr)) >= 0:
            coefficient = coeff_ptr[0]
            if abs(coefficient) <= tol:
                continue
            monomial = tuple(monomial_arr)
            coeffs[monomial] = coefficient

        return coeffs

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

    # --- arithmetic (fresh result on the same descriptor; scalars mix freely) --- #

    def _binop(self, other: Tpsa, fn: str) -> Tpsa:
        if not lib().xgtpsa_check_tpsa_compatibility(self._ptr, other._ptr):
            raise ValueError("Incompatible TPSA descriptors")

        result = self.descriptor.zero()
        getattr(lib(), fn)(self._ptr, other._ptr, result._ptr)
        return result

    def __add__(self, other: Tpsa | float) -> Tpsa:
        if isinstance(other, Tpsa):
            return self._binop(other, "mad_tpsa_add")
        result = self.descriptor.zero()
        lib().mad_tpsa_axpb(1.0, self._ptr, float(other), result._ptr)
        return result

    __radd__ = __add__

    def __sub__(self, other: Tpsa | float) -> Tpsa:
        if isinstance(other, Tpsa):
            return self._binop(other, "mad_tpsa_sub")
        return self.__add__(-float(other))

    def __rsub__(self, other: float) -> Tpsa:
        result = self.descriptor.zero()
        lib().mad_tpsa_axpb(-1.0, self._ptr, float(other), result._ptr)
        return result

    def __mul__(self, other: Tpsa | float) -> Tpsa:
        if isinstance(other, Tpsa):
            return self._binop(other, "mad_tpsa_mul")
        result = self.descriptor.zero()
        lib().mad_tpsa_scl(self._ptr, float(other), result._ptr)
        return result

    __rmul__ = __mul__

    def __truediv__(self, other: Tpsa | float) -> Tpsa:
        if isinstance(other, Tpsa):
            return self._binop(other, "mad_tpsa_div")
        return self.__mul__(1.0 / float(other))

    def __rtruediv__(self, other: float) -> Tpsa:
        result = self.descriptor.zero()
        lib().mad_tpsa_inv(self._ptr, float(other), result._ptr)
        return result

    def __pow__(self, other: int | float) -> Tpsa:
        result = self.descriptor.zero()
        lib().mad_tpsa_pown(self._ptr, float(other), result._ptr)
        return result

    def __neg__(self) -> Tpsa:
        return self * -1.0
