"""Tpsa: one truncated power series over a ``Descriptor``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import _cffi
from ._cffi import ffi, lib
from .descriptor import Descriptor, _wrap_desc

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_DFLT = 255  # full descriptor order (mad_tpsa_dflt)


class Tpsa:
    """A single truncated power series over a given descriptor (thin handle)."""

    __slots__ = ("_p",)  # pointer to the C tpsa_t* (the series itself)

    def __init__(self, desc: Descriptor | None = None, ptr: Any = None) -> None:
        """Create a new ``Tpsa``."""
        if ptr is not None:
            self._p = ptr
        else:
            if desc is None:
                msg = "desc is required when ptr is not provided"
                raise TypeError(msg)
            self._p = lib().mad_tpsa_newd(desc.ptr, _DFLT)

    def __del__(self) -> None:
        # Read the loaded handle directly: shutdown must not reload the library.
        if getattr(self, "_p", None) is not None and _cffi._lib is not None:
            _cffi._lib.mad_tpsa_del(self._p)
            self._p = None

    @classmethod
    def var(cls, desc: Descriptor, iv: int, v: float = 0.0) -> Tpsa:
        """Create identity variable ``iv`` on ``desc``.

        The variable index starts from 1 and is expanded around value ``v``.
        """
        t = cls(desc)
        lib().mad_tpsa_setvar(t._p, float(v), int(iv), 0.0)
        return t

    @classmethod
    def param(cls, desc: Descriptor, ip: int, v: float = 0.0) -> Tpsa:
        """Create identity parameter ``ip`` on ``desc``.

        The parameter index starts from 1 and uses monomial slot
        ``n_variables + ip - 1``. It is expanded around value ``v``.

        The handle is created with ``mo=1`` (setprm requires it); parameters are
        exact order-1 seeds, use them in arithmetic to build higher orders.
        """
        t = cls.__new__(cls)
        t._p = lib().mad_tpsa_newd(desc.ptr, 1)
        lib().mad_tpsa_setprm(t._p, float(v), int(ip))
        return t

    @property
    def ptr(self) -> Any:
        """The raw ``tpsa_t*`` (for consumers marshalling it across an ABI)."""
        return self._p

    @property
    def descriptor(self) -> Descriptor:
        """The ``Descriptor`` this series lives on (queried from C)."""
        return _wrap_desc(lib().mad_tpsa_desc(self._p))

    @property
    def order(self) -> int:
        """This series' max order (queried from C)."""
        return lib().mad_tpsa_ord(self._p, False)  # noqa: FBT003

    @property
    def const_part(self) -> float:
        """The constant part (zero-order coefficient) of the series."""
        return lib().mad_tpsa_geti(self._p, 0)

    def get(self, monomial: Iterable[int]) -> float:
        """Coefficient of ``monomial`` (iterable of per-variable orders)."""
        m = [int(x) for x in monomial]
        arr = ffi().new("unsigned char[]", m)
        return lib().mad_tpsa_getm(self._p, len(m), arr)

    def set_const_part(self, v: float) -> None:
        """Set the constant part (zero-order coefficient) of the series."""
        lib().mad_tpsa_seti(self._p, 0, 0.0, float(v))

    def set(self, monomial: Iterable[int], v: float) -> None:
        """Set the coefficient of ``monomial`` to ``v``."""
        m = [int(x) for x in monomial]
        arr = ffi().new("unsigned char[]", m)
        lib().mad_tpsa_setm(self._p, len(m), arr, 0.0, float(v))

    def coefficient(
        self, monomials: Sequence[int] | Sequence[Sequence[int]] | np.ndarray
    ) -> float | np.ndarray:
        """Coefficient(s) for one monomial or several.

        A monomial is a length-``nv`` tuple of per-variable orders, the same
        shape ``monomial_coeffs`` returns as keys. ``monomials`` is either one
        such monomial (-> a ``float``) or an iterable of them, e.g. a list of
        tuples or an ``(N, nv)`` array (-> a length-``N`` ``numpy`` array).
        Arrays are accepted and converted to tuples internally.
        """
        m = np.asarray(monomials, dtype=int)
        if m.ndim == 1:
            return self.get(tuple(m))
        if m.ndim == 2:
            return np.array([self.get(tuple(row)) for row in m])
        raise ValueError("monomials must be one monomial (nv,) or several (N, nv)")

    def monomial_coeffs(self, tol: float = 1e-14) -> dict[tuple[int, ...], float]:
        """All coefficients with ``|c| > tol`` as ``{monomial_tuple: coefficient}``.

        Enumerates only the stored (nonzero) terms via ``mad_tpsa_cycle``.
        Monomials are full length ``n_variables + n_parameters``.
        """
        n = self.descriptor.monomial_length
        m = ffi().new("unsigned char[]", n)
        v = ffi().new("double*")
        out = {}
        i = -1
        while (i := lib().mad_tpsa_cycle(self._p, i, n, m, v)) >= 0:
            if abs(v[0]) > tol:
                out[tuple(m[j] for j in range(n))] = v[0]
        return out

    def grad(self) -> list[float]:
        """Order-1 coefficients, one per variable."""
        n = self.descriptor.n_variables
        g = []
        for j in range(n):
            mono = [0] * n
            mono[j] = 1
            g.append(self.get(mono))
        return g

    def param_grad(self) -> list[float]:
        """Order-1 coefficients (d out / d param_j), one per parameter."""
        nv, _, np_, _ = self.descriptor._getnv()
        g = []
        for j in range(np_):
            mono = [0] * (nv + np_)
            mono[nv + j] = 1
            g.append(self.get(mono))
        return g

    # --- arithmetic (fresh result on the same descriptor; scalars mix freely) --- #

    def _new_like(self) -> Tpsa:
        return Tpsa(self.descriptor)

    def _binop(self, other: Tpsa, fn: str) -> Tpsa:
        r = self._new_like()
        getattr(lib(), fn)(self._p, other._p, r._p)
        return r

    def __add__(self, other: Tpsa | float) -> Tpsa:
        if isinstance(other, Tpsa):
            return self._binop(other, "mad_tpsa_add")
        r = self._new_like()
        lib().mad_tpsa_axpb(1.0, self._p, float(other), r._p)
        return r

    __radd__ = __add__

    def __sub__(self, other: Tpsa | float) -> Tpsa:
        if isinstance(other, Tpsa):
            return self._binop(other, "mad_tpsa_sub")
        return self.__add__(-float(other))

    def __rsub__(self, other: float) -> Tpsa:
        r = self._new_like()
        lib().mad_tpsa_axpb(-1.0, self._p, float(other), r._p)
        return r

    def __mul__(self, other: Tpsa | float) -> Tpsa:
        if isinstance(other, Tpsa):
            return self._binop(other, "mad_tpsa_mul")
        r = self._new_like()
        lib().mad_tpsa_scl(self._p, float(other), r._p)
        return r

    __rmul__ = __mul__

    def __truediv__(self, other: Tpsa | float) -> Tpsa:
        if isinstance(other, Tpsa):
            return self._binop(other, "mad_tpsa_div")
        return self.__mul__(1.0 / float(other))

    def __rtruediv__(self, other: float) -> Tpsa:
        r = self._new_like()
        lib().mad_tpsa_inv(self._p, float(other), r._p)
        return r

    def __pow__(self, other: int | float) -> Tpsa:
        r = self._new_like()
        lib().mad_tpsa_pown(self._p, float(other), r._p)
        return r

    def __neg__(self) -> Tpsa:
        return self.__mul__(-1.0)

    def copy(self) -> Tpsa:
        """Return a copy of this series."""
        r = self._new_like()
        lib().mad_tpsa_copy(self._p, r._p)
        return r
