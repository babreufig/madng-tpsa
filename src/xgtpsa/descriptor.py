"""The GTPSA descriptor: the (variables, order, parameters) space a series lives in.

There is no Python-side cache of the descriptor state.
A descriptor is created explicitly (``Descriptor.new``)
and can be queried straight from a series (``t.descriptor``).
Properties such as number of variables, order, and parameters are always read from C to avoid inconsistencies.
Every descriptor created is kept as a pointer in ``_DESCRIPTORS`` (MAD-NG reuses
equivalent descriptors, so this is the live set), see ``live_descriptors()``.
"""

from __future__ import annotations

from typing import Any, Iterable

from ._cffi import ffi, lib

_DESCRIPTORS: dict[int, Descriptor] = {}  # int(address) -> Descriptor


class Descriptor:
    """Thin handle on a live GTPSA descriptor; ``n_variables``/``order`` come from C."""

    __slots__ = ("_d",)  # pointer to the C descriptor (mad_desc_t*)

    def __init__(self, ptr: Any) -> None:
        self._d = ptr

    @classmethod
    def new(
        cls,
        num_variables: int,
        order: int,
        num_parameters: int = 0,
        param_order: int = 1,
    ) -> Descriptor:
        """Create (or reuse) a GTPSA descriptor with ``num_variables`` variables and max ``order``.

        With ``num_parameters > 0``, adds parameters: extra variables appended
        positions ``num_variables..num_variables+num_parameters-1`` whose combined order
        is capped at ``param_order``.
        GTPSA reuses equivalent descriptors, so this returns the same ``Descriptor``
        object for the same arguments.
        Warns if GTPSA library coerces ``order``/``param_order`` (minimum is 1).
        """
        if num_parameters > 0:
            d = _wrap_desc(
                lib().mad_desc_newvp(num_variables, order, num_parameters, param_order)
            )
        else:
            d = _wrap_desc(lib().mad_desc_newv(num_variables, order))
        if d.order != order or (num_parameters > 0 and d.param_order != param_order):
            import warnings
            warnings.warn(
                f"Requested order {order}/param_order {param_order} coerced to "
                f"{d.order}/{d.param_order} (GTPSA minimum)",
                stacklevel=2,
            )
        return d

    @property
    def ptr(self) -> Any:
        return self._d

    def _getnv(self) -> tuple[int, int, int, int]:
        mo = ffi().new("unsigned char*")
        np_ = ffi().new("int*")
        po = ffi().new("unsigned char*")
        n = lib().mad_desc_getnv(self._d, mo, np_, po)
        return n, mo[0], np_[0], po[0]

    @property
    def n_variables(self) -> int:
        """Number of variables (queried from C, GTPSA's ``nv``; excludes parameters)."""
        return self._getnv()[0]

    @property
    def order(self) -> int:
        """Maximum order (queried from C, GTPSA's ``mo``)."""
        return self._getnv()[1]

    @property
    def n_parameters(self) -> int:
        """Number of parameters (queried from C, GTPSA's ``np``)."""
        return self._getnv()[2]

    @property
    def param_order(self) -> int:
        """Combined parameter order cap (queried from C, GTPSA's ``po``)."""
        return self._getnv()[3]

    @property
    def monomial_length(self) -> int:
        """Length of a full monomial: ``n_variables + n_parameters``."""
        n, _, np_, _ = self._getnv()
        return n + np_

    def is_valid_monomial(self, monomial: Iterable[int]) -> bool:
        """Whether ``monomial`` is representable (querying beyond-order aborts C)."""
        m = [int(x) for x in monomial]
        arr = ffi().new("unsigned char[]", m)
        return bool(lib().mad_desc_isvalidm(self._d, len(m), arr))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Descriptor) and self._d == other._d

    def __hash__(self) -> int:
        return int(ffi().cast("uintptr_t", self._d))

    def __repr__(self) -> str:
        nv, order, np_, po = self._getnv()
        if np_:
            return f"Descriptor(nv={nv}, order={order}, np={np_}, po={po})"
        return f"Descriptor(nv={nv}, order={order})"


def _wrap_desc(ptr: Any) -> Descriptor:
    """Return the ``Descriptor`` for a raw C pointer, registering it."""
    key = int(ffi().cast("uintptr_t", ptr))
    d = _DESCRIPTORS.get(key)
    if d is None:
        d = _DESCRIPTORS[key] = Descriptor(ptr)
    return d


def live_descriptors() -> list[Descriptor]:
    """Every ``Descriptor`` created so far (MAD-NG reuses equivalent descriptors, so this is the live set)."""
    return list(_DESCRIPTORS.values())
