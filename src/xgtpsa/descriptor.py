"""GTPSA descriptor objects.

A descriptor defines the algebraic space for TPSA series: how many variables are
available, the maximum polynomial order, and optionally how many parameters are
part of the monomials. Every ``Tpsa`` object is created on a descriptor, and
series can only be combined meaningfully when they belong to compatible
descriptors.

The Python object is a small handle to the underlying MAD-NG GTPSA descriptor.
MAD-NG interns equivalent descriptors, and this module mirrors that by reusing the
same Python ``Descriptor`` object for the same C descriptor pointer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

from ._cffi import ffi, lib
from .tpsa import Tpsa

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class _DescriptorAttrs(NamedTuple):
    num_vars: int
    order: int
    num_params: int
    param_order: int


class Descriptor:
    """Algebraic space definition for one or more ``Tpsa`` series.

    Parameters are appended after the variables in monomial tuples. For example,
    a descriptor with two variables and one parameter uses monomials of length
    three, where the last entry is the parameter order.
    """

    _instances_by_ptr: ClassVar[dict[int, Descriptor]] = {}
    _ptr: Any

    __slots__ = ("_ptr",)  # pointer to the C descriptor (mad_desc_t*)

    def __new__(
        cls,
        num_vars: int,
        order: int,
        num_params: int = 0,
        param_order: int = 1,
    ) -> Descriptor:
        """Create or reuse a descriptor."""
        if order <= 0:
            raise ValueError("Descriptor order must be positive")
        if num_params > 0 and param_order <= 0:
            raise ValueError("Descriptor parameter order must be positive")

        if num_params > 0:
            ptr = lib().mad_desc_newvp(num_vars, order, num_params, param_order)
        else:
            ptr = lib().mad_desc_newv(num_vars, order)

        return cls.from_ptr(ptr)

    @classmethod
    def from_ptr(cls, ptr: Any) -> Descriptor:
        """Return the interned ``Descriptor`` for a raw C pointer."""
        key = int(ffi().cast("uintptr_t", ptr))
        descriptor = cls._instances_by_ptr.get(key)
        if descriptor is None:
            descriptor = super().__new__(cls)
            descriptor._ptr = ptr
            cls._instances_by_ptr[key] = descriptor
        return descriptor

    @property
    def ptr(self) -> Any:
        """Descriptor pointer."""
        return self._ptr

    def _get_descriptor_attrs(self) -> _DescriptorAttrs:
        """Query the attributes of the GTPSA descriptor."""
        order_ptr = ffi().new("unsigned char*")
        num_params_ptr = ffi().new("int*")
        param_order_ptr = ffi().new("unsigned char*")

        num_vars = lib().mad_desc_getnv(self._ptr, order_ptr, num_params_ptr, param_order_ptr)

        return _DescriptorAttrs(
            num_vars=num_vars,
            order=order_ptr[0],
            num_params=num_params_ptr[0],
            param_order=param_order_ptr[0],
        )

    @property
    def num_vars(self) -> int:
        """Number of variables (excluding parameters) supported by the descriptor."""
        return self._get_descriptor_attrs().num_vars

    @property
    def order(self) -> int:
        """Maximum order supported by the descriptor."""
        return self._get_descriptor_attrs().order

    @property
    def num_params(self) -> int:
        """Number of parameters supported by the descriptor."""
        return self._get_descriptor_attrs().num_params

    @property
    def param_order(self) -> int:
        """Combined parameter order cap of the descriptor."""
        return self._get_descriptor_attrs().param_order

    @property
    def monomial_length(self) -> int:
        """Length of a full monomial: ``num_vars + num_params``."""
        attrs = self._get_descriptor_attrs()
        return attrs.num_vars + attrs.num_params

    def is_valid_monomial(self, monomial: Iterable[int]) -> bool:
        """Whether ``monomial`` is representable (querying beyond-order aborts C)."""
        m = [int(x) for x in monomial]
        arr = ffi().new("unsigned char[]", m)
        return bool(lib().mad_desc_isvalidm(self._ptr, len(m), arr))

    def constant(self, value: float, /) -> Tpsa:
        """Create a constant TPSA series on this descriptor."""
        t = self.zero()
        lib().mad_tpsa_seti(t.ptr, 0, 0.0, float(value))
        return t

    def zero(self) -> Tpsa:
        """Create a zero TPSA series on this descriptor."""
        return Tpsa(self)

    def var(self, index: int, value: float = 0.0) -> Tpsa:
        """Create identity variable ``index`` on this descriptor.

        The variable index starts from 1 and is expanded around ``value``.
        """
        t = Tpsa(self)
        lib().mad_tpsa_setvar(t.ptr, float(value), int(index), 0.0)
        return t

    def vars(self, values: Sequence[float] | None = None) -> tuple[Tpsa, ...]:
        """Create identity series for all variables on this descriptor.

        If ``values`` is provided, each variable is expanded around the
        corresponding value.
        """
        if values is None:
            values = [0.0] * self.num_vars
        if len(values) != self.num_vars:
            raise ValueError("values must contain one entry per variable")
        return tuple(self.var(index, value) for index, value in enumerate(values, start=1))

    def param(self, index: int, value: float = 0.0) -> Tpsa:
        """Create identity parameter ``index`` on this descriptor.

        The parameter index starts from 1. Parameters are appended after
        variables in monomial tuples.
        """
        t = Tpsa.from_ptr(lib().mad_tpsa_newd(self.ptr, 1))
        lib().mad_tpsa_setprm(t.ptr, float(value), int(index))
        return t

    def params(self) -> tuple[Tpsa, ...]:
        """Create identity series for all parameters on this descriptor."""
        return tuple(self.param(index) for index in range(1, self.num_params + 1))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Descriptor) and self._ptr == other._ptr

    def __hash__(self) -> int:
        return int(ffi().cast("uintptr_t", self._ptr))

    def __repr__(self) -> str:
        attrs = self._get_descriptor_attrs()

        if attrs.num_params:
            return (
                f"Descriptor(num_vars={attrs.num_vars}, order={attrs.order}, "
                f"num_params={attrs.num_params}, param_order={attrs.param_order})"
            )

        return f"Descriptor(num_vars={attrs.num_vars}, order={attrs.order})"
