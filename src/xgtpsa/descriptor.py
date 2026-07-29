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

import warnings
import weakref
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

from . import _cffi
from ._cffi import ffi, lib
from .tpsa import Numeric, Tpsa

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .formatting import PolynomialStyle


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

    _instances_by_ptr: ClassVar[dict[int, weakref.ReferenceType[Descriptor]]] = {}
    _ptr: Any
    var_labels: tuple[str, ...]
    param_labels: tuple[str, ...]

    __slots__ = ('_ptr', 'param_labels', 'var_labels', '__weakref__')

    def __new__(
        cls,
        num_vars: int | None = None,
        order: int | None = None,
        num_params: int = 0,
        param_order: int = 1,
        *,
        vars: Sequence[str] | None = None,  # noqa: A002
        params: Sequence[str] | None = None,
    ) -> Descriptor:
        """Create or reuse a descriptor."""
        if order is None:
            raise TypeError('Descriptor order is required')

        var_labels = cls._resolve_labels('vars', num_vars, vars, 'v')
        param_labels = cls._resolve_labels(
            'params', num_params, params, 'p', zero_count_is_unspecified=True
        )
        num_vars = len(var_labels)
        num_params = len(param_labels)

        if order <= 0:
            raise ValueError('Descriptor order must be positive')
        if num_params > 0 and param_order <= 0:
            raise ValueError('Descriptor parameter order must be positive')

        if num_params > 0:
            ptr = lib().mad_desc_newvp(num_vars, order, num_params, param_order)
        else:
            ptr = lib().mad_desc_newv(num_vars, order)

        return cls.from_ptr(
            ptr,
            var_labels=var_labels,
            param_labels=param_labels,
            explicit_var_labels=vars is not None,
            explicit_param_labels=params is not None,
        )

    @classmethod
    def _resolve_labels(
        cls,
        name: str,
        count: int | None,
        labels: Sequence[str] | None,
        prefix: str,
        *,
        zero_count_is_unspecified: bool = False,
    ) -> tuple[str, ...]:
        if labels is None:
            if count is None:
                raise TypeError(f'Descriptor {name} count or labels are required')
            return tuple(f'{prefix}_{index}' for index in range(1, int(count) + 1))

        resolved = tuple(str(label) for label in labels)
        count_is_unspecified = zero_count_is_unspecified and count == 0
        if count is not None and not count_is_unspecified and int(count) != len(resolved):
            raise ValueError(f'{name} labels must contain {count} entries')
        return resolved

    @classmethod
    def from_ptr(
        cls,
        ptr: Any,
        *,
        var_labels: Sequence[str] | None = None,
        param_labels: Sequence[str] | None = None,
        explicit_var_labels: bool = False,
        explicit_param_labels: bool = False,
    ) -> Descriptor:
        """Return the interned ``Descriptor`` for a raw C pointer."""
        key = int(ffi().cast('uintptr_t', ptr))
        descriptor_ref = cls._instances_by_ptr.get(key)
        descriptor = descriptor_ref() if descriptor_ref is not None else None
        if descriptor is not None:
            descriptor._warn_if_labels_differ(
                var_labels,
                param_labels,
                explicit_var_labels=explicit_var_labels,
                explicit_param_labels=explicit_param_labels,
            )
            return descriptor

        descriptor = super().__new__(cls)
        descriptor._ptr = ptr
        descriptor.var_labels = (
            tuple(var_labels) if var_labels is not None else descriptor._default_var_labels()
        )
        descriptor.param_labels = (
            tuple(param_labels) if param_labels is not None else descriptor._default_param_labels()
        )
        cls._instances_by_ptr[key] = weakref.ref(descriptor)
        return descriptor

    def __del__(self) -> None:
        if getattr(self, '_ptr', None) is None:
            return

        key = int(ffi().cast('uintptr_t', self._ptr))
        descriptor_ref = self._instances_by_ptr.get(key)
        if descriptor_ref is not None and descriptor_ref() is self:
            del self._instances_by_ptr[key]

        if _cffi._lib is not None:
            _cffi._lib.mad_desc_del(self._ptr)
        self._ptr = None

    def _warn_if_labels_differ(
        self,
        var_labels: Sequence[str] | None,
        param_labels: Sequence[str] | None,
        *,
        explicit_var_labels: bool,
        explicit_param_labels: bool,
    ) -> None:
        if explicit_var_labels and var_labels is not None and tuple(var_labels) != self.var_labels:
            warnings.warn(
                'Descriptor already exists with different variable labels; reusing existing labels',
                UserWarning,
                stacklevel=3,
            )
        if (
            explicit_param_labels
            and param_labels is not None
            and tuple(param_labels) != self.param_labels
        ):
            warnings.warn(
                'Descriptor already exists with different parameter labels; '
                'reusing existing labels',
                UserWarning,
                stacklevel=3,
            )

    def _default_var_labels(self) -> tuple[str, ...]:
        return tuple(f'v_{index}' for index in range(1, self.num_vars + 1))

    def _default_param_labels(self) -> tuple[str, ...]:
        return tuple(f'p_{index}' for index in range(1, self.num_params + 1))

    def __init__(
        self,
        num_vars: int | None = None,
        order: int | None = None,
        num_params: int = 0,
        param_order: int = 1,
        *,
        vars: Sequence[str] | None = None,  # noqa: A002
        params: Sequence[str] | None = None,
    ) -> None:
        """Descriptor initialization is handled in ``__new__`` for interning."""
        pass

    @property
    def ptr(self) -> Any:
        """Descriptor pointer."""
        return self._ptr

    def _get_descriptor_attrs(self) -> _DescriptorAttrs:
        """Query the attributes of the GTPSA descriptor."""
        order_ptr = ffi().new('unsigned char*')
        num_params_ptr = ffi().new('int*')
        param_order_ptr = ffi().new('unsigned char*')

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
        arr = ffi().new('unsigned char[]', m)
        return bool(lib().mad_desc_isvalidm(self._ptr, len(m), arr))

    def constant(self, value: Numeric, /) -> Tpsa:
        """Create a constant TPSA series on this descriptor."""
        t = self.zero()
        lib().mad_tpsa_seti(t.ptr, 0, 0.0, float(value))
        return t

    def zero(self) -> Tpsa:
        """Create a zero TPSA series on this descriptor."""
        return Tpsa(self)

    def _var_index(self, variable: int | str) -> int:
        if isinstance(variable, str):
            try:
                return self.var_labels.index(variable) + 1
            except ValueError as exc:
                raise KeyError(variable) from exc
        return int(variable)

    def _param_index(self, parameter: int | str) -> int:
        if isinstance(parameter, str):
            try:
                return self.param_labels.index(parameter) + 1
            except ValueError as exc:
                raise KeyError(parameter) from exc
        return int(parameter)

    def variable_index(self, variable: int | str) -> int:
        """Return the 1-based combined variable/parameter index for ``variable``."""
        if isinstance(variable, str):
            if variable in self.var_labels:
                return self.var_labels.index(variable) + 1
            if variable in self.param_labels:
                return self.num_vars + self.param_labels.index(variable) + 1
            raise KeyError(variable)
        return int(variable)

    def var(self, index: int | str, value: Numeric = 0.0) -> Tpsa:
        """Create identity variable ``index`` on this descriptor.

        The variable index starts from 1 and is expanded around ``value``.
        A variable label may be passed instead of an index.
        """
        index = self._var_index(index)
        t = Tpsa(self)
        lib().mad_tpsa_setvar(t.ptr, float(value), int(index), 0.0)
        return t

    def vars(self, values: Sequence[Numeric] | None = None) -> tuple[Tpsa, ...]:
        """Create identity series for all variables on this descriptor.

        If ``values`` is provided, each variable is expanded around the
        corresponding value.
        """
        if values is None:
            values = [0.0] * self.num_vars
        if len(values) != self.num_vars:
            raise ValueError('values must contain one entry per variable')
        return tuple(self.var(index, value) for index, value in enumerate(values, start=1))

    def param(self, index: int | str, value: Numeric = 0.0) -> Tpsa:
        """Create identity parameter ``index`` on this descriptor.

        The parameter index starts from 1. Parameters are appended after
        variables in monomial tuples. A parameter label may be passed instead of
        an index.
        """
        index = self._param_index(index)
        t = Tpsa.from_ptr(lib().mad_tpsa_newd(self.ptr, 1), descriptor=self)
        lib().mad_tpsa_setprm(t.ptr, float(value), int(index))
        return t

    def params(self) -> tuple[Tpsa, ...]:
        """Create identity series for all parameters on this descriptor."""
        return tuple(self.param(index) for index in range(1, self.num_params + 1))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Descriptor) and self._ptr == other._ptr

    def __hash__(self) -> int:
        return int(ffi().cast('uintptr_t', self._ptr))

    def __repr__(self) -> str:
        attrs = self._get_descriptor_attrs()

        if attrs.num_params:
            return (
                f'Descriptor(num_vars={attrs.num_vars}, order={attrs.order}, '
                f'num_params={attrs.num_params}, param_order={attrs.param_order}, '
                f'var_labels={self.var_labels!r}, param_labels={self.param_labels!r})'
            )

        return (
            f'Descriptor(num_vars={attrs.num_vars}, order={attrs.order}, '
            f'var_labels={self.var_labels!r})'
        )

    def format_polynomial(self, tpsa: Tpsa, style: PolynomialStyle = 'code') -> object:
        """Format ``tpsa`` using this descriptor's variable and parameter labels."""
        if tpsa.descriptor is not self:
            raise ValueError('Cannot format a TPSA from a different descriptor')

        from .formatting import format_polynomial

        return format_polynomial(tpsa, self.var_labels + self.param_labels, style=style)
