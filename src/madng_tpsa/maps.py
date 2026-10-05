"""Map operations for MAD-NG TPSA series.

The native MAD-NG library represents a map as an array of TPSA pointers rather
than as a separate C object. This module mirrors that design: the functional API
accepts sequences of :class:`Tpsa` / :class:`ComplexTpsa`, while :class:`TpsaMap`
provides a small Python container around such a sequence.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from numbers import Integral
from typing import Any, Generic, SupportsComplex, SupportsFloat, TypeAlias, TypeVar, cast, overload

import numpy as np

from ._cffi import ffi, lib
from ._tpsa_base import _TpsaBase
from .complex_tpsa import ComplexTpsa
from .tpsa import Tpsa

Series: TypeAlias = Tpsa | ComplexTpsa
Scalar: TypeAlias = SupportsFloat | SupportsComplex
SeriesT = TypeVar('SeriesT', Tpsa, ComplexTpsa)
SelectionValue: TypeAlias = bool | int | np.integer[Any]


def _raise_mad_error() -> None:
    _TpsaBase._raise_mad_error()


def _series_tuple(values: Sequence[Series] | TpsaMap) -> tuple[Series, ...]:
    if isinstance(values, TpsaMap):
        values = values.coords
    result = tuple(values)
    if not result:
        message = 'TPSA map cannot be empty'
        raise ValueError(message)
    for value in result:
        if not isinstance(value, (Tpsa, ComplexTpsa)):
            message = 'Map components must be Tpsa or ComplexTpsa'
            raise TypeError(message)
    descriptor = result[0].descriptor
    if any(value.descriptor is not descriptor for value in result[1:]):
        message = 'All map components must share the same Descriptor'
        raise ValueError(message)
    return result


def _is_complex(values: Sequence[Series]) -> bool:
    return any(isinstance(value, ComplexTpsa) for value in values)


def _as_complex(value: Series) -> ComplexTpsa:
    if isinstance(value, ComplexTpsa):
        return value.copy()
    return ComplexTpsa.from_tpsa(value)


def _promote_complex(values: Sequence[Series]) -> tuple[ComplexTpsa, ...]:
    return tuple(
        value if isinstance(value, ComplexTpsa) else ComplexTpsa.from_tpsa(value)
        for value in values
    )


def _coerce_pair(
    left: Sequence[Series] | TpsaMap, right: Sequence[Series] | TpsaMap
) -> tuple[tuple[Series, ...], tuple[Series, ...]]:
    left_series = _series_tuple(left)
    right_series = _series_tuple(right)
    if left_series[0].descriptor is not right_series[0].descriptor:
        message = 'TPSA maps must share the same Descriptor'
        raise ValueError(message)
    if _is_complex(left_series) or _is_complex(right_series):
        return _promote_complex(left_series), _promote_complex(right_series)
    return left_series, right_series


def _zeros_like(values: Sequence[Series]) -> tuple[Series, ...]:
    result = []
    for value in values:
        if isinstance(value, ComplexTpsa):
            result.append(value.descriptor.complex_zero(order=value.order))
        else:
            result.append(value.descriptor.zero(order=value.order))
    return tuple(result)


def _parameter_identities(descriptor, *, complex_: bool) -> tuple[Series, ...]:
    params = descriptor.params()
    if complex_:
        return tuple(ComplexTpsa.from_tpsa(value) for value in params)
    return tuple(params)


def _complete_substitution(values: Sequence[Series]) -> tuple[Series, ...]:
    descriptor = values[0].descriptor
    if len(values) != descriptor.num_vars:
        message = (
            f'A full map requires {descriptor.num_vars} coordinate components, got {len(values)}'
        )
        raise ValueError(message)
    return (
        *values,
        *_parameter_identities(descriptor, complex_=_is_complex(values)),
    )


def _pointer_array(values: Sequence[Series]):
    return ffi.new('void*[]', [value.ptr for value in values])


def _protected_call(family: str, function: Any, *, complex_: bool, args: Any) -> None:
    kind = 'ctpsa' if complex_ else 'tpsa'
    wrapper = getattr(
        lib,
        f'madng_{kind}_protected_{family}_call',
    )
    status = wrapper(function, *args)
    if status:
        _raise_mad_error()


def _normalise_selection(select: Sequence[SelectionValue], expected_length: int) -> list[int]:
    if len(select) != expected_length:
        message = f'select must contain {expected_length} entries, got {len(select)}'
        raise ValueError(message)
    result = []
    for value in select:
        if isinstance(value, (bool, np.bool_)):
            result.append(int(value))
            continue
        if not isinstance(value, Integral):
            message = 'select entries must be booleans or integers 0 or 1'
            raise TypeError(message)
        value = int(value)
        if value not in (0, 1):
            message = 'integer select entries must be 0 or 1'
            raise ValueError(message)
        result.append(value)
    return result


def _set_series_const_part(component: Series, value: Scalar) -> None:
    converted = complex(value)
    if isinstance(component, Tpsa):
        if converted.imag != 0:
            err_mess = 'Cannot assign a complex constant to a real TPSA map'
            raise TypeError(err_mess)
        component.set_const_part(converted.real)
    else:
        component.set_const_part(converted)


def _set_series_coefficient(component: Series, monomial, value: Scalar) -> None:
    if isinstance(component, Tpsa):
        converted = complex(value)
        if converted.imag != 0:
            err_mess = 'Cannot assign a complex coefficient to a real TPSA map'
            raise TypeError(err_mess)
        component.set(monomial, converted.real)
    else:
        component.set(monomial, value)


@overload
def compose(
    left: Sequence[Tpsa] | TpsaMap[Tpsa], right: Sequence[Tpsa] | TpsaMap[Tpsa]
) -> tuple[Tpsa, ...]: ...
@overload
def compose(
    left: Sequence[Tpsa] | TpsaMap[Tpsa], right: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa]
) -> tuple[ComplexTpsa, ...]: ...
@overload
def compose(
    left: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa],
    right: (Sequence[Tpsa] | Sequence[ComplexTpsa] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]),
) -> tuple[ComplexTpsa, ...]: ...
@overload
def compose(
    left: Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
    right: Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
) -> tuple[Series, ...]: ...


def compose(
    left: Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
    right: Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
) -> tuple[Series, ...]:
    """Return ``left ∘ right``, i.e. ``left(right(z))``.

    ``right`` must contain one component for every descriptor variable. Descriptor
    parameters are automatically appended as identity substitutions.
    """
    left_series, right_series = _coerce_pair(left, right)
    descriptor = left_series[0].descriptor

    if len(left_series) > descriptor.num_vars:
        message = (
            f'Left map has {len(left_series)} components, but the descriptor '
            f'has only {descriptor.num_vars} variables'
        )
        raise ValueError(message)
    if len(right_series) != descriptor.num_vars:
        message = f'Right map must have {descriptor.num_vars} components, got {len(right_series)}'
        raise ValueError(message)

    right_full = _complete_substitution(right_series)
    result = _zeros_like(left_series)
    complex_ = _is_complex(left_series)

    _protected_call(
        'map_binary',
        lib.mad_ctpsa_compose if complex_ else lib.mad_tpsa_compose,
        complex_=complex_,
        args=(
            len(left_series),
            _pointer_array(left_series),
            len(right_full),
            _pointer_array(right_full),
            _pointer_array(result),
        ),
    )
    return result


@overload
def inverse(values: Sequence[Tpsa] | TpsaMap[Tpsa]) -> tuple[Tpsa, ...]: ...
@overload
def inverse(values: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa]) -> tuple[ComplexTpsa, ...]: ...
@overload
def inverse(values: Sequence[Series]) -> tuple[Series, ...]: ...


def inverse(values: Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]) -> tuple[Series, ...]:
    """Return the inverse of a full variable map centered at the origin.

    Descriptor parameters are treated as independent parameters and carried
    through the inversion unchanged.
    """
    series = _series_tuple(values)
    descriptor = series[0].descriptor
    if len(series) != descriptor.num_vars:
        message = f'Inverse requires a full {descriptor.num_vars}-component map, got {len(series)}'
        raise ValueError(message)

    full_input = _complete_substitution(series)
    coordinate_output = _zeros_like(series)
    full_output = (
        *coordinate_output,
        *_parameter_identities(descriptor, complex_=_is_complex(series)),
    )
    complex_ = _is_complex(series)

    _protected_call(
        'map_inverse',
        lib.mad_ctpsa_minv if complex_ else lib.mad_tpsa_minv,
        complex_=complex_,
        args=(
            len(full_input),
            _pointer_array(full_input),
            descriptor.num_vars,
            _pointer_array(full_output),
        ),
    )
    return tuple(full_output[: descriptor.num_vars])


@overload
def partial_inverse(
    values: Sequence[Tpsa] | TpsaMap[Tpsa], select: Sequence[SelectionValue]
) -> tuple[Tpsa, ...]: ...
@overload
def partial_inverse(
    values: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa], select: Sequence[SelectionValue]
) -> tuple[ComplexTpsa, ...]: ...
@overload
def partial_inverse(
    values: Sequence[Series], select: Sequence[SelectionValue]
) -> tuple[Series, ...]: ...


def partial_inverse(
    values: Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
    select: Sequence[SelectionValue],
) -> tuple[Series, ...]:
    """Return a partial inverse for selected variable rows of a full map.

    ``select`` has one entry per descriptor variable. Parameters are never
    selected for inversion. The map has to be a full variable map centered
    at the origin.
    """
    series = _series_tuple(values)
    descriptor = series[0].descriptor
    if len(series) != descriptor.num_vars:
        message = (
            f'Partial inverse requires a full {descriptor.num_vars}-component map, '
            f'got {len(series)}'
        )
        raise ValueError(message)

    full_input = _complete_substitution(series)
    coordinate_output = _zeros_like(series)
    full_output = (
        *coordinate_output,
        *_parameter_identities(descriptor, complex_=_is_complex(series)),
    )
    full_select = _normalise_selection(select, descriptor.num_vars)
    full_select.extend([0] * descriptor.num_params)

    select_array = ffi.new('int[]', full_select)
    complex_ = _is_complex(series)

    _protected_call(
        'map_partial_inverse',
        lib.mad_ctpsa_pminv if complex_ else lib.mad_tpsa_pminv,
        complex_=complex_,
        args=(
            len(full_input),
            _pointer_array(full_input),
            descriptor.num_vars,
            _pointer_array(full_output),
            select_array,
        ),
    )
    return tuple(full_output[: descriptor.num_vars])


def evaluate(
    values: Sequence[Series] | TpsaMap,
    coordinates: Sequence[Scalar],
    *,
    parameters: Sequence[Scalar] | None = None,
) -> np.ndarray:
    """Evaluate a TPSA map at numerical coordinates and optional parameters."""
    series = _series_tuple(values)
    descriptor = series[0].descriptor

    coordinate_values = tuple(coordinates)
    if len(coordinate_values) != descriptor.num_vars:
        message = f'Expected {descriptor.num_vars} coordinate values, got {len(coordinate_values)}'
        raise ValueError(message)

    if parameters is None:
        parameter_values: tuple[Scalar, ...] = (0.0,) * descriptor.num_params
    else:
        parameter_values = tuple(parameters)
        if len(parameter_values) != descriptor.num_params:
            message = (
                f'Expected {descriptor.num_params} parameter values, got {len(parameter_values)}'
            )
            raise ValueError(message)

    point = (*coordinate_values, *parameter_values)
    complex_ = _is_complex(series) or any(complex(value).imag != 0.0 for value in point)
    if complex_ and not _is_complex(series):
        series = _promote_complex(series)

    if complex_:
        input_array = ffi.new('double _Complex[]', [complex(value) for value in point])
        output_array = ffi.new('double _Complex[]', len(series))
    else:
        input_array = ffi.new('double[]', [complex(value).real for value in point])
        output_array = ffi.new('double[]', len(series))

    _protected_call(
        'map_eval',
        lib.mad_ctpsa_eval if complex_ else lib.mad_tpsa_eval,
        complex_=complex_,
        args=(
            len(series),
            _pointer_array(series),
            len(point),
            input_array,
            output_array,
        ),
    )

    if complex_:
        return np.asarray([complex(output_array[ii]) for ii in range(len(series))])
    return np.asarray([float(output_array[ii]) for ii in range(len(series))])


@overload
def translate(
    values: Sequence[Tpsa] | TpsaMap[Tpsa], offsets: Sequence[SupportsFloat]
) -> tuple[Tpsa, ...]: ...
@overload
def translate(
    values: Sequence[Tpsa] | TpsaMap[Tpsa],
    offsets: Sequence[SupportsComplex],
) -> tuple[ComplexTpsa, ...]: ...
@overload
def translate(
    values: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa], offsets: Sequence[Scalar]
) -> tuple[ComplexTpsa, ...]: ...
@overload
def translate(values: Sequence[Series], offsets: Sequence[Scalar]) -> tuple[Series, ...]: ...


def translate(
    values: Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa], offsets: Sequence[Scalar]
) -> tuple[Series, ...]:
    """Return the map after substituting ``z -> z + offsets``.

    This is implemented through native composition so descriptor parameters remain
    identity parameters. It therefore also works for parametric maps.
    """
    series = _series_tuple(values)
    descriptor = series[0].descriptor
    offsets = tuple(offsets)
    if len(offsets) != descriptor.num_vars:
        message = f'Expected {descriptor.num_vars} offsets, got {len(offsets)}'
        raise ValueError(message)

    complex_ = _is_complex(series) or any(complex(value).imag != 0.0 for value in offsets)
    if complex_ and not _is_complex(series):
        series = _promote_complex(series)

    if complex_:
        substitutions: tuple[Series, ...] = tuple(
            descriptor.var(index, complex(value)) for index, value in enumerate(offsets, start=1)
        )
    else:
        substitutions = tuple(
            descriptor.var(index, complex(value).real)
            for index, value in enumerate(offsets, start=1)
        )
    return compose(series, substitutions)


@overload
def vector_to_field(generator: Tpsa) -> tuple[Tpsa, ...]: ...
@overload
def vector_to_field(generator: ComplexTpsa) -> tuple[ComplexTpsa, ...]: ...
@overload
def vector_to_field(generator: Series) -> tuple[Series, ...]: ...


def vector_to_field(generator: Series) -> tuple[Series, ...]:
    """Convert a scalar generator to MAD-NG's Hamiltonian vector-field form.

    This mirrors ``mad_[c]tpsa_vec2fld`` and returns ``G = -J grad(generator)``.
    """
    if not isinstance(generator, (Tpsa, ComplexTpsa)):
        message = 'generator must be Tpsa or ComplexTpsa'
        raise TypeError(message)
    descriptor = generator.descriptor
    if descriptor.num_vars % 2:
        message = 'Hamiltonian vector fields require an even number of variables'
        raise ValueError(message)

    if isinstance(generator, ComplexTpsa):
        result = tuple(
            descriptor.complex_zero(order=generator.order) for _ in range(descriptor.num_vars)
        )
        complex_ = True
    else:
        result = tuple(descriptor.zero(order=generator.order) for _ in range(descriptor.num_vars))
        complex_ = False

    _protected_call(
        'scalar_to_map',
        lib.mad_ctpsa_vec2fld if complex_ else lib.mad_tpsa_vec2fld,
        complex_=complex_,
        args=(
            descriptor.num_vars,
            generator.ptr,
            _pointer_array(result),
        ),
    )
    return result


@overload
def field_to_vector(field: Sequence[Tpsa] | TpsaMap[Tpsa]) -> Tpsa: ...
@overload
def field_to_vector(field: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa]) -> ComplexTpsa: ...
@overload
def field_to_vector(field: Sequence[Series]) -> Series: ...


def field_to_vector(field: Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]) -> Series:
    """Convert a Hamiltonian vector field to its scalar generator."""
    series = _series_tuple(field)
    descriptor = series[0].descriptor
    if len(series) != descriptor.num_vars:
        message = f'Field must have {descriptor.num_vars} components, got {len(series)}'
        raise ValueError(message)
    if descriptor.num_vars % 2:
        message = 'Hamiltonian vector fields require an even number of variables'
        raise ValueError(message)

    max_order = min(descriptor.order, max(value.order for value in series) + 1)
    if _is_complex(series):
        result: Series = descriptor.complex_zero(order=max_order)
        complex_ = True
    else:
        result = descriptor.zero(order=max_order)
        complex_ = False

    _protected_call(
        'map_to_scalar',
        lib.mad_ctpsa_fld2vec if complex_ else lib.mad_tpsa_fld2vec,
        complex_=complex_,
        args=(
            len(series),
            _pointer_array(series),
            result.ptr,
        ),
    )
    return result


@overload
def lie_bracket(
    left: Sequence[Tpsa] | TpsaMap[Tpsa], right: Sequence[Tpsa] | TpsaMap[Tpsa]
) -> tuple[Tpsa, ...]: ...
@overload
def lie_bracket(
    left: Sequence[Tpsa] | TpsaMap[Tpsa], right: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa]
) -> tuple[ComplexTpsa, ...]: ...
@overload
def lie_bracket(
    left: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa],
    right: (Sequence[Tpsa] | Sequence[ComplexTpsa] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]),
) -> tuple[ComplexTpsa, ...]: ...
@overload
def lie_bracket(
    left: Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
    right: Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
) -> tuple[Series, ...]: ...


def lie_bracket(
    left: Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
    right: Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
) -> tuple[Series, ...]:
    """Return the Lie bracket of two vector fields."""
    left_series, right_series = _coerce_pair(left, right)
    if len(left_series) != len(right_series):
        message = 'Vector fields must have the same number of components'
        raise ValueError(message)

    result = _zeros_like(left_series)
    complex_ = _is_complex(left_series)
    _protected_call(
        'map_pair',
        lib.mad_ctpsa_liebra if complex_ else lib.mad_tpsa_liebra,
        complex_=complex_,
        args=(
            len(left_series),
            _pointer_array(left_series),
            _pointer_array(right_series),
            _pointer_array(result),
        ),
    )
    return result


@overload
def exp_poisson(
    values: Sequence[Tpsa] | TpsaMap[Tpsa], generator: Tpsa | Sequence[Tpsa] | TpsaMap[Tpsa]
) -> tuple[Tpsa, ...]: ...
@overload
def exp_poisson(
    values: Sequence[Tpsa] | TpsaMap[Tpsa],
    generator: (ComplexTpsa | Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa]),
) -> tuple[ComplexTpsa, ...]: ...
@overload
def exp_poisson(
    values: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa],
    generator: (Series | Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]),
) -> tuple[ComplexTpsa, ...]: ...
@overload
def exp_poisson(
    values: Sequence[Series],
    generator: (Series | Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]),
) -> tuple[Series, ...]: ...


def exp_poisson(
    values: Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
    generator: (Series | Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]),
) -> tuple[Series, ...]:
    """Apply MAD-NG's exponential Poisson-bracket map.

    When ``generator`` is a scalar TPSA this mirrors MAD-NG ``damap:exppb(f)``.
    MAD-NG converts the scalar generator as ``-vec2fld(f)`` before applying the
    native map exponential.
    """
    series = _series_tuple(values)
    descriptor = series[0].descriptor
    if len(series) != descriptor.num_vars:
        message = (
            f'exp_poisson requires a full {descriptor.num_vars}-component map, got {len(series)}'
        )
        raise ValueError(message)

    if isinstance(generator, (Tpsa, ComplexTpsa)):
        if generator.descriptor is not descriptor:
            message = 'Generator and map must share the same Descriptor'
            raise ValueError(message)
        complex_ = _is_complex(series) or isinstance(generator, ComplexTpsa)
        if complex_ and not _is_complex(series):
            series = _promote_complex(series)
        if complex_ and isinstance(generator, Tpsa):
            generator = ComplexTpsa.from_tpsa(generator)
        field = tuple(-component for component in vector_to_field(generator))
    else:
        series, field = _coerce_pair(series, generator)
        complex_ = _is_complex(series)
        if len(field) != descriptor.num_vars:
            message = (
                f'Generator field must have {descriptor.num_vars} components, got {len(field)}'
            )
            raise ValueError(message)

    result = _zeros_like(series)
    _protected_call(
        'map_binary',
        lib.mad_ctpsa_exppb if complex_ else lib.mad_tpsa_exppb,
        complex_=complex_,
        args=(
            len(field),
            _pointer_array(field),
            len(series),
            _pointer_array(series),
            _pointer_array(result),
        ),
    )
    return result


def _log_poisson_impl(
    values: Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
    initial_guess: (Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None) = None,
) -> tuple[Series, ...]:
    series = _series_tuple(values)
    descriptor = series[0].descriptor
    if len(series) != descriptor.num_vars:
        err_mess = (
            f'log_poisson requires a full {descriptor.num_vars}-component map, got {len(series)}'
        )
        raise ValueError(err_mess)
    initial_series = None
    if initial_guess is not None:
        series, initial_series = _coerce_pair(series, initial_guess)
        if len(initial_series) != descriptor.num_vars:
            err_mess = (
                f'Field `initial_guess` must have {descriptor.num_vars} components, '
                f'got {len(initial_series)}'
            )
            raise ValueError(err_mess)
    result = _zeros_like(series)
    complex_ = _is_complex(series)
    initial_ptrs = ffi.NULL if initial_series is None else _pointer_array(initial_series)
    _protected_call(
        'map_pair',
        lib.mad_ctpsa_logpb if complex_ else lib.mad_tpsa_logpb,
        complex_=complex_,
        args=(
            len(series),
            _pointer_array(series),
            initial_ptrs,
            _pointer_array(result),
        ),
    )
    return result


@overload
def log_poisson(
    values: Sequence[Tpsa] | TpsaMap[Tpsa],
    initial_guess: Sequence[Tpsa] | TpsaMap[Tpsa] | None = None,
) -> tuple[Tpsa, ...]: ...
@overload
def log_poisson(
    values: Sequence[Tpsa] | TpsaMap[Tpsa],
    initial_guess: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa],
) -> tuple[ComplexTpsa, ...]: ...
@overload
def log_poisson(
    values: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa],
    initial_guess: (Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None) = None,
) -> tuple[ComplexTpsa, ...]: ...
@overload
def log_poisson(
    values: Sequence[Series],
    initial_guess: (Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None) = None,
) -> tuple[Series, ...]: ...


def log_poisson(
    values: Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
    initial_guess: (Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None) = None,
) -> tuple[Series, ...]:
    """Return MAD-NG's vector-field logarithm of a map.

    The result is the vector field accepted by the native ``exppb`` operation,
    not yet the scalar Hamiltonian generator.
    """
    return _log_poisson_impl(values, initial_guess)


@overload
def log_generator(
    values: Sequence[Tpsa] | TpsaMap[Tpsa],
    initial_guess: Sequence[Tpsa] | TpsaMap[Tpsa] | None = None,
) -> Tpsa: ...
@overload
def log_generator(
    values: Sequence[Tpsa] | TpsaMap[Tpsa],
    initial_guess: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa],
) -> ComplexTpsa: ...
@overload
def log_generator(
    values: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa],
    initial_guess: (Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None) = None,
) -> ComplexTpsa: ...
@overload
def log_generator(
    values: Sequence[Series],
    initial_guess: (Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None) = None,
) -> Series: ...


def log_generator(
    values: Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
    initial_guess: (Sequence[Series] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None) = None,
) -> Series:
    """Return the scalar generator corresponding to :func:`log_poisson`.

    The minus sign mirrors MAD-NG's high-level ``damap:exppb(f)`` convention,
    which converts a scalar ``f`` to ``-vec2fld(f)``.
    """
    return -field_to_vector(_log_poisson_impl(values, initial_guess))


def _pullback_impl(function: Series, map_: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]) -> Series:
    if function.descriptor is not map_.descriptor:
        message = 'Function and map must share the same Descriptor'
        raise ValueError(message)
    if len(map_) != map_.num_vars:
        message = f'Pullback requires a full {map_.num_vars}-component map, got {len(map_)}'
        raise ValueError(message)
    return compose((function,), map_.coords)[0]


@overload
def pullback(function: Tpsa, map_: TpsaMap[Tpsa]) -> Tpsa: ...
@overload
def pullback(function: Tpsa, map_: TpsaMap[ComplexTpsa]) -> ComplexTpsa: ...
@overload
def pullback(function: ComplexTpsa, map_: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]) -> ComplexTpsa: ...
@overload
def pullback(function: Series, map_: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]) -> Series: ...


def pullback(function: Series, map_: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]) -> Series:
    """Return the pullback ``function ∘ map_``."""
    return _pullback_impl(function, map_)


def map_order(values: Sequence[Series] | TpsaMap) -> int:
    """Largest allocated TPSA order among map components."""
    series = _series_tuple(values)
    return max(value.order for value in series)


def max_nonzero_order(values: Sequence[Series] | TpsaMap) -> int:
    """Largest populated TPSA order among map components."""
    series = _series_tuple(values)
    return max(value.max_nonzero_order for value in series)


def norm(values: Sequence[Series] | TpsaMap) -> float:
    """MAD-NG map norm, equal to the sum of component coefficient norms."""
    series = _series_tuple(values)
    return float(sum(value.norm() for value in series))


class TpsaMap(Generic[SeriesT]):
    """A vector-valued TPSA map sharing one :class:`Descriptor`.

    ``TpsaMap`` owns only its component TPSA objects. It deliberately contains
    no accelerator-specific reference-particle or tracking state.
    """

    __slots__ = ('coords', 'coord_names')

    coords: list[SeriesT]
    coord_names: tuple[str, ...]

    @overload
    def __init__(
        self: TpsaMap[Tpsa],
        coords: Sequence[Tpsa] | Mapping[str, Tpsa],
        *,
        coord_names: Sequence[str] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: TpsaMap[ComplexTpsa],
        coords: Sequence[Series] | Mapping[str, Series],
        *,
        coord_names: Sequence[str] | None = None,
    ) -> None: ...

    def __init__(
        self,
        coords: Sequence[Series] | Mapping[str, Series],
        *,
        coord_names: Sequence[str] | None = None,
    ) -> None:
        if isinstance(coords, Mapping):
            if coord_names is None:
                coord_names = tuple(coords)
            coords = tuple(coords[name] for name in coord_names)

        series = _series_tuple(coords)
        if _is_complex(series):
            series = _promote_complex(series)
        self.coords = cast('list[SeriesT]', list(series))

        descriptor = series[0].descriptor
        if len(series) > descriptor.num_vars:
            message = (
                f'Map has {len(series)} components but descriptor has only '
                f'{descriptor.num_vars} variables'
            )
            raise ValueError(message)
        if coord_names is None:
            coord_names = descriptor.var_labels[: len(series)]
        coord_names = tuple(coord_names)
        if len(coord_names) != len(series):
            message = 'coord_names must contain one name per map component'
            raise ValueError(message)
        if len(set(coord_names)) != len(coord_names):
            message = 'coord_names must be unique'
            raise ValueError(message)
        self.coord_names = coord_names

    @classmethod
    def identity(
        cls,
        descriptor,
        *,
        values: Sequence[SupportsFloat] | None = None,
        coord_names: Sequence[str] | None = None,
    ) -> TpsaMap[Tpsa]:
        """Construct the identity TPSA map."""
        # Explicit real Tpsa components
        return TpsaMap(
            descriptor.vars(values),
            coord_names=coord_names,
        )

    @classmethod
    def from_monomial_coeffs(
        cls,
        descriptor,
        coefficients: Sequence[Mapping] | Mapping[str, Mapping],
        *,
        coord_names: Sequence[str] | None = None,
    ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Construct a TPSA map from monomial coefficient dictionaries."""
        if isinstance(coefficients, Mapping):
            if coord_names is None:
                coord_names = tuple(coefficients)
            coords = [descriptor.from_monomial_coeffs(coefficients[name]) for name in coord_names]
        else:
            coords = [descriptor.from_monomial_coeffs(component) for component in coefficients]
        return TpsaMap(coords, coord_names=coord_names)

    def to_complex(self) -> TpsaMap[ComplexTpsa]:
        """Return this map represented by complex TPSAs."""
        return TpsaMap(
            [_as_complex(value) for value in self.coords],
            coord_names=self.coord_names,
        )

    @property
    def descriptor(self):
        """Return the descriptor shared by all coordinates."""
        return self.coords[0].descriptor

    @property
    def order(self) -> int:
        """Return the maximum allocated order of the map."""
        return map_order(self.coords)

    @property
    def max_nonzero_order(self) -> int:
        """Return the highest populated order of the map."""
        return max_nonzero_order(self.coords)

    @property
    def num_vars(self) -> int:
        """Return the number of descriptor variables."""
        return self.descriptor.num_vars

    @property
    def num_params(self) -> int:
        """Return the number of descriptor parameters."""
        return self.descriptor.num_params

    @property
    def is_complex(self) -> bool:
        """Return whether the map contains complex TPSAs."""
        return any(isinstance(value, ComplexTpsa) for value in self.coords)

    @property
    def const_part(self) -> np.ndarray:
        """Return the constant vector of the map."""
        dtype = complex if self.is_complex else float
        return np.asarray([value.const_part for value in self.coords], dtype=dtype)

    def __len__(self) -> int:
        return len(self.coords)

    def __iter__(self) -> Iterator[SeriesT]:
        return iter(self.coords)

    def __getitem__(self, key: int | str) -> SeriesT:
        if isinstance(key, str):
            try:
                key = self.coord_names.index(key)
            except ValueError as exc:
                raise KeyError(key) from exc
        return self.coords[key]

    def __getattr__(self, name: str) -> SeriesT:
        if name in self.coord_names:
            return self[name]
        raise AttributeError(name)

    def __repr__(self) -> str:
        components = ', '.join(
            f'{name}={value!r}' for name, value in zip(self.coord_names, self.coords, strict=True)
        )
        return f'TpsaMap({components})'

    def copy(self) -> TpsaMap[SeriesT]:
        """Return an independent copy of the map."""
        result = TpsaMap(
            [value.copy() for value in self.coords],
            coord_names=self.coord_names,
        )
        return cast('TpsaMap[SeriesT]', result)

    def monomial_coeffs(self, coord: int | str | None = None, tol: float = 1e-14):
        """Return monomial coefficients for one or all coordinates."""
        if coord is not None:
            return self[coord].monomial_coeffs(tol)
        return {
            name: value.monomial_coeffs(tol)
            for name, value in zip(self.coord_names, self.coords, strict=True)
        }

    def coefficient(self, coord: int | str, monomials):
        """Return coefficients from one coordinate."""
        return self[coord].coefficient(monomials)

    def set_coefficient(self, coord: int | str, monomial, value) -> None:
        """Set a coefficient in one coordinate."""
        _set_series_coefficient(self[coord], monomial, value)

    def set_const_part(self, values: Sequence[Scalar]) -> None:
        """Set the constant part of every map component."""
        values = tuple(values)
        if len(values) != len(self):
            message = f'Expected {len(self)} values, got {len(values)}'
            raise ValueError(message)
        for component, value in zip(self.coords, values, strict=True):
            _set_series_const_part(component, value)

    def evaluate(self, coordinates, *, parameters=None) -> np.ndarray:
        """Evaluate the map at numerical coordinates."""
        return evaluate(self.coords, coordinates, parameters=parameters)

    evaluate_array = evaluate

    def jacobian(
        self,
        coordinates: Sequence[Scalar] | None = None,
        *,
        parameters: Sequence[Scalar] | None = None,
    ) -> np.ndarray:
        """Return the Jacobian matrix of the map."""
        if coordinates is None:
            return np.asarray([value.grad() for value in self.coords])

        columns = []
        for variable in range(1, self.descriptor.num_vars + 1):
            derivatives = tuple(value.derivative(variable) for value in self.coords)
            columns.append(evaluate(derivatives, coordinates, parameters=parameters))
        return np.column_stack(columns)

    def param_jacobian(self) -> np.ndarray:
        """Return first-order derivatives with respect to descriptor parameters."""
        return np.asarray([value.param_grad() for value in self.coords])

    def set_jacobian(self, jacobian) -> None:
        """Set the first-order coefficients with respect to descriptor variables."""
        jacobian = np.asarray(jacobian)
        expected_shape = (len(self), self.num_vars)
        if jacobian.shape != expected_shape:
            message = f'Jacobian must have shape {expected_shape}, got {jacobian.shape}'
            raise ValueError(message)
        for component, row in zip(self.coords, jacobian, strict=True):
            for variable, value in enumerate(row):
                monomial = [0] * self.descriptor.monomial_length
                monomial[variable] = 1
                _set_series_coefficient(component, monomial, value)

    def set_param_jacobian(self, jacobian) -> None:
        """Set first-order coefficients with respect to descriptor parameters."""
        jacobian = np.asarray(jacobian)
        expected_shape = (len(self), self.num_params)
        if jacobian.shape != expected_shape:
            message = f'Parameter Jacobian must have shape {expected_shape}, got {jacobian.shape}'
            raise ValueError(message)
        for component, row in zip(self.coords, jacobian, strict=True):
            for parameter, value in enumerate(row):
                monomial = [0] * self.descriptor.monomial_length
                monomial[self.num_vars + parameter] = 1
                _set_series_coefficient(component, monomial, value)

    def sensitivity(self, coord: int | str, parameter: int | str):
        """Return the first-order sensitivity of one output to one parameter."""
        if isinstance(parameter, str):
            try:
                parameter = self.descriptor.param_labels.index(parameter)
            except ValueError as exc:
                raise KeyError(parameter) from exc
        elif not 0 <= parameter < self.num_params:
            raise IndexError(parameter)
        return self[coord].param_grad()[parameter]

    def homogeneous(self, order: int) -> TpsaMap[SeriesT]:
        """Return the homogeneous part of the requested order."""
        result = TpsaMap(
            [value.homogeneous(order) for value in self.coords],
            coord_names=self.coord_names,
        )
        return cast('TpsaMap[SeriesT]', result)

    def truncate(self, order: int) -> TpsaMap[SeriesT]:
        """Return the map truncated through the requested order."""
        result = TpsaMap(
            [value.truncate(order) for value in self.coords],
            coord_names=self.coord_names,
        )
        return cast('TpsaMap[SeriesT]', result)

    def clear_order(self, order: int) -> TpsaMap[SeriesT]:
        """Return a copy with one homogeneous order removed."""
        result = TpsaMap(
            [value.clear_order(order) for value in self.coords],
            coord_names=self.coord_names,
        )
        return cast('TpsaMap[SeriesT]', result)

    @overload
    def compose(self: TpsaMap[Tpsa], other: TpsaMap[Tpsa]) -> TpsaMap[Tpsa]: ...
    @overload
    def compose(self: TpsaMap[Tpsa], other: TpsaMap[ComplexTpsa]) -> TpsaMap[ComplexTpsa]: ...
    @overload
    def compose(
        self: TpsaMap[ComplexTpsa], other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    ) -> TpsaMap[ComplexTpsa]: ...

    def compose(
        self, other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Return the composition ``self ∘ other``."""
        if not isinstance(other, TpsaMap):
            return NotImplemented
        return TpsaMap(
            compose(self.coords, other.coords),
            coord_names=self.coord_names,
        )

    __matmul__ = compose

    def inverse(self) -> TpsaMap[SeriesT]:
        """Return the formal inverse of the map."""
        if np.any(self.const_part != 0):
            err_mess = (
                'Map inversion requires a zero constant part; '
                'translate/recentre the map before inversion'
            )
            raise ValueError(err_mess)
        result = TpsaMap(
            inverse(self.coords),
            coord_names=self.coord_names,
        )
        return cast('TpsaMap[SeriesT]', result)

    def partial_inverse(self, select: Sequence[SelectionValue]) -> TpsaMap[SeriesT]:
        """Return a partial inverse of the map."""
        if np.any(self.const_part != 0):
            err_mess = (
                'Map inversion requires a zero constant part; '
                'translate/recentre the map before inversion'
            )
            raise ValueError(err_mess)
        result = TpsaMap(
            partial_inverse(self.coords, select),
            coord_names=self.coord_names,
        )
        return cast('TpsaMap[SeriesT]', result)

    @overload
    def translate(self: TpsaMap[Tpsa], offsets: Sequence[SupportsFloat]) -> TpsaMap[Tpsa]: ...
    @overload
    def translate(
        self: TpsaMap[Tpsa], offsets: Sequence[SupportsComplex]
    ) -> TpsaMap[ComplexTpsa]: ...
    @overload
    def translate(
        self: TpsaMap[ComplexTpsa], offsets: Sequence[Scalar]
    ) -> TpsaMap[ComplexTpsa]: ...

    def translate(self, offsets: Sequence[Scalar]) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Translate the map arguments."""
        return TpsaMap(
            translate(self.coords, offsets),
            coord_names=self.coord_names,
        )

    @overload
    def lie_bracket(self: TpsaMap[Tpsa], other: TpsaMap[Tpsa]) -> TpsaMap[Tpsa]: ...
    @overload
    def lie_bracket(self: TpsaMap[Tpsa], other: TpsaMap[ComplexTpsa]) -> TpsaMap[ComplexTpsa]: ...
    @overload
    def lie_bracket(
        self: TpsaMap[ComplexTpsa], other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    ) -> TpsaMap[ComplexTpsa]: ...

    def lie_bracket(
        self, other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Return the Lie bracket with another vector field."""
        if not isinstance(other, TpsaMap):
            return NotImplemented
        return TpsaMap(
            lie_bracket(self.coords, other.coords),
            coord_names=self.coord_names,
        )

    @overload
    def exp_poisson(self: TpsaMap[Tpsa], generator: Tpsa) -> TpsaMap[Tpsa]: ...
    @overload
    def exp_poisson(self: TpsaMap[Tpsa], generator: ComplexTpsa) -> TpsaMap[ComplexTpsa]: ...
    @overload
    def exp_poisson(self: TpsaMap[ComplexTpsa], generator: Series) -> TpsaMap[ComplexTpsa]: ...
    @overload
    def exp_poisson(self: TpsaMap[Tpsa], generator: TpsaMap[Tpsa]) -> TpsaMap[Tpsa]: ...
    @overload
    def exp_poisson(
        self: TpsaMap[Tpsa], generator: TpsaMap[ComplexTpsa]
    ) -> TpsaMap[ComplexTpsa]: ...
    @overload
    def exp_poisson(
        self: TpsaMap[ComplexTpsa], generator: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    ) -> TpsaMap[ComplexTpsa]: ...

    def exp_poisson(
        self,
        generator: Series | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
    ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Return the action of the Lie exponential generated by ``generator``.

        In mathematical notation, ``map.exp_poisson(f)`` returns
            exp(:f:) map
        following the same argument ordering as MAD-NG ``damap:exppb(f)``.
        """
        generator_arg: Series | Sequence[Series]
        generator_arg = generator.coords if isinstance(generator, TpsaMap) else generator
        return TpsaMap(
            exp_poisson(self.coords, generator_arg),
            coord_names=self.coord_names,
        )

    @overload
    def log_poisson(self: TpsaMap[Tpsa], initial_guess: None = None) -> TpsaMap[Tpsa]: ...
    @overload
    def log_poisson(self: TpsaMap[Tpsa], initial_guess: TpsaMap[Tpsa]) -> TpsaMap[Tpsa]: ...
    @overload
    def log_poisson(
        self: TpsaMap[Tpsa], initial_guess: TpsaMap[ComplexTpsa]
    ) -> TpsaMap[ComplexTpsa]: ...
    @overload
    def log_poisson(
        self: TpsaMap[ComplexTpsa],
        initial_guess: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None = None,
    ) -> TpsaMap[ComplexTpsa]: ...

    def log_poisson(
        self, initial_guess: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None = None
    ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Return the logarithmic Hamiltonian vector field."""
        initial_coords = None if initial_guess is None else initial_guess.coords
        return TpsaMap(
            log_poisson(self.coords, initial_coords),
            coord_names=self.coord_names,
        )

    @overload
    def log_generator(self: TpsaMap[Tpsa], initial_guess: None = None) -> Tpsa: ...
    @overload
    def log_generator(self: TpsaMap[Tpsa], initial_guess: TpsaMap[Tpsa]) -> Tpsa: ...
    @overload
    def log_generator(self: TpsaMap[Tpsa], initial_guess: TpsaMap[ComplexTpsa]) -> ComplexTpsa: ...
    @overload
    def log_generator(
        self: TpsaMap[ComplexTpsa],
        initial_guess: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None = None,
    ) -> ComplexTpsa: ...

    def log_generator(
        self, initial_guess: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None = None
    ) -> Series:
        """Return the scalar logarithmic generator."""
        initial_coords = None if initial_guess is None else initial_guess.coords
        return log_generator(self.coords, initial_coords)

    @overload
    def pullback(self: TpsaMap[Tpsa], function: Tpsa) -> Tpsa: ...
    @overload
    def pullback(self: TpsaMap[Tpsa], function: ComplexTpsa) -> ComplexTpsa: ...
    @overload
    def pullback(self: TpsaMap[ComplexTpsa], function: Series) -> ComplexTpsa: ...

    def pullback(self, function: Series) -> Series:
        """Return ``function ∘ self``."""
        map_ = cast(
            'TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]',
            self,
        )
        return _pullback_impl(function, map_)

    def norm(self) -> float:
        """Return the MAD-NG map norm."""
        return norm(self.coords)

    def __neg__(self) -> TpsaMap[SeriesT]:
        result = TpsaMap(
            [-value for value in self.coords],
            coord_names=self.coord_names,
        )
        return cast('TpsaMap[SeriesT]', result)

    @overload
    def __add__(self: TpsaMap[Tpsa], other: TpsaMap[Tpsa]) -> TpsaMap[Tpsa]: ...
    @overload
    def __add__(self: TpsaMap[Tpsa], other: TpsaMap[ComplexTpsa]) -> TpsaMap[ComplexTpsa]: ...
    @overload
    def __add__(
        self: TpsaMap[ComplexTpsa], other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    ) -> TpsaMap[ComplexTpsa]: ...

    def __add__(
        self, other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        if not isinstance(other, TpsaMap):
            return NotImplemented
        left, right = _coerce_pair(self.coords, other.coords)
        if len(left) != len(right):
            message = 'Maps must have the same number of components'
            raise ValueError(message)
        return TpsaMap(
            [a + b for a, b in zip(left, right, strict=True)],
            coord_names=self.coord_names,
        )

    @overload
    def __sub__(self: TpsaMap[Tpsa], other: TpsaMap[Tpsa]) -> TpsaMap[Tpsa]: ...
    @overload
    def __sub__(self: TpsaMap[Tpsa], other: TpsaMap[ComplexTpsa]) -> TpsaMap[ComplexTpsa]: ...
    @overload
    def __sub__(
        self: TpsaMap[ComplexTpsa], other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    ) -> TpsaMap[ComplexTpsa]: ...

    def __sub__(
        self, other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        if not isinstance(other, TpsaMap):
            return NotImplemented
        left, right = _coerce_pair(self.coords, other.coords)
        if len(left) != len(right):
            message = 'Maps must have the same number of components'
            raise ValueError(message)
        return TpsaMap(
            [a - b for a, b in zip(left, right, strict=True)],
            coord_names=self.coord_names,
        )

    @overload
    def __mul__(self: TpsaMap[Tpsa], scalar: SupportsFloat) -> TpsaMap[Tpsa]: ...
    @overload
    def __mul__(self: TpsaMap[Tpsa], scalar: SupportsComplex) -> TpsaMap[ComplexTpsa]: ...
    @overload
    def __mul__(self: TpsaMap[ComplexTpsa], scalar: Scalar) -> TpsaMap[ComplexTpsa]: ...

    def __mul__(self, scalar: Scalar) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        if not isinstance(scalar, (SupportsFloat, SupportsComplex)):
            return NotImplemented
        return TpsaMap(
            [scalar * value for value in self.coords],
            coord_names=self.coord_names,
        )

    __rmul__ = __mul__
