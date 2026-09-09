"""Tests for complex TPSA series."""

from decimal import Decimal
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import scipy.special

import madng_tpsa

if TYPE_CHECKING:
    from typing import Any


def test_complex_variable_and_parameter_seeds():
    d = madng_tpsa.Descriptor(variables=['x', 'y'], order=3, params=['k'])
    x = d.complex_var('x', 1 + 2j)
    y = d.complex_var('y')
    k = d.complex_param('k', 3 - 4j)

    assert x.const_part == pytest.approx(1 + 2j)
    assert x.grad() == pytest.approx([1, 0])
    assert y.grad() == pytest.approx([0, 1])
    assert k.const_part == pytest.approx(3 - 4j)
    assert k.param_grad() == pytest.approx([1])


def test_complex_coefficients_and_copy_are_independent():
    d = madng_tpsa.Descriptor(2, 2)
    t = d.complex_zero()
    t.set_const_part(1 + 2j)
    t[(1, 1)] = -3 + 4j

    copied = t.copy()
    t.set_const_part(9j)

    assert copied.to_dict() == pytest.approx({(0, 0): 1 + 2j, (1, 1): -3 + 4j})
    assert t.coefficient((1, 1)) == pytest.approx(-3 + 4j)
    np.testing.assert_allclose(t.coefficient([(0, 0), (1, 1)]), [9j, -3 + 4j])


def test_from_tpsa_promotes_real_tpsas():
    d = madng_tpsa.Descriptor(1, 2)
    real = d.var(1, 1.0)
    imag = d.var(1, 2.0)

    promoted = madng_tpsa.ComplexTpsa.from_tpsa(real)
    combined = madng_tpsa.ComplexTpsa.from_tpsa(real, imag)

    assert promoted.real().equals(real)
    assert promoted.imag().is_zero()
    assert combined.real().equals(real)
    assert combined.imag().equals(imag)

    with pytest.raises(ValueError, match='Incompatible TPSA descriptors'):
        madng_tpsa.ComplexTpsa.from_tpsa(real, madng_tpsa.Descriptor(2, 2).var(1))


def test_complex_arithmetic_and_mixed_real_operands():
    d = madng_tpsa.Descriptor(1, 3)
    x = d.var(1, 1.0)
    z = d.complex_var(1, 1 + 2j)

    assert (z + x).const_part == pytest.approx(2 + 2j)
    assert (x + z).const_part == pytest.approx(2 + 2j)
    assert (z - x).const_part == pytest.approx(2j)
    assert (x - z).const_part == pytest.approx(-2j)
    assert (z * x).const_part == pytest.approx(1 + 2j)
    assert (z / x).const_part == pytest.approx(1 + 2j)
    assert (x / z).const_part == pytest.approx((1 - 2j) / 5)
    assert (z * (2 - 3j)).const_part == pytest.approx(8 + 1j)
    assert (z / (2 - 3j)).const_part == pytest.approx((-4 + 7j) / 13)
    assert ((2 - 3j) / z).const_part == pytest.approx((-4 - 7j) / 5)


def test_complex_power_conjugate_and_real_imaginary_parts():
    d = madng_tpsa.Descriptor(1, 2)
    z = d.complex_var(1, 1 + 2j)

    assert (z**2).const_part == pytest.approx(-3 + 4j)
    assert z.conjugate().const_part == pytest.approx(1 - 2j)
    assert z.real().const_part == pytest.approx(1)
    assert z.imag().const_part == pytest.approx(2)
    assert z.real().grad() == pytest.approx([1])
    assert z.imag().grad() == pytest.approx([0])


def test_complex_differential_algebra():
    d = madng_tpsa.Descriptor(variables=['x', 'y'], order=4)
    x, y = d.complex_vars()
    f = (1 + 2j) * x * x * y

    assert f.derivative('x').get((1, 1)) == pytest.approx(2 + 4j)
    assert f.derivative((2, 1)).const_part == pytest.approx(2 + 4j)
    assert f.integrate('x').get((3, 1)) == pytest.approx((1 + 2j) / 3)
    assert x.poisson_bracket(y).const_part == pytest.approx(1)


def test_complex_elementary_functions_and_numpy_dispatch():
    d = madng_tpsa.Descriptor(1, 2)
    z = d.complex_constant(1 + 2j)

    assert z.exp().const_part == pytest.approx(np.exp(1 + 2j))
    assert z.log().const_part == pytest.approx(np.log(1 + 2j))
    sin_z = cast('madng_tpsa.ComplexTpsa', np.sin(z))
    conjugate_z = cast('madng_tpsa.ComplexTpsa', np.conjugate(z))
    assert sin_z.const_part == pytest.approx(np.sin(1 + 2j))
    assert conjugate_z.const_part == pytest.approx(1 - 2j)
    assert cast('Any', scipy.special.erf)(z).const_part == pytest.approx(
        cast('Any', scipy.special.erf)(1 + 2j)
    )


def test_complex_operations_reject_incompatible_descriptors():
    x = madng_tpsa.Descriptor(1, 2).complex_var(1)
    y = madng_tpsa.Descriptor(2, 2).complex_var(1)

    with pytest.raises(ValueError, match='Incompatible TPSA descriptors'):
        x + y


def test_tpsa_arithmetic_dunder_dispatches_are_exhaustive():
    d = madng_tpsa.Descriptor(1, 2)
    t = d.var(1, 2.0)
    other_t = d.constant(3.0)

    for method_name in ('__add__', '__sub__', '__mul__', '__truediv__'):
        method = getattr(t, method_name)
        assert isinstance(method(other_t), madng_tpsa.Tpsa)
        assert isinstance(method(Decimal('2')), madng_tpsa.Tpsa)
        assert isinstance(method(1j), madng_tpsa.ComplexTpsa)
        assert method(object()) is NotImplemented

    for method_name in ('__rsub__', '__rtruediv__'):
        method = getattr(t, method_name)
        assert isinstance(method(Decimal('2')), madng_tpsa.Tpsa)
        assert isinstance(method(1j), madng_tpsa.ComplexTpsa)
        assert method(object()) is NotImplemented

    assert isinstance(t.__pow__(other_t), madng_tpsa.Tpsa)
    assert isinstance(t.__pow__(2), madng_tpsa.Tpsa)
    assert isinstance(t.__pow__(Decimal('0.5')), madng_tpsa.Tpsa)
    assert isinstance(t.__pow__(1j), madng_tpsa.ComplexTpsa)
    assert t.__pow__(cast('Any', object())) is NotImplemented
    assert isinstance(t.__rpow__(Decimal('2')), madng_tpsa.Tpsa)
    assert isinstance(t.__rpow__(1j), madng_tpsa.ComplexTpsa)
    assert t.__rpow__(cast('Any', object())) is NotImplemented
    assert isinstance(+t, madng_tpsa.Tpsa)
    assert isinstance(-t, madng_tpsa.Tpsa)
    assert isinstance(abs(t), madng_tpsa.Tpsa)
    assert repr(t).startswith('Tpsa(')
    assert t == t.copy()
    assert t != object()
    assert t.__lt__(Decimal('3'))
    assert t.__le__(Decimal('2'))
    assert t.__gt__(Decimal('1'))
    assert t.__ge__(Decimal('2'))
    assert t.__array_ufunc__(np.add, 'reduce', t) is NotImplemented
    assert t.__array_ufunc__(np.floor, '__call__', t) is NotImplemented
    assert isinstance(np.sinc(cast('Any', t)), madng_tpsa.Tpsa)
    assert t.__array_function__(np.cos, (), (), {}) is NotImplemented

    with pytest.raises(ZeroDivisionError):
        t.__truediv__(0)
    with pytest.raises(ZeroDivisionError):
        d.zero().__rtruediv__(1)
    with pytest.raises(TypeError):
        float(cast('Any', t))
    with pytest.raises(ValueError, match='Incompatible TPSA descriptors'):
        t.__pow__(madng_tpsa.Descriptor(1, 3).constant(2.0))


def test_complex_tpsa_arithmetic_dunder_dispatches_are_exhaustive():
    d = madng_tpsa.Descriptor(1, 2)
    t = d.complex_var(1, 2 + 1j)
    other_t = d.var(1, 3.0)
    other_c = d.complex_constant(3 + 2j)

    for method_name in ('__add__', '__sub__', '__mul__', '__truediv__'):
        method = getattr(t, method_name)
        assert isinstance(method(other_c), madng_tpsa.ComplexTpsa)
        assert isinstance(method(other_t), madng_tpsa.ComplexTpsa)
        assert isinstance(method(Decimal('2')), madng_tpsa.ComplexTpsa)
        assert method(object()) is NotImplemented

    for method_name in ('__rsub__', '__rtruediv__'):
        method = getattr(t, method_name)
        assert isinstance(method(other_t), madng_tpsa.ComplexTpsa)
        assert isinstance(method(Decimal('2')), madng_tpsa.ComplexTpsa)
        assert method(object()) is NotImplemented

    assert isinstance(t.__pow__(other_c), madng_tpsa.ComplexTpsa)
    assert isinstance(t.__pow__(other_t), madng_tpsa.ComplexTpsa)
    assert isinstance(t.__pow__(2), madng_tpsa.ComplexTpsa)
    assert isinstance(t.__pow__(Decimal('0.5')), madng_tpsa.ComplexTpsa)
    assert t.__pow__(cast('Any', object())) is NotImplemented
    assert isinstance(t.__rpow__(other_t), madng_tpsa.ComplexTpsa)
    assert isinstance(t.__rpow__(Decimal('2')), madng_tpsa.ComplexTpsa)
    assert t.__rpow__(object()) is NotImplemented
    assert isinstance(+t, madng_tpsa.ComplexTpsa)
    assert isinstance(-t, madng_tpsa.ComplexTpsa)
    assert repr(t).startswith('ComplexTpsa(')
    assert t == t.copy()
    assert t != object()
    assert t.__array_ufunc__(np.add, 'reduce', t) is NotImplemented
    assert t.__array_ufunc__(np.floor, '__call__', t) is NotImplemented

    with pytest.raises(ZeroDivisionError):
        t.__truediv__(0)
    with pytest.raises(ZeroDivisionError):
        d.complex_zero().__rtruediv__(1)
    with pytest.raises(TypeError):
        complex(cast('Any', t))
