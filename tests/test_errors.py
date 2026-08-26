"""Tests for converting MAD-NG fatal TPSA errors into Python exceptions."""

import pytest

from madng_tpsa import Descriptor, TpsaError


def test_unary_mad_error_becomes_tpsa_error():
    zero = Descriptor(1, 2).zero()

    with pytest.raises(TpsaError, match=r'invalid domain sqrt'):
        zero.sqrt()


def test_binary_mad_error_becomes_tpsa_error():
    descriptor = Descriptor(1, 2)
    x = descriptor.var(1)

    with pytest.raises(TpsaError, match=r'invalid domain div'):
        (3 + 2 * x) / x


@pytest.mark.parametrize(
    'operation',
    [
        lambda zero: zero.atan2(0),
        lambda zero: zero.hypot(0),
        lambda zero: zero.hypot3(0, 0),
    ],
)
def test_zero_domain_errors_are_tpsa_errors(operation):
    with pytest.raises(TpsaError):
        operation(Descriptor(1, 2).zero())


@pytest.mark.parametrize(
    'operation',
    [
        lambda zero: zero / 0,
        lambda zero: 1 / zero,
    ],
)
def test_scalar_division_by_zero_raises_zero_division_error(operation):
    with pytest.raises(ZeroDivisionError):
        operation(Descriptor(1, 2).zero())


@pytest.mark.parametrize(
    ('method_name', 'value', 'message'),
    [
        ('sqrt', 0.0, 'sqrt'),
        ('log', 0.0, 'log'),
        ('asin', 1.0, 'asin'),
        ('acos', 1.0, 'acos'),
        ('acosh', 1.0, 'acosh'),
        ('atanh', 1.0, 'atanh'),
    ],
)
def test_unary_domain_errors_are_tpsa_errors(method_name, value, message):
    series = Descriptor(1, 2).constant(value)

    with pytest.raises(TpsaError, match=rf'invalid domain {message}'):
        getattr(series, method_name)()
