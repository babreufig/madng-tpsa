"""Tests for TPSA series."""

import numpy as np
import pytest

import xgtpsa


def test_var_is_an_identity_seed():
    d = xgtpsa.Descriptor(6, 2)
    t = d.var(2, 0.25)  # variable indices are 1-based
    assert t.const_part == 0.25
    assert t.grad() == [0, 1, 0, 0, 0, 0]
    assert t.descriptor is d
    assert t.order == d.order


def test_vars_unpack_all_identity_seeds():
    d = xgtpsa.Descriptor(4, 2)
    x, px, y, py = d.vars()

    assert x.grad() == [1, 0, 0, 0]
    assert px.grad() == [0, 1, 0, 0]
    assert y.grad() == [0, 0, 1, 0]
    assert py.grad() == [0, 0, 0, 1]


def test_vars_can_set_expansion_values():
    d = xgtpsa.Descriptor(2, 2)
    x, y = d.vars(values=[0.5, 2.0])

    assert x.const_part == 0.5
    assert y.const_part == 2.0
    assert x.grad() == [1, 0]
    assert y.grad() == [0, 1]


def test_vars_values_must_match_num_vars():
    d = xgtpsa.Descriptor(2, 2)

    with pytest.raises(ValueError, match="values must contain one entry per variable"):
        d.vars(values=[0.5])


def test_constant_series():
    d = xgtpsa.Descriptor(3, 2)
    zero = d.zero()
    const = d.constant(1.5)

    assert zero.monomial_coeffs() == {}
    assert zero.const_part == 0.0
    assert zero.grad() == [0.0] * d.num_vars
    assert const.const_part == 1.5
    assert const.grad() == [0.0] * d.num_vars


def test_arithmetic_carries_derivatives():
    d = xgtpsa.Descriptor(2, 3)
    x = d.var(1, 0.5)
    y = d.var(2, 2.0)

    f = x * x + 2.0 * y
    assert f.const_part == pytest.approx(0.25 + 4.0)
    assert f.grad() == pytest.approx([2 * 0.5, 2.0])
    assert f.monomial_coeffs()[(2, 0)] == pytest.approx(1.0)

    # scalars mix on either side, and the reflected ops are not the mirrored ones
    assert (3.0 - x).const_part == pytest.approx(2.5)
    assert (1.0 / y).const_part == pytest.approx(0.5)
    assert (-x).const_part == pytest.approx(-0.5)
    assert (x / 2.0).const_part == pytest.approx(0.25)
    assert (x**2).const_part == pytest.approx(0.25)


def test_addition_paths():
    d = xgtpsa.Descriptor(2, 2)
    x = d.var(1, 0.5)
    y = d.var(2, 2.0)

    xy = x + y
    xs = x + 3.0
    sx = 3.0 + x

    assert xy.const_part == pytest.approx(2.5)
    assert xy.grad() == pytest.approx([1.0, 1.0])
    assert xs.const_part == pytest.approx(3.5)
    assert xs.grad() == pytest.approx([1.0, 0.0])
    assert sx.const_part == pytest.approx(3.5)
    assert sx.grad() == pytest.approx([1.0, 0.0])


def test_subtraction_paths():
    d = xgtpsa.Descriptor(2, 2)
    x = d.var(1, 0.5)
    y = d.var(2, 2.0)

    xy = x - y
    xs = x - 3.0
    sx = 3.0 - x

    assert xy.const_part == pytest.approx(-1.5)
    assert xy.grad() == pytest.approx([1.0, -1.0])
    assert xs.const_part == pytest.approx(-2.5)
    assert xs.grad() == pytest.approx([1.0, 0.0])
    assert sx.const_part == pytest.approx(2.5)
    assert sx.grad() == pytest.approx([-1.0, 0.0])


def test_multiplication_paths():
    d = xgtpsa.Descriptor(2, 2)
    x = d.var(1, 0.5)
    y = d.var(2, 2.0)

    xy = x * y
    xs = x * 3.0
    sx = 3.0 * x

    assert xy.const_part == pytest.approx(1.0)
    assert xy.grad() == pytest.approx([2.0, 0.5])
    assert xy.get((1, 1)) == pytest.approx(1.0)
    assert xs.const_part == pytest.approx(1.5)
    assert xs.grad() == pytest.approx([3.0, 0.0])
    assert sx.const_part == pytest.approx(1.5)
    assert sx.grad() == pytest.approx([3.0, 0.0])


def test_division_paths():
    d = xgtpsa.Descriptor(1, 2)
    x = d.var(1, 2.0)

    quotient = (x * x) / x
    scaled = x / 2.0
    inverse = 1.0 / x

    assert quotient.const_part == pytest.approx(2.0)
    assert quotient.grad() == pytest.approx([1.0])
    assert scaled.const_part == pytest.approx(1.0)
    assert scaled.grad() == pytest.approx([0.5])
    assert inverse.const_part == pytest.approx(0.5)
    assert inverse.grad() == pytest.approx([-0.25])
    assert inverse.get((2,)) == pytest.approx(0.125)


def test_power_and_negation():
    d = xgtpsa.Descriptor(1, 3)
    x = d.var(1, 0.5)

    square = x**2
    cube = x**3
    neg = -x

    assert square.const_part == pytest.approx(0.25)
    assert square.grad() == pytest.approx([1.0])
    assert square.get((2,)) == pytest.approx(1.0)
    assert cube.const_part == pytest.approx(0.125)
    assert cube.grad() == pytest.approx([0.75])
    assert cube.get((2,)) == pytest.approx(1.5)
    assert cube.get((3,)) == pytest.approx(1.0)
    assert neg.const_part == pytest.approx(-0.5)
    assert neg.grad() == pytest.approx([-1.0])


def test_division_by_series():
    d = xgtpsa.Descriptor(1, 2)
    x = d.var(1, 2.0)
    q = (x * x) / x
    assert q.const_part == pytest.approx(2.0)
    assert q.grad() == pytest.approx([1.0])


def test_binary_ops_reject_incompatible_descriptors():
    x = xgtpsa.Descriptor(1, 2).var(1)
    y = xgtpsa.Descriptor(2, 2).var(1)

    with pytest.raises(ValueError, match="Incompatible TPSA descriptors"):
        x + y


def test_copy_is_independent():
    d = xgtpsa.Descriptor(1, 1)
    x = d.var(1, 1.0)
    c = x.copy()
    x.set_const_part(9.0)
    assert (c.const_part, x.const_part) == (1.0, 9.0)


def test_set_and_get_coefficients():
    d = xgtpsa.Descriptor(2, 2)
    t = d.zero()
    t.set_const_part(1.5)
    t.set((1, 1), -2.0)
    assert t.const_part == 1.5
    assert t.get((1, 1)) == -2.0
    assert t.coefficient((1, 1)) == -2.0
    np.testing.assert_allclose(t.coefficient([(0, 0), (1, 1)]), [1.5, -2.0])
    with pytest.raises(ValueError):
        t.coefficient(np.zeros((1, 2, 2), dtype=int))


def test_monomial_coeffs_skips_zeros_and_tiny_terms():
    d = xgtpsa.Descriptor(2, 2)
    t = d.zero()
    t.set((1, 0), 1e-20)
    t.set((0, 1), 1.0)
    assert t.monomial_coeffs() == {(0, 1): 1.0}
    assert (1, 0) in t.monomial_coeffs(tol=1e-30)


def test_param_seed_and_param_grad():
    d = xgtpsa.Descriptor(2, 2, num_params=2, param_order=1)
    x = d.var(1, 0.5)
    k = d.param(1, 3.0)

    assert k.const_part == 3.0
    assert k.param_grad() == [1.0, 0.0]
    assert k.grad() == [0.0, 0.0]  # a parameter is not a variable

    f = k * x
    assert f.grad() == pytest.approx([3.0, 0.0])  # d(kx)/dx = k
    assert f.get((1, 0, 1, 0)) == pytest.approx(1.0)  # mixed d2/dx dk


def test_params_unpack_all_identity_seeds():
    d = xgtpsa.Descriptor(2, 2, num_params=2, param_order=1)
    k1, k2 = d.params()

    assert k1.param_grad() == [1, 0]
    assert k2.param_grad() == [0, 1]
