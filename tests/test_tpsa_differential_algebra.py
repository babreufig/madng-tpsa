"""Tests for TPSA differential algebra operations."""

import pytest

import xgtpsa


def test_integrate_accepts_index_or_identity_variable():
    d = xgtpsa.Descriptor(2, 3)
    x, y = d.vars()
    f = x * x + 3.0 * y

    by_index = f.integrate(1)
    by_variable = f.integrate(x)

    assert by_index == by_variable
    assert by_index.get((3, 0)) == pytest.approx(1.0 / 3.0)
    assert by_index.get((1, 1)) == pytest.approx(3.0)


def test_integrate_accepts_labels():
    d = xgtpsa.Descriptor(variables=['x', 'y'], order=3)
    x, y = d.vars()
    f = x * x + 3.0 * y

    assert f.integrate('x') == f.integrate(x)
    assert f.integrate('y') == f.integrate(y)


def test_integrate_rejects_non_identity_tpsa_variable():
    d = xgtpsa.Descriptor(1, 2)
    x = d.var(1)

    with pytest.raises(ValueError, match='identity variable'):
        x.integrate(2.0 * x)


def test_derivative_accepts_index_identity_variable_tuple_or_single_monomial_tpsa():
    d = xgtpsa.Descriptor(2, 3)
    x, y = d.vars()
    f = x * x * y

    assert f.derivative(1) == f.derivative(x)
    assert f.derivative(1).get((1, 1)) == pytest.approx(2.0)

    mixed = f.derivative((2, 1))
    assert mixed.const_part == pytest.approx(2.0)

    monomial = d.zero()
    monomial.set((2, 1), 7.0)
    assert f.derivative(monomial) == mixed


def test_derivative_accepts_labels():
    d = xgtpsa.Descriptor(variables=['x', 'y'], order=3, params=['k'])
    x = d.var('x')
    y = d.var('y')
    k = d.param('k')
    f = x * x * y + 4.0 * k

    assert f.derivative('x') == f.derivative(x)
    assert f.derivative('y') == f.derivative(y)
    assert f.derivative('k') == f.derivative(k)


def test_derivative_rejects_invalid_monomials():
    d = xgtpsa.Descriptor(2, 2)
    x, y = d.vars()

    with pytest.raises(ValueError, match='Derivative monomial must have positive order'):
        x.derivative((0, 0))
    with pytest.raises(ValueError, match='Derivative monomial must have length'):
        x.derivative((1,))
    with pytest.raises(ValueError, match='Derivative monomial is not valid'):
        x.derivative((3, 0))
    with pytest.raises(ValueError, match='exactly one'):
        x.derivative(x + y)


def test_poisson_bracket_uses_canonical_pairs():
    d = xgtpsa.Descriptor(2, 2)
    q, p = d.vars()

    assert q.poisson_bracket(p).const_part == pytest.approx(1.0)
    assert p.poisson_bracket(q).const_part == pytest.approx(-1.0)
    assert q.poisson_bracket(p, num_pairs='all').const_part == pytest.approx(1.0)
    assert q.poisson_bracket(p, num_pairs=1).const_part == pytest.approx(1.0)


def test_poisson_bracket_validates_num_pairs():
    d = xgtpsa.Descriptor(2, 2)
    q, p = d.vars()

    with pytest.raises(ValueError, match='num_pairs'):
        q.poisson_bracket(p, num_pairs=0)
    with pytest.raises(ValueError, match='num_pairs'):
        q.poisson_bracket(p, num_pairs=2)
    with pytest.raises(ValueError, match='num_pairs'):
        q.poisson_bracket(p, num_pairs='bad')  # ty: ignore[invalid-argument-type]
