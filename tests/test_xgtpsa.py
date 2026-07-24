"""Tests for the standalone GTPSA engine bindings. Nothing here imports xtrack."""

import numpy as np
import pytest

import xgtpsa

pytestmark = pytest.mark.skipif(
    not xgtpsa.have_core(), reason="madng_tpsa shared library unavailable; build package"
)


def test_no_xtrack_needed():
    # In a fresh interpreter: the whole engine works with xtrack never imported.
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-c",
         "import sys, xgtpsa;"
         "d = xgtpsa.Descriptor.new(2, 2);"
         "assert (xgtpsa.Tpsa.var(d, 1, 2.0) ** 2).const_part == 4.0;"
         "assert 'xtrack' not in sys.modules"],
        check=True,
    )


def test_paths():
    import os

    assert os.path.exists(xgtpsa.core_library())
    assert os.path.exists(os.path.join(xgtpsa.include_dir(), "madng_tpsa.h"))


def test_lib_and_ffi_singletons():
    assert xgtpsa.lib() is xgtpsa.lib()
    assert xgtpsa.ffi() is xgtpsa.ffi()


# --- descriptors ---------------------------------------------------------- #

def test_descriptor_is_reused():
    a = xgtpsa.Descriptor.new(6, 2)
    b = xgtpsa.Descriptor.new(6, 2)
    assert a is b                                  # GTPSA dedups equivalent descriptors
    assert a != xgtpsa.Descriptor.new(6, 3)
    assert a != xgtpsa.Descriptor.new(4, 2)
    assert a in xgtpsa.live_descriptors()


@pytest.mark.parametrize("nv, order", [(1, 1), (2, 3), (6, 4)])
def test_descriptor_queries_come_from_c(nv, order):
    d = xgtpsa.Descriptor.new(nv, order)
    assert (d.n_variables, d.order) == (nv, order)
    assert (d.n_parameters, d.monomial_length) == (0, nv)


def test_descriptor_order_zero_is_coerced():
    with pytest.warns(UserWarning, match="coerced"):
        assert xgtpsa.Descriptor.new(6, 0).order == 1


def test_descriptor_with_parameters():
    d = xgtpsa.Descriptor.new(6, 2, num_parameters=2, param_order=1)
    assert (d.n_variables, d.n_parameters, d.param_order) == (6, 2, 1)
    assert d.monomial_length == 8
    assert "np=2" in repr(d)


def test_is_valid_monomial():
    # Never call get() beyond the order: GTPSA exit(1)s the interpreter, so this
    # predicate is the only safe way to ask.
    d = xgtpsa.Descriptor.new(2, 2)
    assert d.is_valid_monomial((1, 1))
    assert not d.is_valid_monomial((2, 1))         # total order 3 > 2
    assert not d.is_valid_monomial((1, 1, 1))      # wrong length

def test_live_descriptors_and_wrap():
    d = xgtpsa.Descriptor.new(6, 4)
    live = xgtpsa.live_descriptors()
    assert d in live
    assert all(isinstance(x, xgtpsa.Descriptor) for x in live)
    # the same C pointer always wraps to the same Python object
    assert xgtpsa.descriptor._wrap_desc(d.ptr) is d

# --- series --------------------------------------------------------------- #

def test_var_is_an_identity_seed():
    d = xgtpsa.Descriptor.new(6, 2)
    t = xgtpsa.Tpsa.var(d, 2, 0.25)                # variable indices are 1-based
    assert t.const_part == 0.25
    assert t.grad() == [0, 1, 0, 0, 0, 0]
    assert t.descriptor is d
    assert t.order == d.order


def test_new_series_is_zero():
    d = xgtpsa.Descriptor.new(3, 2)
    assert xgtpsa.Tpsa(d).monomial_coeffs() == {}
    assert xgtpsa.Tpsa(d).const_part == 0.0
    assert xgtpsa.Tpsa(d).grad() == [0.0] * d.n_variables


def test_arithmetic_carries_derivatives():
    d = xgtpsa.Descriptor.new(2, 3)
    x = xgtpsa.Tpsa.var(d, 1, 0.5)
    y = xgtpsa.Tpsa.var(d, 2, 2.0)

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


def test_division_by_series():
    d = xgtpsa.Descriptor.new(1, 2)
    x = xgtpsa.Tpsa.var(d, 1, 2.0)
    q = (x * x) / x
    assert q.const_part == pytest.approx(2.0)
    assert q.grad() == pytest.approx([1.0])


def test_copy_is_independent():
    d = xgtpsa.Descriptor.new(1, 1)
    x = xgtpsa.Tpsa.var(d, 1, 1.0)
    c = x.copy()
    x.set_const_part(9.0)
    assert (c.const_part, x.const_part) == (1.0, 9.0)


def test_set_and_get_coefficients():
    d = xgtpsa.Descriptor.new(2, 2)
    t = xgtpsa.Tpsa(d)
    t.set_const_part(1.5)
    t.set((1, 1), -2.0)
    assert t.const_part == 1.5
    assert t.get((1, 1)) == -2.0
    assert t.coefficient((1, 1)) == -2.0
    np.testing.assert_allclose(t.coefficient([(0, 0), (1, 1)]), [1.5, -2.0])
    with pytest.raises(ValueError):
        t.coefficient(np.zeros((1, 2, 2), dtype=int))


def test_monomial_coeffs_skips_zeros_and_tiny_terms():
    d = xgtpsa.Descriptor.new(2, 2)
    t = xgtpsa.Tpsa(d)
    t.set((1, 0), 1e-20)
    t.set((0, 1), 1.0)
    assert t.monomial_coeffs() == {(0, 1): 1.0}
    assert (1, 0) in t.monomial_coeffs(tol=1e-30)


# --- parameters ----------------------------------------------------------- #

def test_param_seed_and_param_grad():
    d = xgtpsa.Descriptor.new(2, 2, num_parameters=2, param_order=1)
    x = xgtpsa.Tpsa.var(d, 1, 0.5)
    k = xgtpsa.Tpsa.param(d, 1, 3.0)

    assert k.const_part == 3.0
    assert k.param_grad() == [1.0, 0.0]
    assert k.grad() == [0.0, 0.0]                  # a parameter is not a variable

    f = k * x
    assert f.grad() == pytest.approx([3.0, 0.0])   # d(kx)/dx = k
    assert f.get((1, 0, 1, 0)) == pytest.approx(1.0)   # mixed d2/dx dk
