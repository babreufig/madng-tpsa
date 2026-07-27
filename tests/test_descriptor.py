"""Tests for GTPSA descriptors."""

import pytest

import xgtpsa


def test_descriptor_is_reused():
    a = xgtpsa.Descriptor(6, 2)
    b = xgtpsa.Descriptor(6, 2)
    assert a is b  # GTPSA dedups equivalent descriptors
    assert a != xgtpsa.Descriptor(6, 3)
    assert a != xgtpsa.Descriptor(4, 2)
    assert a in xgtpsa.Descriptor._instances_by_ptr.values()


@pytest.mark.parametrize("nv, order", [(1, 1), (2, 3), (6, 4)])
def test_descriptor_queries_come_from_c(nv, order):
    d = xgtpsa.Descriptor(nv, order)
    assert (d.num_vars, d.order) == (nv, order)
    assert (d.num_params, d.monomial_length) == (0, nv)


def test_descriptor_order_must_be_positive():
    with pytest.raises(ValueError, match="Descriptor order must be positive"):
        xgtpsa.Descriptor(6, 0)


def test_descriptor_with_parameters():
    d = xgtpsa.Descriptor(6, 2, num_params=2, param_order=1)
    assert (d.num_vars, d.num_params, d.param_order) == (6, 2, 1)
    assert d.monomial_length == 8
    assert "num_params=2" in repr(d)


def test_descriptor_param_order_must_be_positive():
    with pytest.raises(ValueError, match="Descriptor parameter order must be positive"):
        xgtpsa.Descriptor(6, 2, num_params=2, param_order=0)


def test_is_valid_monomial():
    # Never call get() beyond the order: GTPSA exit(1)s the interpreter, so this
    # predicate is the only safe way to ask.
    d = xgtpsa.Descriptor(2, 2)
    assert d.is_valid_monomial((1, 1))
    assert not d.is_valid_monomial((2, 1))  # total order 3 > 2
    assert not d.is_valid_monomial((1, 1, 1))  # wrong length


def test_descriptor_instances_and_from_ptr():
    d = xgtpsa.Descriptor(6, 4)
    live = xgtpsa.Descriptor._instances_by_ptr.values()
    assert d in live
    assert all(isinstance(x, xgtpsa.Descriptor) for x in live)
    # the same C pointer always wraps to the same Python object
    assert xgtpsa.Descriptor.from_ptr(d.ptr) is d
