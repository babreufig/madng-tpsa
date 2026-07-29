"""Tests for packaged CFFI bindings."""

from pathlib import Path

import xgtpsa


def test_paths():
    assert Path(xgtpsa.core_library()).exists()


def test_lib_and_ffi_singletons():
    assert xgtpsa.lib() is xgtpsa.lib()
    assert xgtpsa.ffi() is xgtpsa.ffi()
