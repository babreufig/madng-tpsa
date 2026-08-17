"""Tests for packaged CFFI bindings."""

from pathlib import Path

import madng_tpsa


def test_paths():
    assert Path(madng_tpsa.core_library()).exists()


def test_lib_and_ffi_singletons():
    assert madng_tpsa.lib() is madng_tpsa.lib()
    assert madng_tpsa.ffi() is madng_tpsa.ffi()
