"""cffi ABI binding to ``libgtpsa_core.so`` (the ``mad_*`` engine).

ABI mode (``ffi.dlopen``) against the prebuilt shared object, so nothing is compiled
here. Loading is lazy: importing ``xgtpsa`` never fails, the first call does.

The cdef below is the engine ABI, manually maintained.
Add a line when a new ``mad_*`` function is needed.
TPSA handles stay opaque ``void*``.
(MAD-NG) C conventions: ``setvar`` variable indices start at 1,
and ``mad_tpsa_dflt = 255`` is the full descriptor order.
"""

from __future__ import annotations

from typing import Any

import cffi

from .paths import core_library

CDEF = r"""
    void* mad_desc_newv(int nv, unsigned char mo);
    void* mad_desc_newvp(int nv, unsigned char mo, int np, unsigned char po);
    int   mad_desc_getnv(const void* d, unsigned char* mo_, int* np_, unsigned char* po_);
    _Bool mad_desc_isvalidm(const void* d, int n, const unsigned char* m);
    void* mad_tpsa_newd(void* desc, unsigned char mo);
    const void* mad_tpsa_desc(const void* t);
    unsigned char mad_tpsa_ord(const void* t, _Bool hi);
    void  mad_tpsa_del(void* t);
    void  mad_tpsa_setvar(void* t, double v, int iv, double scl);
    void  mad_tpsa_setprm(void* t, double v, int ip);
    void  mad_tpsa_setval(void* t, double v);
    double mad_tpsa_geti(void* t, int i);
    double mad_tpsa_getm(void* t, int n, const unsigned char* m);
    void  mad_tpsa_seti(void* t, int i, double a, double b);
    void  mad_tpsa_setm(void* t, int n, const unsigned char* m, double a, double b);
    int    mad_tpsa_cycle(void* t, int i, int n, unsigned char* m, double* v);
    void  mad_tpsa_copy(const void* t, void* r);
    void  mad_tpsa_add(const void* a, const void* b, void* c);
    void  mad_tpsa_sub(const void* a, const void* b, void* c);
    void  mad_tpsa_mul(const void* a, const void* b, void* c);
    void  mad_tpsa_div(const void* a, const void* b, void* c);
    void  mad_tpsa_pown(const void* a, double v, void* c);
    void  mad_tpsa_scl(const void* a, double v, void* c);
    void  mad_tpsa_inv(const void* a, double v, void* c);
    void  mad_tpsa_axpb(double a, const void* x, double b, void* r);
"""

_ffi = cffi.FFI()
_ffi.cdef(CDEF)
_lib: Any = None  # the dlopened core; its mad_* members come from the cdef


def lib() -> Any:
    """Dlopen the GTPSA core, keeping one handle for the process.

    The core owns mad's global descriptor state, so consumers that link it (the xtrack
    bridge modules) share the descriptors created here.
    """
    global _lib
    if _lib is None:
        _lib = _ffi.dlopen(core_library())
    return _lib


def ffi() -> cffi.FFI:
    return _ffi
