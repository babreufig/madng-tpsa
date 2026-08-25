"""CFFI ABI bindings for the packaged ``libmadng_tpsa`` shared library.

The declarations below are the small MAD-NG GTPSA subset used by the Python
wrapper. TPSA and descriptor objects stay opaque at this layer. MAD-NG uses
1-based variable and parameter indices.
"""

from __future__ import annotations

from typing import Any

import cffi

from .paths import core_library

CDEF = """
    extern unsigned char mad_tpsa_dflt;
    extern unsigned char mad_tpsa_same;

    void* mad_desc_newv(int nv, unsigned char mo);
    void* mad_desc_newvp(int nv, unsigned char mo, int np, unsigned char po);
    void* mad_desc_newvpo(
        int nv,
        unsigned char mo,
        int np,
        unsigned char po,
        const unsigned char* no
    );
    void mad_desc_del(const void* d);
    int mad_desc_getnv(const void* d, unsigned char* mo_, int* np_, unsigned char* po_);
    unsigned char mad_desc_maxord(const void* d, int n, unsigned char* no);
    _Bool mad_desc_isvalidm(const void* d, int n, const unsigned char* m);

    void* mad_tpsa_newd(void* desc, unsigned char mo);
    const void* mad_tpsa_desc(const void* t);
    unsigned char mad_tpsa_ord(const void* t, _Bool hi);
    void mad_tpsa_del(void* t);
    void mad_tpsa_setvar(void* t, double v, int iv, double scl);
    void mad_tpsa_setprm(void* t, double v, int ip);
    void mad_tpsa_setval(void* t, double v);
    void mad_tpsa_clear(void* t);
    int mad_tpsa_isnul(const void* t);
    int mad_tpsa_isval(const void* t);
    double mad_tpsa_geti(void* t, int i);
    double mad_tpsa_getm(void* t, int n, const unsigned char* m);
    void mad_tpsa_seti(void* t, int i, double a, double b);
    void mad_tpsa_setm(void* t, int n, const unsigned char* m, double a, double b);
    void mad_tpsa_copy(const void* t, void* r);
    int mad_tpsa_cycle(void* t, int i, int n, unsigned char* m, double* v);

    void mad_tpsa_add(const void* a, const void* b, void* c);
    void mad_tpsa_sub(const void* a, const void* b, void* c);
    void mad_tpsa_mul(const void* a, const void* b, void* c);
    void mad_tpsa_div(const void* a, const void* b, void* c);
    int mad_tpsa_equ(const void* a, const void* b, double tol);
    void mad_tpsa_pow(const void* a, const void* b, void* c);
    void mad_tpsa_powi(const void* a, int n, void* c);
    void mad_tpsa_pown(const void* a, double v, void* c);

    void mad_tpsa_scl(const void* a, double v, void* c);
    void mad_tpsa_divn(const void* a, double v, void* c);
    void mad_tpsa_inv(const void* a, double v, void* c);

    double mad_tpsa_nrm(const void* a);
    void mad_tpsa_unit(const void* a, void* c);
    void mad_tpsa_abs(const void* a, void* c);
    void mad_tpsa_sqrt(const void* a, void* c);
    void mad_tpsa_exp(const void* a, void* c);
    void mad_tpsa_log(const void* a, void* c);

    void mad_tpsa_sin(const void* a, void* c);
    void mad_tpsa_cos(const void* a, void* c);
    void mad_tpsa_tan(const void* a, void* c);
    void mad_tpsa_sinc(const void* a, void* c);
    void mad_tpsa_sincos(const void* a, void* s, void* c);
    void mad_tpsa_sincosq(const void* a, void* s, void* c);
    void mad_tpsa_sincosmq(const void* a, void* s, void* c);

    void mad_tpsa_sinh(const void* a, void* c);
    void mad_tpsa_cosh(const void* a, void* c);
    void mad_tpsa_tanh(const void* a, void* c);
    void mad_tpsa_sinhc(const void* a, void* c);
    void mad_tpsa_sincosh(const void* a, void* s, void* c);
    void mad_tpsa_sincoshq(const void* a, void* s, void* c);
    void mad_tpsa_sincoshmq(const void* a, void* s, void* c);

    void mad_tpsa_asin(const void* a, void* c);
    void mad_tpsa_acos(const void* a, void* c);
    void mad_tpsa_atan(const void* a, void* c);
    void mad_tpsa_asinh(const void* a, void* c);
    void mad_tpsa_acosh(const void* a, void* c);
    void mad_tpsa_atanh(const void* a, void* c);

    void mad_tpsa_erf(const void* a, void* c);
    void mad_tpsa_erfc(const void* a, void* c);
    void mad_tpsa_erfcx(const void* a, void* c);
    void mad_tpsa_erfi(const void* a, void* c);
    void mad_tpsa_wf(const void* a, void* c);

    void mad_tpsa_atan2(const void* y, const void* x, void* r);
    void mad_tpsa_hypot(const void* x, const void* y, void* r);
    void mad_tpsa_hypot3(const void* x, const void* y, const void* z, void* r);
    void mad_tpsa_axpb(double a, const void* x, double b, void* r);
    void mad_tpsa_integ(const void* a, void* c, int iv);
    void mad_tpsa_deriv(const void* a, void* c, int iv);
    void mad_tpsa_derivm(const void* a, void* c, int n, const unsigned char* m);
    void mad_tpsa_poisbra(const void* a, const void* b, void* c, int nv);

    typedef void (*madng_tpsa_unary_fn)(const void* input, void* output);
    typedef void (*madng_tpsa_binary_fn)(const void* left, const void* right, void* output);
    typedef void (*madng_tpsa_two_output_fn)(
        const void* input,
        void* first_output,
        void* second_output
    );
    int madng_tpsa_protected_unary_call(
        madng_tpsa_unary_fn function,
        const void* input,
        void* output
    );
    int madng_tpsa_protected_binary_call(
        madng_tpsa_binary_fn function,
        const void* left,
        const void* right,
        void* output
    );
    int madng_tpsa_protected_two_output_call(
        madng_tpsa_two_output_fn function,
        const void* input,
        void* first_output,
        void* second_output
    );
    const char* madng_tpsa_last_error_location(void);
    const char* madng_tpsa_last_error_message(void);

    int madng_tpsa_check_tpsa_compatibility(const void* left, const void* right);
    int madng_tpsa_tpsa_variable_index(const void* series);
    int madng_tpsa_tpsa_single_monomial(
        const void* series,
        int monomial_length,
        unsigned char* monomial_orders
    );
"""

_ffi = cffi.FFI()
_ffi.cdef(CDEF)
_lib: Any = None


def lib() -> Any:
    """Return the lazily loaded ``libmadng_tpsa`` handle."""
    global _lib
    if _lib is None:
        _lib = _ffi.dlopen(core_library())
    return _lib


def ffi() -> cffi.FFI:
    """Return the shared CFFI parser/context."""
    return _ffi
