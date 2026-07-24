/* gtpsa_stubs.c - stubs for the minimal CMake build.
 *
 * The real-TPSA functions erf/erfc/erfcx/erfi/wf are implemented in MAD-NG through
 * complex TPSA (mad_ctpsa_*) and the Faddeeva package (mad_erfw.c). Our
 * tracking/matching use cases usually do not call those, so the minimal build drops the
 * whole mad_ctpsa_*.o + mad_erfw.o set and links these aborting
 * stubs instead, keeping the shared object free of undefined symbols.
 *
 * If erf/wf or the complex TPSA is needed, add a full CMake build variant.
 */
#include <stdio.h>
#include <stdlib.h>

#define STUB(name)                                                            \
  void name(void) {                                                           \
    fprintf(stderr,                                                           \
            "gtpsa: '%s' is unavailable in the minimal build "                \
            "(complex-TPSA / erf functions disabled).\n", #name);            \
    abort();                                                                  \
  }

/* Faddeeva functions (mad_erfw.c) */
STUB(Faddeeva_w)
STUB(Faddeeva_erf)
STUB(Faddeeva_erf_re)
STUB(Faddeeva_erfc)
STUB(Faddeeva_erfc_re)
STUB(Faddeeva_erfcx)
STUB(Faddeeva_erfcx_re)
STUB(Faddeeva_erfi)
STUB(Faddeeva_erfi_re)
STUB(Faddeeva_Dawson)
STUB(Faddeeva_Dawson_re)

/* complex TPSA (mad_ctpsa_*) */
STUB(mad_ctpsa_new)
STUB(mad_ctpsa_del)
STUB(mad_ctpsa_cplx)
STUB(mad_ctpsa_real)
STUB(mad_ctpsa_scl)
STUB(mad_ctpsa_axpb)
STUB(mad_ctpsa_erf)
STUB(mad_ctpsa_wf)
STUB(mad_ctpsa_logaxpsqrtbpcx2)
