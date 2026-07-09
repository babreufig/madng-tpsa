#!/usr/bin/env bash
#
# Build the standalone GTPSA core shared library (libgtpsa_core.so) from the MAD-NG
# sources. It follows src/libgtpsa/README.GTPSA but produces a shared object
# (-fPIC + -shared) that can be loaded from Python with cffi.
#
# This is the underlying C code MAD-NG uses for its TPSA/DA engine. We do not build the
# LuaJIT/Mad-application layer and use only the GTPSA core as a standalone library,
# which can be called from Python. For the bridge between Python and C, there are two module flavors
# that are built on demand depending on the selected type (physics with real numbers, or with TPSA).
# This script only creates libgtpsa_core.so as well as the generated sources those module compiles use (gen_bridge.py, below).
#
#   ./build.sh              # minimal real-TPSA build (fetches MAD-NG if needed)
#   ./build.sh --full       # also includes complex TPSA + erf/Faddeeva functions
#   MAD_SRC=/path/to/MAD-NG/src ./build.sh    # use a local checkout, skip the fetch
#   ./build.sh /path/to/MAD-NG/src            # same
#
# LICENSING
# ---------
# The MAD-NG GTPSA library is licensed under the GPLv3 license. The source files are not vendored here,
# as the script fetches the source from the public repository and creates a local shared object library
# from it. However, discussions should be held as Xsuite is licensed under the Apache 2.0 license.
# Additionally, the additional sources here could be considered derivative works of the GTPSA library.
# For now, this shared object libgtpsa_core.so is a local build artifact.
#
# MINIMAL vs FULL
# ---------------
# Many tracking use cases do not require complex TPSA. For this reason, we separated the
# build into two variants. The 'minimal' variant includes only the real TPSA engine, while
# the 'full' variant additionally includes the TPSA for complex numbers and functions (ctpsa).
# In the minimal build, the links to those functions are replaced with aborting stubs (gtpsa_stubs.c)
# so the .so has no undefined symbols.

set -euo pipefail

MODE="minimal"
MAD_SRC="${MAD_SRC:-}"                  # set (env or arg) -> use it, skip the fetch
for arg in "$@"; do
  case "$arg" in
    --full)    MODE="full" ;;
    --minimal) MODE="minimal" ;;
    *)         MAD_SRC="$arg" ;;
  esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="$HERE/build"

# Pinned upstream GTPSA sources.  Kept OUT of this repo (GPLv3, see above) and out of
# BUILD (which is wiped below), so the clone survives rebuilds. Bump both together.
MAD_URL="${MAD_URL:-https://github.com/MethodicalAcceleratorDesign/MAD.git}"
MAD_TAG="${MAD_TAG:-v1.1.14}"
MAD_COMMIT="${MAD_COMMIT:-6e9f93fa222d4f3a3eb4b2685b2c45f2d07feb62}"
MAD_CACHE="${MAD_CACHE:-$HERE/.mad-ng-$MAD_TAG}"

if [ -z "$MAD_SRC" ]; then
  if [ ! -d "$MAD_CACHE/src" ]; then
    echo ">>> fetching MAD-NG $MAD_TAG (GPLv3) -> $MAD_CACHE"
    rm -rf "$MAD_CACHE"
    git clone --depth 1 --branch "$MAD_TAG" "$MAD_URL" "$MAD_CACHE"
  fi
  got="$(git -C "$MAD_CACHE" rev-parse HEAD)"
  [ "$got" = "$MAD_COMMIT" ] || {
    echo "ERROR: $MAD_CACHE is at $got, expected $MAD_COMMIT ($MAD_TAG)." >&2
    echo "       Delete it to re-fetch, or set MAD_COMMIT to accept a new pin." >&2
    exit 1; }
  MAD_SRC="$MAD_CACHE/src"
fi
[ -f "$MAD_SRC/mad_tpsa.c" ] || { echo "ERROR: $MAD_SRC is not a MAD-NG src/ dir" >&2; exit 1; }

echo ">>> mode       : $MODE"
echo ">>> MAD-NG src : $MAD_SRC"
echo ">>> build dir  : $BUILD"

rm -rf "$BUILD"; mkdir -p "$BUILD"; cd "$BUILD"

# Core GTPSA sources + the standalone replacements for the MAD core dependencies.
# The .hpp headers (mad_tpsa.hpp, the C++ GTPSA wrapper) are needed by the bridge
# (xt_bridge.cpp uses mad::tpsa), so they are copied too.
cp -a "$MAD_SRC"/mad_*.h "$MAD_SRC"/mad_*.hpp "$MAD_SRC"/mad_*.c "$MAD_SRC"/sse .
cp -a "$MAD_SRC"/libgtpsa/mad_*.c "$MAD_SRC"/libgtpsa/*.h . 2>/dev/null || true
rm -f mad_main.c mad_fft.c mad_nlopt.c mad_nlopt.h   # MAD app / FFT / nlopt

CFLAGS=(-W -Wall -Wextra -O3 -fPIC -ffast-math -ftree-vectorize
        -Wno-vla-parameter -Wno-misleading-indentation -Wno-empty-body -I.)

# Source set. mad_mat/mad_vec are only needed for map inversion (mad_tpsa_minv,
# pulls in LAPACK/BLAS); mad_str for I/O; mad_num for special-fn helpers.
MIN_SRC=(mad_tpsa.c mad_tpsa_ops.c mad_tpsa_div.c mad_tpsa_fun.c mad_tpsa_io.c
         mad_tpsa_mops.c mad_tpsa_comp.c mad_tpsa_minv.c mad_desc.c mad_mono.c
         mad_bit.c mad_log.c mad_num.c mad_mat.c mad_vec.c mad_str.c mad_cst.c)

if [ "$MODE" = "full" ]; then
  SRC=( $(ls mad_*.c) )                 # everything (incl. ctpsa + erfw)
  SSE="sse/*.c"
  STUB=""
else
  SRC=( "${MIN_SRC[@]}" )
  SSE=""                                # sse intrinsics not referenced by core
  cp -f "$HERE/gtpsa_stubs.c" .
  STUB="gtpsa_stubs.o"
fi

# mad_mem.c needs C11 (because of using max_align_t), the rest can build as C99.
echo ">>> compiling ${#SRC[@]} core sources ..."
gcc -std=c99   "${CFLAGS[@]}" -c $(printf '%s ' "${SRC[@]}" | sed 's/mad_mem.c//') $SSE
gcc -std=gnu11 "${CFLAGS[@]}" -c mad_mem.c 2>/dev/null || true   # only if present
[ -n "$STUB" ] && gcc -std=c99 "${CFLAGS[@]}" -c gtpsa_stubs.c

# The GTPSA-core .so consists only of the mad_* engine and LAPACK/BLAS (no bridge).
# The xt_bridge compiles below. Pure C => link with gcc (no libstdc++).  --no-undefined keeps it
# self-contained (minimal relies on stubs).
echo ">>> linking libgtpsa_core.so (GTPSA core only) ..."
gcc -shared -Wl,--no-undefined -o libgtpsa_core.so *.o \
    -l:liblapack.so.3 -l:libblas.so.3 -lm
cp -f libgtpsa_core.so "$HERE/libgtpsa_core.so"
echo ">>> built $HERE/libgtpsa_core.so ($(stat -c%s "$HERE/libgtpsa_core.so") bytes, $(ls *.o | wc -l) core objects)"

# Generate the Python-C bridge artifacts. xt_bridge.cpp is NOT compiled here, as the
# flavor modules are built on demand by xtrack (_gtpsa.bridge_lib -> xobjects build_kernels)
# and cached under _bridge_cache/. This step only outputs the sources those module compiles consume,
# which are saved in generated/ (element C-API, dispatch, access to LocalParticle).
# gen_bridge.py imports xtrack/xobjects from the active environment -- whichever ones
# `python -c "import xtrack"` resolves to are the ones written into.
python -c "import xtrack, xobjects" 2>/dev/null || {
  echo "ERROR: xtrack + xobjects must be importable (pip install -e ...)" >&2; exit 1; }
echo ">>> xtrack     : $(python -c 'import xtrack,os;print(os.path.dirname(xtrack.__file__))')"
echo ">>> generating bridge artifacts (gen_bridge.py) ..."
python "$HERE/gen_bridge.py"
echo ">>> done. Bridge modules build lazily on first use (xtrack _gtpsa.bridge_lib)."
