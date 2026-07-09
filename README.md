# gtpsa_lib — the xtrack TPSA tracking bridge

This folder builds `libgtpsa_core.so`, the GTPSA engine behind `xtrack.tpsa` — the
backend that turns `line.track(ParticlesTpsa)` into an element/line **Taylor map**.
The repository is made up of two parts:

1. The **GTPSA core** (`libgtpsa_core.so`, built here) — the Generalized Truncated
   Power Series Algebra engine extracted from [MAD-NG][madng] at a pinned tag (now 1.1.14), the
   *same* C code MAD-NG uses for its DA/TPSA maps, without the LuaJIT/MAD
   application layer. Number-agnostic (`mad_*` + LAPACK/BLAS only) and the single owner of mad's global descriptor
   state. Called from Python via the `mad_tpsa_*` ABI (`xtrack/tpsa/_gtpsa.py`).
2. The **bridge** — `xt_bridge.cpp`, the Xsuite element physics run on
   a single particle whose coordinates are either `tpsa_t` (`XT_FLAVOR_TPSA`) or
   `double` (`XT_FLAVOR_NUM`), from one source compiled twice. It is not compiled here but built on demand through xobjects
   `build_kernels` (linking `libgtpsa_core.so`) and cached under `_bridge_cache/`
   by `xtrack.tpsa._gtpsa.bridge_lib`. They export
   `xt_bridge_track_{element,line}_{tpsa,num}`.

Everything that crosses the C/Python boundary — the `XtBridgeParticle` struct
(an `xo.Struct`), the element C-API, the dispatch switch, the cffi cdef, the
registry — is emitted by `gen_bridge.py` from one `ELEMENTS` list.

## Build

```bash
./build.sh                               # fetches MAD-NG v1.1.13, then builds
MAD_SRC=/path/to/MAD-NG/src ./build.sh   # or reuse a local checkout, no network
```

The first build clones the pinned MAD-NG tag into `.mad-ng-<tag>/` and
verifies its commit. Later builds reuse the repository.
Bump `MAD_TAG` **and** `MAD_COMMIT` together to move the pin.

`generated/` is created, never edited — `build.sh` (via
`gen_bridge.py`) writes it, and `xt_bridge.cpp` includes it, so run the build once
in a fresh clone before anything will compile.

Requirements: `git` (for the fetch), `gcc` (for the core `.so`) and `g++` (the bridge
modules are C++, using MAD-NG's header-only `mad_tpsa.hpp` wrapper, invoked lazily by
xobjects, not by `build.sh`), `liblapack.so.3`, `libblas.so.3` (LAPACK is only needed
for map inversion `mad_tpsa_minv`). `xtrack` and `xobjects` must be importable from the
active environment. `gen_bridge.py` writes into whichever `xtrack` that resolves to.
`build.sh` emits `libgtpsa_core.so` in this folder plus the generated sources the bridge modules
compile against. Then, point `XTRACK_GTPSA_LIB` at a file in this folder.

## Validation

The `_num` flavor is compiled with `XT_NUM = double` and is identical to
native `line.track` — the correctness check (never built with `-ffast-math`).
This repo has no test of its own yet, tests in Xtrack will follow.

## Licensing — unresolved, resolve before publishing

MAD-NG's GTPSA sources are **GPLv3 (or later)**; `xtrack` and `xobjects` are
**Apache-2.0**. No MAD-NG source is vendored here as `build.sh` fetches it at a pinned tag
and `libgtpsa_core.so` is a local build artifact which is not redistributed.

However, `xt_local_particle.hpp` `#include`s MAD-NG's `mad_tpsa.hpp`, which is
not an interface header and contains the inline/template definitions supplying the
`mad::tpsa` operators the bridge is built on. This makes our own sources a derivative work,
so distributing *them* — with or without any MAD-NG file, with or without the `.so` would have to carry GPLv3 terms.

Consequently, that would mean: the combined work must be GPLv3-compatible. Apache-2.0
code may be folded into it (the result is GPLv3), but the reverse is not allowed, so this
could not ship as an Apache-2.0-only part of Xsuite. Nothing triggers until the work is
**conveyed** — building and using it locally is unrestricted (GPLv3 §2).

MAD-NG and Xsuite are both CERN accelerator codes, developed in the same group, so these
licensing issues should be discussed.

[madng]: https://github.com/MethodicalAcceleratorDesign/MAD
