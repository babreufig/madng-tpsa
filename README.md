# xgtpsa — GTPSA (truncated power series) for Python

This repository builds `libgtpsa_core.so` — the Generalized Truncated Power Series
Algebra engine extracted from [MAD-NG][madng] at a pinned tag (now 1.1.14), the *same*
C code MAD-NG uses for its DA/TPSA maps, without the LuaJIT/MAD application layer —
and wraps it in the standalone `xgtpsa` Python package.

The core is number-agnostic (`mad_*` + LAPACK/BLAS) and is the single owner of mad's
global descriptor state. `xgtpsa` reaches it through the `mad_tpsa_*` ABI (cffi
`dlopen`, no compilation).

```python
import xgtpsa
d = xgtpsa.Descriptor.new(2, 3)          # 2 variables, order 3
x = xgtpsa.Tpsa.var(d, 1, 0.5)           # x, expanded around 0.5
y = xgtpsa.Tpsa.var(d, 2)
f = x * x + 2.0 * y
f.const_part, f.grad(), f.monomial_coeffs()
```

Nothing here knows about tracking or about xtrack. The xtrack TPSA backend
(`xtrack.tpsa`) is a *consumer*: it owns its element bridge and compiles it against this package, via

- `xgtpsa.core_library()` — path to `libgtpsa_core.so` (to link and dlopen), and
- `xgtpsa.include_dir()` — the MAD-NG headers (`mad_tpsa.hpp` and friends).

## Build

```bash
./build.sh                               # fetches MAD-NG at the pinned tag, then builds
MAD_SRC=/path/to/MAD-NG/src ./build.sh   # or reuse a local checkout, no network
```

The first build clones the pinned MAD-NG tag into `.mad-ng-<tag>/` and verifies its
commit. Later builds reuse the repository. Bump `MAD_TAG` **and** `MAD_COMMIT`
together to move the pin.

Artifacts are installed into `xgtpsa/_core/` (`lib/libgtpsa_core.so` +
`include/mad_*.h[pp]`), which is a build product, not source: put the repo on
`PYTHONPATH` or `pip install -e .`, then run `./build.sh` once. `XGTPSA_LIB` overrides
the location if you want to point at a different build.

Requirements: `git` (for the fetch), `gcc`, `liblapack.so.3`, `libblas.so.3` (LAPACK is
only needed for map inversion, `mad_tpsa_minv`). Consumers that compile C++ against the
headers also need `g++`.

## Tests

```bash
pytest tests/            # pure engine tests; xtrack is not imported
```

The tracking side is validated in xtrack (`tests/test_tpsa.py`), including a `_num`
flavor of the bridge compiled with `double` instead of TPSA, which must stay identical
to native `line.track`.

## Licensing — unresolved, resolve before publishing

MAD-NG's GTPSA sources are **GPLv3 (or later)**; `xtrack` and `xobjects` are
**Apache-2.0**. No MAD-NG source is vendored here: `build.sh` fetches it at a pinned tag
and `libgtpsa_core.so` is a local build artifact which is not redistributed.

However, `xt_local_particle.hpp` `#include`s MAD-NG's `mad_tpsa.hpp` (which were now moved to Xtrack),
which is not an interface header and contains the inline/template definitions supplying the
`mad::tpsa` operators the bridge is built on. This makes our own sources a derivative work,
so distributing *them* — with or without any MAD-NG file, with or without the `.so` would have to carry GPLv3 terms.

Consequently, that would mean: the combined work must be GPLv3-compatible. Apache-2.0
code may be folded into it (the result is GPLv3), but the reverse is not allowed, so this
could not ship as an Apache-2.0-only part of Xsuite. Nothing triggers until the work is
**conveyed** — building and using it locally is unrestricted (GPLv3 §2).

MAD-NG and Xsuite are both CERN accelerator codes, developed in the same group, so these
licensing issues should be discussed.

[madng]: https://github.com/MethodicalAcceleratorDesign/MAD
