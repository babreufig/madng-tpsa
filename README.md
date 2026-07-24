# xgtpsa - GTPSA (truncated power series) for Python

This repository builds `madng_tpsa.so` on Linux or `madng_tpsa.dylib` on macOS:
the Generalized Truncated Power Series Algebra engine extracted from MAD-NG, without
the LuaJIT/MAD application layer. The Python package uses CFFI ABI mode (`dlopen`),
so importing `xgtpsa` does not compile anything at runtime.

```python
import xgtpsa

d = xgtpsa.Descriptor.new(2, 3)
x = xgtpsa.Tpsa.var(d, 1, 0.5)
y = xgtpsa.Tpsa.var(d, 2)
f = x * x + 2.0 * y
f.const_part, f.grad(), f.monomial_coeffs()
```

Nothing here knows about tracking or xtrack. Consumers can discover the packaged
native artifacts with:

- `xgtpsa.core_library()` - path to `xgtpsa/lib/madng_tpsa.{so,dylib}`
- `xgtpsa.include_dir()` - path to `xgtpsa/include`

## Build

The project uses scikit-build-core and CMake:

```bash
pip install .
pip install -e .
python -m build
```

`pip install -e .` runs CMake and copies the generated shared library into
`src/xgtpsa/lib/`, matching the installed wheel layout. The public header
`src/xgtpsa/include/madng_tpsa.h` is checked into Git; platform shared libraries in
`src/xgtpsa/lib/` are generated artifacts and ignored by Git.

By default CMake builds from the bundled `madng/src` checkout. To use another MAD-NG
source tree:

```bash
pip install . --config-settings=cmake.define.MADNG_SRC=/path/to/madng/src
```

Requirements: a C compiler, CMake, BLAS/LAPACK on Linux, and Accelerate on macOS.

## Tests

```bash
pytest tests/
```

The tests exercise the standalone engine bindings; xtrack is not imported.

## Licensing

MAD-NG's GTPSA sources are GPLv3 or later. This package builds and ships a shared
library derived from those sources, so distribution needs to follow GPL-compatible
terms. See `LICENSE`.

[madng]: https://github.com/MethodicalAcceleratorDesign/MAD
