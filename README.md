# xgtpsa - GTPSA (truncated power series) for Python

This repository builds `libmadng_tpsa.so` on Linux or `libmadng_tpsa.dylib` on macOS:
the Generalized Truncated Power Series Algebra engine extracted from MAD-NG, without
the LuaJIT/MAD application layer. The Python package uses CFFI ABI mode (`dlopen`),
so importing `xgtpsa` does not compile anything at runtime.

```python
import xgtpsa

d = xgtpsa.Descriptor(2, 3)
x = xgtpsa.Tpsa.var(d, 1, 0.5)
y = xgtpsa.Tpsa.var(d, 2)
f = x * x + 2.0 * y
f.const_part, f.grad(), f.monomial_coeffs()
```

Nothing here knows about tracking or xtrack. Consumers can discover the packaged
native artifacts with:

- `xgtpsa.core_library()` - path to `xgtpsa/lib/libmadng_tpsa.{so,dylib}`
- `xgtpsa.include_dir()` - path to `xgtpsa/include`

## Build

The project uses scikit-build-core and CMake:

```bash
pip install .
pip install -e .
python -m build
```

`pip install -e .` runs CMake and copies the generated shared library into
`src/xgtpsa/lib/`, matching the installed wheel layout. MAD-NG headers copied into
`src/xgtpsa/include/` and platform shared libraries in `src/xgtpsa/lib/` are generated
artifacts and ignored by Git.

By default CMake builds from the bundled `madng/src` checkout. To use another MAD-NG
source tree:

```bash
pip install . --config-settings=cmake.define.MADNG_SRC=/path/to/madng/src
```

Requirements: a C compiler, CMake, BLAS/LAPACK on Linux, and Accelerate on macOS.

## Development

Install the package together with the development tools:

```bash
python -m pip install --group dev -e .
```

Enable the pre-commit hooks:

```bash
pre-commit install
```

The hooks run Ruff formatting/linting and Ty type checking. To run the same checks
manually:

```bash
pre-commit run --all-files
```

## Tests

```bash
pytest tests/
```

The tests exercise the standalone engine bindings; xtrack is not imported.

[madng]: https://github.com/MethodicalAcceleratorDesign/MAD
