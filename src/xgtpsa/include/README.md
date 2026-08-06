# GTPSA MAD-NG API notes

This folder is automatically populated by the build process (`pip install -e path/to/xgtpsa`)
and contains the patched GTPSA public headers from MAD-NG.

These notes collect practical information about the subset of the MAD-NG GTPSA API
that xgtpsa uses and exposes.

## Descriptor

A `desc_t` descriptor defines the algebraic space in which TPSA objects live. It is
not itself a polynomial. Instead, it describes which monomials can exist and how
they are indexed.

At the public C API level, descriptors are opaque:

```c
typedef struct desc_ desc_t;
```

A descriptor is created with functions such as:

```c
const desc_t* mad_desc_newv(int nv, ord_t mo);
const desc_t* mad_desc_newvp(int nv, ord_t mo, int np_, ord_t po_);
const desc_t* mad_desc_newvpo(int nv, ord_t mo, int np_, ord_t po_, const ord_t no_[]);
```

Conceptually, a descriptor holds:

- `nv`: number of ordinary variables.
- `np`: number of parameters.
- $nn = nv + np$: total monomial length.
- `mo`: maximum polynomial order.
- `po`: maximum combined parameter order.
- `no_`: optional per-variable/per-parameter maximum orders.
- monomial tables and lookup structures used to map monomials to coefficient
  indexes.

Parameters are appended after ordinary variables in monomial tuples. For example,
a descriptor with `nv = 2` and `np = 1` uses monomials of length 3:

$$(x_\mathrm{order}, y_\mathrm{order}, p_\mathrm{order})$$

The descriptor owns the indexing rules. Functions such as `mad_desc_idxm`,
`mad_desc_mono`, and `mad_desc_isvalidm` convert between monomial descriptions
and internal coefficient indexes, or check whether a monomial is representable in
the descriptor.

The descriptor constructors differ only in how much structure they specify:

| Function | Description |
| --- | --- |
| `const desc_t* mad_desc_newv(int nv, ord_t mo)` | Create a descriptor with `nv` variables and maximum total order `mo`. All variables may appear up to order `mo`. |
| `const desc_t* mad_desc_newvp(int nv, ord_t mo, int np_, ord_t po_)` | Create a descriptor with `nv` variables, `np_` parameters, maximum total order `mo`, and maximum combined parameter order `po_`. Parameters are appended after variables. |
| `const desc_t* mad_desc_newvpo(int nv, ord_t mo, int np_, ord_t po_, const ord_t no_[])` | Create a descriptor with explicit per-variable/per-parameter maximum orders. The `no_` array has length `nv + np_`; variable limits come first, then parameter limits. MAD-NG raises `mo` to at least the maximum entry in `no_`, and raises `po_` to at least the maximum parameter entry. |

MAD-NG reuses equivalent descriptors internally. A TPSA object keeps a pointer to
its descriptor, and TPSA objects should only be combined when their descriptors
are compatible.

## TPSA

A `tpsa_t` is one truncated power series in the algebraic space described by a
`desc_t`.

At the public C API level, TPSA objects are also opaque:

```c
typedef struct tpsa_ tpsa_t;
```

A TPSA is created on a descriptor:

```c
tpsa_t* mad_tpsa_newd(const desc_t *d, ord_t mo);
```

The `mo` argument is the maximum order for this particular series. Passing
`mad_tpsa_dflt` asks MAD-NG to use the descriptor's maximum order.

Conceptually, a TPSA holds:

- a pointer to its descriptor;
- its own active maximum order;
- the range of orders that currently contain non-zero terms;
- optional metadata such as a user id and name;
- the coefficient array.

Coefficient index `0` is the scalar, or constant, coefficient. It is not a
variable. Higher coefficient indexes correspond to monomials defined by the
descriptor tables.

The main access patterns are:

- `mad_tpsa_setvar`: initialize an identity variable series.
- `mad_tpsa_setprm`: initialize an identity parameter series.
- `mad_tpsa_geti` / `mad_tpsa_seti`: get or update by internal coefficient index.
- `mad_tpsa_getm` / `mad_tpsa_setm`: get or update by monomial tuple.
- `mad_tpsa_cycle`: iterate over stored coefficients.

The setters `mad_tpsa_seti` and `mad_tpsa_setm` use the update rule:

$$\mathrm{new\_value} = a\,\mathrm{old\_value} + b$$

For example, setting the constant coefficient to `5` uses index `0` with
$a = 0$ and $b = 5$.

## TPSA construction, shape, and lifetime

| Function | Description |
| --- | --- |
| `tpsa_t* mad_tpsa_newd(const desc_t *d, ord_t mo)` | Allocate a TPSA on descriptor `d`. If `mo` is `mad_tpsa_dflt`, MAD-NG uses the descriptor's maximum order. |
| `tpsa_t* mad_tpsa_new(const tpsa_t *t, ord_t mo)` | Allocate a TPSA compatible with an existing TPSA. |
| `void mad_tpsa_del(const tpsa_t *t)` | Free a TPSA. |
| `const desc_t* mad_tpsa_desc(const tpsa_t *t)` | Return the descriptor that defines the TPSA's algebraic space. |
| `ord_t mad_tpsa_ord(const tpsa_t *t, log_t hi_)` | Return the active maximum order, or the highest currently non-zero order when `hi_` is true. |
| `log_t mad_tpsa_isnul(const tpsa_t *t)` | Test whether the series is zero. |
| `log_t mad_tpsa_isval(const tpsa_t *t)` | Test whether the series contains only a constant coefficient. |
| `void mad_tpsa_clear(tpsa_t *t)` | Clear all coefficients. xgtpsa uses this before loading coefficients from a dictionary. |
| `void mad_tpsa_copy(const tpsa_t *t, tpsa_t *r)` | Copy one TPSA into another compatible TPSA. |

## Coefficients and monomials

MAD-NG uses 1-based indices for variables and parameters in calls such as
`mad_tpsa_setvar`, `mad_tpsa_setprm`, `mad_tpsa_integ`, and `mad_tpsa_deriv`.
Monomial arrays are 0-based C arrays whose length is $nv + np$.

| Function | Description                                                                                                                                                            |
| --- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `void mad_tpsa_setvar(tpsa_t *t, num_t v, idx_t iv, num_t scl_)` | Initialize `t` as identity variable `iv`, expanded around constant value `v`. The variable index is 1-based.                                                           |
| `void mad_tpsa_setprm(tpsa_t *t, num_t v, idx_t ip)` | Initialize `t` as identity parameter `ip`, expanded around value `v`. Parameters are appended after variables in monomial tuples.                                      |
| `void mad_tpsa_setval(tpsa_t *t, num_t v)` | Set `t` to the constant series `v`.                                                                                                                                    |
| `idx_t mad_tpsa_cycle(const tpsa_t *t, idx_t i, ssz_t n, ord_t m_[], num_t *v_)` | Iterate over stored coefficients. Start with `i = -1`; each call returns the next coefficient index, fills `m_` with the monomial orders, and fills `v_` with the coefficient. |
| `num_t mad_tpsa_geti(const tpsa_t *t, idx_t i)` | Return coefficient by internal coefficient index. Index `0` is the constant coefficient.                                                                               |
| `num_t mad_tpsa_getm(const tpsa_t *t, ssz_t n, const ord_t m[])` | Return coefficient by full monomial tuple.                                                                                                                             |
| `void mad_tpsa_seti(tpsa_t *t, idx_t i, num_t a, num_t b)` | Update coefficient index `i` with $\mathrm{new\_value} = a\cdot\mathrm{old\_value} + b$.                                                                               |
| `void mad_tpsa_setm(tpsa_t *t, ssz_t n, const ord_t m[], num_t a, num_t b)` | Update a coefficient selected by full monomial tuple with $\mathrm{new\_value} = a\cdot\mathrm{old\_value} + b$.                                                       |
| `idx_t mad_tpsa_idxm(const tpsa_t *t, ssz_t n, const ord_t m[])` | Convert a valid monomial tuple to MAD-NG's internal coefficient index. Returns `-1` if invalid.                                                                        |
| `ord_t mad_tpsa_mono(const tpsa_t *t, idx_t i, ssz_t n, ord_t m_[], ord_t *p_)` | Convert an internal coefficient index back to monomial orders.                                                                                                         |

## Arithmetic operators

All binary TPSA operations require compatible descriptors. MAD-NG normally enforces
this with `ensure(...)`, which terminates the process; xgtpsa checks compatibility
first at the Python layer where practical.

| Function | Description |
| --- | --- |
| `log_t mad_tpsa_equ(const tpsa_t *a, const tpsa_t *b, num_t tol_)` | Compare two compatible TPSAs coefficient-by-coefficient, with tolerance `tol_`. |
| `void mad_tpsa_add(const tpsa_t *a, const tpsa_t *b, tpsa_t *c)` | Compute $c = a + b$. |
| `void mad_tpsa_sub(const tpsa_t *a, const tpsa_t *b, tpsa_t *c)` | Compute $c = a - b$. |
| `void mad_tpsa_mul(const tpsa_t *a, const tpsa_t *b, tpsa_t *c)` | Compute $c = ab$, truncated to `c`'s order. |
| `void mad_tpsa_div(const tpsa_t *a, const tpsa_t *b, tpsa_t *c)` | Compute $c = a / b$. |
| `void mad_tpsa_pow(const tpsa_t *a, const tpsa_t *b, tpsa_t *c)` | Compute TPSA exponentiation $c = a^b$. |
| `void mad_tpsa_powi(const tpsa_t *a, int n, tpsa_t *c)` | Compute integer power $c = a^n$. |
| `void mad_tpsa_pown(const tpsa_t *a, num_t v, tpsa_t *c)` | Compute scalar real power $c = a^v$. |
| `void mad_tpsa_axpb(num_t a, const tpsa_t *x, num_t b, tpsa_t *r)` | Compute the affine scalar/TPSA expression $r = ax + b$. xgtpsa uses this for scalar addition and subtraction. |

## Scalar helper operations

These are scalar/TPSA operations in the C API. In Python they are implementation
details of arithmetic operators, not currently separate public methods.

| Function | Description |
| --- | --- |
| `void mad_tpsa_acc(const tpsa_t *a, num_t v, tpsa_t *c)` | Accumulate $c \leftarrow c + va$. Aliasing is supported. |
| `void mad_tpsa_scl(const tpsa_t *a, num_t v, tpsa_t *c)` | Compute $c = va$. |
| `void mad_tpsa_divn(const tpsa_t *a, num_t v, tpsa_t *c)` | Compute $c = a/v$. |
| `void mad_tpsa_inv(const tpsa_t *a, num_t v, tpsa_t *c)` | Compute $c = v/a$. xgtpsa uses this for reflected scalar division. |
| `void mad_tpsa_invsqrt(const tpsa_t *a, num_t v, tpsa_t *c)` | Compute $c = v/\sqrt{a}$. |

## Elementary math functions

Most unary functions evaluate a scalar function on a TPSA by expanding around the
constant part and propagating the truncated series terms.

| Function | Description |
| --- | --- |
| `num_t mad_tpsa_nrm(const tpsa_t *a)` | Return the sum of absolute values of stored coefficients. This is a coefficient norm, not a Euclidean norm. |
| `void mad_tpsa_unit(const tpsa_t *a, tpsa_t *c)` | Normalize by the magnitude/sign of the constant coefficient. xgtpsa rejects zero constant part before calling this. |
| `void mad_tpsa_abs(const tpsa_t *a, tpsa_t *c)` | Absolute value branch determined by the constant coefficient. |
| `void mad_tpsa_sqrt(const tpsa_t *a, tpsa_t *c)` | Square root of a TPSA. |
| `void mad_tpsa_exp(const tpsa_t *a, tpsa_t *c)` | Exponential. |
| `void mad_tpsa_log(const tpsa_t *a, tpsa_t *c)` | Natural logarithm. |
| `void mad_tpsa_sin(const tpsa_t *a, tpsa_t *c)` | Sine. |
| `void mad_tpsa_cos(const tpsa_t *a, tpsa_t *c)` | Cosine. |
| `void mad_tpsa_tan(const tpsa_t *a, tpsa_t *c)` | Tangent. |
| `void mad_tpsa_sinc(const tpsa_t *a, tpsa_t *c)` | Unnormalized sinc, $\sin(x)/x$, with MAD-NG's regularization at zero. This differs from NumPy's normalized $\mathrm{sinc}(x) = \sin(\pi x)/(\pi x)$. |
| `void mad_tpsa_sinh(const tpsa_t *a, tpsa_t *c)` | Hyperbolic sine. |
| `void mad_tpsa_cosh(const tpsa_t *a, tpsa_t *c)` | Hyperbolic cosine. |
| `void mad_tpsa_tanh(const tpsa_t *a, tpsa_t *c)` | Hyperbolic tangent. |
| `void mad_tpsa_sinhc(const tpsa_t *a, tpsa_t *c)` | Unnormalized $\sinh(x)/x$, with MAD-NG's regularization at zero. |
| `void mad_tpsa_asin(const tpsa_t *a, tpsa_t *c)` | Inverse sine. |
| `void mad_tpsa_acos(const tpsa_t *a, tpsa_t *c)` | Inverse cosine. |
| `void mad_tpsa_atan(const tpsa_t *a, tpsa_t *c)` | Inverse tangent. |
| `void mad_tpsa_asinh(const tpsa_t *a, tpsa_t *c)` | Inverse hyperbolic sine. |
| `void mad_tpsa_acosh(const tpsa_t *a, tpsa_t *c)` | Inverse hyperbolic cosine. |
| `void mad_tpsa_atanh(const tpsa_t *a, tpsa_t *c)` | Inverse hyperbolic tangent. |

## Paired trigonometric and hyperbolic functions

These functions compute two related outputs in one call. We checked the
implementation, not only the comments, for the `sincos*` conventions.

| Function | Description |
| --- | --- |
| `void mad_tpsa_sincos(const tpsa_t *a, tpsa_t *s, tpsa_t *c)` | Compute $(s, c) = (\sin(a), \cos(a))$. |
| `void mad_tpsa_sincosq(const tpsa_t *a, tpsa_t *s, tpsa_t *c)` | Compute $(s, c) = (\mathrm{sinc}(\sqrt{a}), \cos(\sqrt{a}))$. |
| `void mad_tpsa_sincosmq(const tpsa_t *a, tpsa_t *s, tpsa_t *c)` | Compute $(s, c) = ((\mathrm{sinc}(\sqrt{a}) - 1)/a, (\cos(\sqrt{a}) - 1)/a)$. |
| `void mad_tpsa_sincosh(const tpsa_t *a, tpsa_t *s, tpsa_t *c)` | Compute $(s, c) = (\sinh(a), \cosh(a))$. |
| `void mad_tpsa_sincoshq(const tpsa_t *a, tpsa_t *s, tpsa_t *c)` | Compute $(s, c) = (\mathrm{sinhc}(\sqrt{a}), \cosh(\sqrt{a}))$. |
| `void mad_tpsa_sincoshmq(const tpsa_t *a, tpsa_t *s, tpsa_t *c)` | Compute $(s, c) = ((\mathrm{sinhc}(\sqrt{a}) - 1)/a, (\cosh(\sqrt{a}) - 1)/a)$. |

## Error and Faddeeva functions

MAD-NG includes Faddeeva-based functions from `mad_erfw`. For real TPSAs, complex
intermediate results are projected back to a real TPSA.

| Function | Description |
| --- | --- |
| `void mad_tpsa_erf(const tpsa_t *a, tpsa_t *c)` | Error function. |
| `void mad_tpsa_erfc(const tpsa_t *a, tpsa_t *c)` | Complementary error function. |
| `void mad_tpsa_erfcx(const tpsa_t *a, tpsa_t *c)` | Scaled complementary error function, related to $\mathrm{wf}(ix)$. |
| `void mad_tpsa_erfi(const tpsa_t *a, tpsa_t *c)` | Imaginary error function, implemented through a complex intermediate. |
| `void mad_tpsa_wf(const tpsa_t *a, tpsa_t *c)` | Faddeeva function $w(z) = \exp(-z^2)\mathrm{erfc}(-iz)$. In xgtpsa this is exposed as `wofz`; for real TPSAs it returns the real part of SciPy's complex-valued `scipy.special.wofz`. |

## Binary and ternary math functions

| Function | Description |
| --- | --- |
| `void mad_tpsa_atan2(const tpsa_t *y, const tpsa_t *x, tpsa_t *r)` | Compute $r = \mathrm{atan2}(y, x)$. xgtpsa lifts scalar operands to constant TPSAs before calling this. |
| `void mad_tpsa_hypot(const tpsa_t *x, const tpsa_t *y, tpsa_t *r)` | Compute $r = \sqrt{x^2 + y^2}$. |
| `void mad_tpsa_hypot3(const tpsa_t *x, const tpsa_t *y, const tpsa_t *z, tpsa_t *r)` | Compute $r = \sqrt{x^2 + y^2 + z^2}$. |

## Differential algebra functions

These functions operate on the formal power series structure. Indices are 1-based
and may refer to ordinary variables or appended parameters unless otherwise noted.

| Function | Description |
| --- | --- |
| `void mad_tpsa_integ(const tpsa_t *a, tpsa_t *c, idx_t iv)` | Formal indefinite integral of `a` with respect to variable/parameter index `iv`. The integration constant is zero. |
| `void mad_tpsa_deriv(const tpsa_t *a, tpsa_t *c, idx_t iv)` | First partial derivative with respect to variable/parameter index `iv`. |
| `void mad_tpsa_derivm(const tpsa_t *a, tpsa_t *c, ssz_t n, const ord_t m[])` | Higher or mixed partial derivative. The monomial `m` gives derivative orders; for example $(2, 1)$ means $\partial^3/(\partial x^2\,\partial y)$ in a two-variable descriptor. |
| `void mad_tpsa_poisbra(const tpsa_t *a, const tpsa_t *b, tpsa_t *c, int nv)` | Poisson bracket $[a,b] = \sum_i \partial a/\partial q_i\,\partial b/\partial p_i - \partial a/\partial p_i\,\partial b/\partial q_i$, using canonical variable pairs $(q_1,p_1), (q_2,p_2), \ldots$ stored as indices $(1,2), (3,4), \ldots$. Passing $nv = 0$ uses all available variable pairs; otherwise MAD-NG uses $nv/2$ pairs. xgtpsa exposes this as `num_pairs`, validates $0 < 2\,\mathrm{num\_pairs} \le \mathrm{num\_vars}$, and passes $2\,\mathrm{num\_pairs}$ to C. |
| `void mad_tpsa_taylor(const tpsa_t *a, ssz_t n, const num_t coef[], tpsa_t *c)` | Evaluate $\sum_i \mathrm{coef}[i]\,(a - a_0)^i$, where $a_0$ is the constant part. Coefficients are already Taylor coefficients, so for a scalar function $f$ pass $f(a_0)$, $f'(a_0)$, $f''(a_0)/2!$, and so on. |
| `void mad_tpsa_taylor_h(const tpsa_t *a, ssz_t n, const num_t coef[], tpsa_t *c)` | Same mathematical result as `mad_tpsa_taylor`, evaluated with Horner's method. MAD-NG's implementation notes say this can be slower because multiplication is always full order. |

## High-level combined expressions

These functions fuse common expressions into one C call. xgtpsa currently uses
only a small subset directly.

| Function | Description |
| --- | --- |
| `void mad_tpsa_axpb(num_t a, const tpsa_t *x, num_t b, tpsa_t *r)` | Compute $r = ax + b$. |
| `void mad_tpsa_axpbypc(num_t a, const tpsa_t *x, num_t b, const tpsa_t *y, num_t c, tpsa_t *r)` | Compute $r = ax + by + c$. |
| `void mad_tpsa_axypb(num_t a, const tpsa_t *x, const tpsa_t *y, num_t b, tpsa_t *r)` | Compute $r = axy + b$. |
| `void mad_tpsa_axypbzpc(num_t a, const tpsa_t *x, const tpsa_t *y, num_t b, const tpsa_t *z, num_t c, tpsa_t *r)` | Compute $r = axy + bz + c$. |

## xgtpsa extensions

The `xgtpsa.h` header contains small xgtpsa-owned additions to the MAD-NG API.
These helpers are compiled into the packaged `libmadng_tpsa` shared library, but
are not upstream MAD-NG functions.

| Function | Description |
| --- | --- |
| `int xgtpsa_check_tpsa_compatibility(const tpsa_t *left, const tpsa_t *right)` | Return non-zero when two TPSA objects are compatible according to MAD-NG's descriptor compatibility rule. |
| `int xgtpsa_tpsa_variable_index(const tpsa_t *series)` | Return the 1-based variable/parameter index represented by an identity TPSA. Return `-1` if the TPSA is not an identity variable with exactly one first-order monomial of coefficient `1`. |
| `int xgtpsa_tpsa_single_monomial(const tpsa_t *series, int monomial_length, ord_t monomial_orders[])` | Copy the monomial orders from a TPSA that contains exactly one non-constant monomial. Return non-zero on success, or `0` if the series has a constant part, no non-constant monomial, more than one non-constant monomial, or an incompatible output length. |

xgtpsa uses these helpers to validate inputs in Python before calling MAD-NG
functions that would otherwise terminate through `ensure(...)`.
