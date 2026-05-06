````markdown
# Typing Guidelines for NumPy/SciPy-Style Functions (Python 3.14+)

## Goals

- Accept both **scalars** and **arrays** (“array-like” inputs).
- Keep annotations **simple, stable, and checker-friendly** (including Astral `ty`).
- Prefer **consistent return types**; only model scalar-vs-array returns when it’s a real API guarantee.

---

## 0) Python 3.14+ annotation baseline

- **Do not add** `from __future__ import annotations` (Python 3.14+ already uses deferred/lazy annotations by default).
- Use modern typing syntax: `X | Y`, built-in generics (`list[int]`), etc.

```python
from typing import Any, overload, TypeVar
import numpy as np
import numpy.typing as npt
````

---

## 1) Inputs: default to `npt.ArrayLike`

### Rule

If an argument is “anything NumPy can turn into an array”, annotate it as:

* `x: npt.ArrayLike`

### Pattern

Convert immediately for safe computation/indexing:

```python
def f(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    a = np.asarray(x, dtype=np.float64)
    return a + 1.0
```

### Notes

* `ArrayLike` is broad and **not safely indexable** until you normalize with `np.asarray`.
* Prefer **one** `ArrayLike` + `np.asarray` over giant unions.

---

## 2) Returns: default to `NDArray[...]` (even if scalar input is allowed)

### Rule

If the function is vectorized in spirit, return an array type:

* `-> npt.NDArray[...]`

Even if scalar input yields a 0-D array at runtime, that’s fine and common.

```python
def g(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    a = np.asarray(x, dtype=np.float64)
    return np.sin(a)
```

Callers who need a Python scalar can do `.item()`.

---

## 3) Only use scalar/array overloads when the API *promises* it

### Rule

If your public contract is “scalar in → Python scalar out; array in → array out”, then use `@overload`.

### Template (single input)

```python
@overload
def h(x: float) -> float: ...
@overload
def h(x: npt.ArrayLike) -> npt.NDArray[np.float64]: ...

def h(x: float | npt.ArrayLike):
    a = np.asarray(x, dtype=np.float64)
    y = np.sqrt(a)
    if np.isscalar(x):
        return float(y.item())
    return y
```

### Template (two inputs)

```python
@overload
def dotlike(a: float, b: float) -> float: ...
@overload
def dotlike(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.float64]: ...

def dotlike(a: float | npt.ArrayLike, b: float | npt.ArrayLike):
    A = np.asarray(a, dtype=np.float64)
    B = np.asarray(b, dtype=np.float64)
    y = A * B
    if np.isscalar(a) and np.isscalar(b):
        return float(y.item())
    return y
```

### Guidance

* Overloads increase maintenance. Use them only when callers truly rely on scalar returns.

---

## 4) Dtype strategy: be explicit when it matters

### Rules

* If output is always float: return `npt.NDArray[np.float64]` (or `npt.NDArray[np.floating[Any]]` if you want “any float”).
* If you preserve dtype: link input/output with a `TypeVar`.
* Don’t try to encode NumPy casting rules exhaustively.

### Templates

#### Force float64 output

```python
def normalize(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    a = np.asarray(x, dtype=np.float64)
    return a / np.linalg.norm(a)
```

#### Preserve element type (array in → array out)

```python
D = TypeVar("D", bound=np.generic)

def identity_like(x: npt.NDArray[D]) -> npt.NDArray[D]:
    return x.copy()
```

#### Preserve float precision for numpy scalars

```python
F = TypeVar("F", bound=np.floating)

def add_same_precision(a: F, b: F) -> F:
    return a + b
```

---

## 5) Public wrapper + typed core is a good pattern

### Rule

Keep public APIs permissive (`ArrayLike`), but write a strongly-typed internal core that assumes arrays.

```python
def _core(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return a * a + 1.0

def public(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return _core(np.asarray(x, dtype=np.float64))
```

Benefits:

* Cleaner internals (no scalar edge cases).
* Stronger types where they matter.

---

## 6) Shape requirements: enforce at runtime (typing stays generic)

### Rule

Unless your toolchain fully supports shape typing and you’re committed to it, enforce shapes with runtime checks.

```python
def needs_1d(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError("x must be 1D")
    return a
```

---

## 7) Decision checklist for an AI agent

1. **Is the input “array-like”?** → `npt.ArrayLike` and `np.asarray(...)` immediately.
2. **Can the function be treated as vectorized?** → return `npt.NDArray[...]`.
3. **Does the API guarantee scalar-in → scalar-out?** → add overloads + `.item()`.
4. **Does dtype matter?** → specify dtype in `NDArray[...]` and/or in `np.asarray(..., dtype=...)`.
5. **Need shape constraints?** → runtime checks (keep typing generic).

---

## Recommended default templates

### Default

```python
def func(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    a = np.asarray(x, dtype=np.float64)
    ...
    return out
```

### Two inputs

```python
def func2(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    ...
    return out
```

### Scalar-preserving API

```python
@overload
def func_scalar(x: float) -> float: ...
@overload
def func_scalar(x: npt.ArrayLike) -> npt.NDArray[np.float64]: ...

def func_scalar(x: float | npt.ArrayLike):
    a = np.asarray(x, dtype=np.float64)
    y = ...
    return float(y.item()) if np.isscalar(x) else y
```
