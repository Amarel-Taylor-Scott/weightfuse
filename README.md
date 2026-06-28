# weightfuse

> Fuse sample-weight signals, then make the fused vector prove itself. The demo
> combines **recency** (trust current-regime rows) and **residual stability**
> (trust low-residual rows). A reported win must beat uniform, the best single
> base signal, and a magnitude-matched permutation null. numpy-only.

```python
from weightfuse import held_out
from weightfuse.synth import make_mixed_problem

X, y, _ = make_mixed_problem()
r = held_out(X, y)
print(r["best_mode"], r["best_single"], r["learned"], r["beats_null"])
```

```
$ weightfuse demo
## 1. Drift plus corrupted recent rows
  selected mode     : product
  held-out transfer : 0.9050   (uniform 0.6705, lift +0.2345)
  best single / gap : 0.7582 / +0.1468
  null p95 / p      : 0.8050 / 0.000
  VERDICT           : REAL - fused weights beat single signals + null

## 2. Clean control (must NOT claim a fused gain)
  selected mode     : product
  held-out transfer : 0.9689   (uniform 0.9692, lift -0.0003)
  best single / gap : 0.9692 / -0.0003
  null p95 / p      : 0.9690 / 0.060
  VERDICT           : no fused gain
```

## Why Fuse

Single weighting signals can be too blunt. Recency handles drift, but it trusts
recent corrupted rows. Residual stability rejects bad rows, but it does not know
which old rows are obsolete. Multiplying the two signals keeps rows that are
both current and stable.

`held_out` selects among base and fused candidates using only the LOW split,
then validates on HIGH. `beats_null` is true only for a fused mode that beats
uniform, the best single signal, and a permutation null.

MIT. Depends only on numpy.
