"""Base sample-weight signals and fusion rules."""

from __future__ import annotations

import numpy as np

from .core import normalize, weighted_ridge


def recency_signal(n):
    """High weight on recent rows."""
    t = np.linspace(0.0, 1.0, int(n))
    return normalize(np.exp(7.0 * (t - 1.0)))


def residual_stability_signal(X, y):
    """High weight on low-residual rows under a conservative ridge fit."""
    pred = weighted_ridge(X, y, X, lam=5.0)
    resid = np.abs(np.asarray(y, float).ravel() - pred)
    ranks = np.argsort(np.argsort(resid)) / max(1, len(resid) - 1)
    return normalize(0.1 + (1.0 - ranks) ** 3)


def fuse_weights(a, b, mode="product"):
    a = normalize(a)
    b = normalize(b)
    if mode == "product":
        w = a * b
    elif mode == "geo":
        w = np.sqrt(a * b)
    elif mode == "average":
        w = 0.5 * a + 0.5 * b
    else:
        raise ValueError("unknown fusion mode %r" % mode)
    return normalize(w)


FUSION_MODES = ("product", "geo", "average")
