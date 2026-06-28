"""Held-out + null validation for fused sample weights."""

from __future__ import annotations

import numpy as np

from .core import pearson, weighted_ridge
from .signals import FUSION_MODES, fuse_weights, recency_signal, residual_stability_signal


def candidate_weights(X, y):
    r = recency_signal(len(y))
    s = residual_stability_signal(X, y)
    out = {"recency": r, "residual": s}
    for mode in FUSION_MODES:
        out[mode] = fuse_weights(r, s, mode)
    return out


def held_out(X, y, *, low_frac=0.6, select_frac=0.7, n_null=200, lam=2.0,
             seed=0) -> dict:
    """Select a candidate on LOW only and validate on HIGH.

    ``beats_null`` requires the selected fused vector to beat uniform, the best
    single base signal, and a permutation null.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n = len(y)
    cut = max(4, int(n * low_frac))
    split = max(2, int(cut * select_frac))
    Xl, yl, Xh, yh = X[:cut], y[:cut], X[cut:], y[cut:]

    cands = candidate_weights(Xl, yl)
    best = None
    for name, w in cands.items():
        pred = weighted_ridge(Xl[:split], yl[:split], Xl[split:],
                              sw=w[:split], lam=lam)
        score = pearson(pred, yl[split:])
        if best is None or score > best[0]:
            best = (score, name, w)
    _, name, w = best

    scores = {
        cname: pearson(weighted_ridge(Xl, yl, Xh, sw=cw, lam=lam), yh)
        for cname, cw in cands.items()
    }
    learned = scores[name]
    uniform = pearson(weighted_ridge(Xl, yl, Xh, lam=lam), yh)
    best_single = max(scores["recency"], scores["residual"])

    rng = np.random.default_rng(seed)
    nulls = np.empty(n_null)
    for i in range(n_null):
        nulls[i] = pearson(
            weighted_ridge(Xl, yl, Xh, sw=w[rng.permutation(cut)], lam=lam), yh
        )
    p95 = float(np.quantile(nulls, 0.95))
    is_fused = name in FUSION_MODES

    return {
        "learned": learned, "uniform": uniform, "lift": learned - uniform,
        "best_single": best_single, "single_gap": learned - best_single,
        "null_p95": p95, "null_mean": float(nulls.mean()),
        "null_p": float((nulls >= learned).mean()),
        "beats_null": bool(is_fused and learned > p95 and learned > best_single
                           and (learned - uniform) > 0.005),
        "best_mode": name, "weights": w, "scores": scores,
    }
