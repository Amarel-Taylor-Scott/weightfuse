"""Synthetic data where two weighting signals are needed together."""

from __future__ import annotations

import numpy as np


def make_mixed_problem(n=900, p=8, change=0.35, noise=0.25, out_frac=0.15,
                       badscale=6.0, seed=0):
    """Regime drift plus corrupted recent rows.

    Recency alone trusts the current regime but also trusts bad recent rows.
    Residual stability alone downweights corruption but still mixes obsolete old
    rows. A product fusion keeps rows that are both recent and stable.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta_old = rng.normal(size=p)
    beta_new = -3.0 * beta_old + rng.normal(0.0, 0.3, p)
    change_i = int(change * n)
    y = np.empty(n)
    y[:change_i] = X[:change_i] @ beta_old
    y[change_i:] = X[change_i:] @ beta_new
    y += noise * float(y.std()) * rng.standard_normal(n)

    low = int(0.6 * n)
    candidates = np.arange(change_i, low)
    bad_n = max(1, int(len(candidates) * out_frac))
    bad = rng.choice(candidates, bad_n, replace=False)
    y[bad] = -(X[bad] @ beta_new) + rng.normal(0.0, badscale * float(y.std()), bad_n)
    return X, y, {"bad": bad, "change": change_i}


def make_clean(n=900, p=8, noise=0.25, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = rng.normal(size=p)
    signal = X @ beta
    y = signal + noise * float(signal.std()) * rng.standard_normal(n)
    return X, y, {}
