import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weightfuse import (  # noqa: E402
    FUSION_MODES, candidate_weights, fuse_weights, pearson, recency_signal,
    residual_stability_signal, weighted_ridge, held_out,
)
from weightfuse.synth import make_clean, make_mixed_problem  # noqa: E402


def test_weighted_ridge_ols_baseline():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 4))
    beta = np.array([1.0, -2.0, 0.5, 3.0])
    y = X @ beta
    pred = weighted_ridge(X[:90], y[:90], X[90:], lam=1e-6)
    assert pearson(pred, y[90:]) > 0.999


def test_signals_and_fusions_are_positive_mean_one():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 3))
    y = X @ rng.normal(size=3) + rng.standard_normal(100)
    a = recency_signal(100)
    b = residual_stability_signal(X, y)
    for w in [a, b] + [fuse_weights(a, b, mode) for mode in FUSION_MODES]:
        assert np.all(w > 0)
        assert abs(float(w.mean()) - 1.0) < 1e-9


def test_candidate_weights_contains_base_and_fused_modes():
    X, y, _ = make_mixed_problem(seed=0)
    c = candidate_weights(X[:200], y[:200])
    assert {"recency", "residual", "product", "geo", "average"} <= set(c)


def test_mixed_problem_fused_weights_beat_single_and_null():
    X, y, _ = make_mixed_problem(seed=0)
    r = held_out(X, y, n_null=200, seed=0)
    assert r["best_mode"] in FUSION_MODES, r
    assert r["single_gap"] > 0.01, r
    assert r["beats_null"], r


def test_clean_control_does_not_claim_fused_gain():
    X, y, _ = make_clean(seed=0)
    r = held_out(X, y, n_null=200, seed=0)
    assert not r["beats_null"], r


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print("\n%d passed" % len(fns))
