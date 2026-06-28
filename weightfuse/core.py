"""Numeric core for weightfuse."""

from __future__ import annotations

import numpy as np


def pearson(a, b) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def normalize(w):
    w = np.maximum(np.asarray(w, float).ravel(), 1e-12)
    m = float(w.mean())
    return w / m if m > 0 else np.ones_like(w)


def weighted_ridge(Xtr, ytr, Xev, sw=None, lam=2.0):
    Xtr = np.asarray(Xtr, float)
    ytr = np.asarray(ytr, float).ravel()
    Xev = np.asarray(Xev, float)
    n, p = Xtr.shape
    w = np.ones(n) if sw is None else normalize(sw)
    sw_sum = w.sum()
    mu = (Xtr * w[:, None]).sum(axis=0) / sw_sum
    ybar = float((ytr * w).sum() / sw_sum)
    Xc = Xtr - mu
    A = Xc.T @ (Xc * w[:, None])
    b = Xc.T @ (w * (ytr - ybar))
    beta = np.linalg.solve(A + lam * np.eye(p), b)
    return (Xev - mu) @ beta + ybar
