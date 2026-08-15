"""UNG node adapters for weightfuse — fused sample-weight signals with honest validation.

Pure top-level functions over the documented public API (``recency_signal``,
``residual_stability_signal``, ``fuse_weights``, ``candidate_weights``,
``held_out``).  Matrices and vectors cross the boundary as nested lists; numpy
arrays never leak out.  Stochastic steps are seeded, so every node is
deterministic given ``seed``.  Each function returns a dict keyed by its
declared output port names.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from weightfuse import candidate_weights, fuse_weights, held_out


def _jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays (and containers of them) to plain JSON types."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    return obj


def compute_weights(X: list[list[float]], y: list[float],
                    mode: str = "product") -> dict[str, Any]:
    """Base recency/residual-stability weight signals and the selected (possibly fused) vector."""
    cands = candidate_weights(np.asarray(X, float), np.asarray(y, float))
    if mode not in cands:
        raise ValueError(f"unknown mode {mode!r} (use one of {sorted(cands)})")
    return {"weights": _jsonable(cands[mode]),
            "report": {"recency": _jsonable(cands["recency"]),
                       "residual": _jsonable(cands["residual"])}}


def fuse(weights_a: list[float], weights_b: list[float],
         mode: str = "product") -> dict[str, Any]:
    """Fuse two mean-1 weight vectors (product, geometric mean, or average)."""
    fused = fuse_weights(np.asarray(weights_a, float),
                         np.asarray(weights_b, float), mode=mode)
    return {"weights": _jsonable(fused)}


def validate_held_out(X: list[list[float]], y: list[float], low_frac: float = 0.6,
                      select_frac: float = 0.7, n_null: int = 200,
                      lam: float = 2.0, seed: int = 0) -> dict[str, Any]:
    """Select among base and fused weight candidates on early rows; validate on held-out rows vs single signals and a null."""
    rep = held_out(np.asarray(X, float), np.asarray(y, float),
                   low_frac=float(low_frac), select_frac=float(select_frac),
                   n_null=int(n_null), lam=float(lam), seed=int(seed))
    return {"report": _jsonable(rep)}


_TAGS = ["license.mit", "runtime.python"]
_X_PORT = {"name": "X", "type_id": "amarel.types.matrix",
           "description": "Feature matrix, rows in time order (nested lists)."}
_Y_PORT = {"name": "y", "type_id": "amarel.types.vector",
           "description": "Target vector aligned with the rows of X."}

NODES = [
    {
        "fn": compute_weights,
        "id": "amarel.weightfuse.compute-weights",
        "capabilities": ["weights.compute", "weights.fuse"],
        "summary": "Compute recency and residual-stability weight signals and return the chosen base or fused vector.",
        "inputs": [_X_PORT, _Y_PORT],
        "outputs": [
            {"name": "weights", "type_id": "amarel.types.weights",
             "description": "Mean-1 per-row weights for the selected mode."},
            {"name": "report", "type_id": "amarel.types.report",
             "description": "{'recency': [...], 'residual': [...]} — the base signals."},
        ],
        "parameters": [
            {"name": "mode", "value_type": "string", "default": "product",
             "required": False,
             "choices": ["recency", "residual", "product", "geo", "average"]},
        ],
        "effects": [],
        "determinism": "deterministic",
        "idempotency": "idempotent",
        "tags": _TAGS,
    },
    {
        "fn": fuse,
        "id": "amarel.weightfuse.fuse-weights",
        "capabilities": ["weights.fuse"],
        "summary": "Fuse two mean-1 sample-weight vectors by product, geometric mean, or average.",
        "inputs": [
            {"name": "weights_a", "type_id": "amarel.types.weights",
             "description": "First weight signal (any positive vector; renormalized to mean 1)."},
            {"name": "weights_b", "type_id": "amarel.types.weights",
             "description": "Second weight signal (same length)."},
        ],
        "outputs": [
            {"name": "weights", "type_id": "amarel.types.weights",
             "description": "Fused mean-1 weight vector."},
        ],
        "parameters": [
            {"name": "mode", "value_type": "string", "default": "product",
             "required": False, "choices": ["product", "geo", "average"]},
        ],
        "effects": [],
        "determinism": "deterministic",
        "idempotency": "idempotent",
        "tags": _TAGS,
    },
    {
        "fn": validate_held_out,
        "id": "amarel.weightfuse.validate-held-out",
        "capabilities": ["weights.validate", "validation.null-controlled"],
        "summary": "Held-out validation of the fused weights against uniform, the best single signal, and a permutation null.",
        "inputs": [_X_PORT, _Y_PORT],
        "outputs": [
            {"name": "report", "type_id": "amarel.types.report",
             "description": "{learned, uniform, lift, best_single, single_gap, null_p95, null_mean, null_p, beats_null, best_mode, weights, scores}."},
        ],
        "parameters": [
            {"name": "low_frac", "value_type": "number", "default": 0.6, "required": False},
            {"name": "select_frac", "value_type": "number", "default": 0.7, "required": False},
            {"name": "n_null", "value_type": "integer", "default": 200, "required": False},
            {"name": "lam", "value_type": "number", "default": 2.0, "required": False},
            {"name": "seed", "value_type": "integer", "default": 0, "required": False},
        ],
        "effects": [],
        "determinism": "deterministic",
        "idempotency": "idempotent",
        "tags": _TAGS,
    },
]
