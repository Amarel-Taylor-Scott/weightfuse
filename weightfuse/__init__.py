"""weightfuse — fuse sample-weight signals with held-out/null validation."""

from __future__ import annotations

from .core import normalize, pearson, weighted_ridge
from .signals import FUSION_MODES, fuse_weights, recency_signal, residual_stability_signal
from .validate import candidate_weights, held_out

__all__ = [
    "pearson", "weighted_ridge", "normalize", "recency_signal",
    "residual_stability_signal", "fuse_weights", "FUSION_MODES",
    "candidate_weights", "held_out",
]
__version__ = "0.1.0"
