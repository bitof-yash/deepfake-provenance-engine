"""Shared helpers for within-clip outlier scoring."""
import numpy as np

from . import config


def robust_z(values: np.ndarray) -> np.ndarray:
    """Robust z-score of each value against the clip's own median/MAD.

    The scale is floored at 10% of |median| (plus a tiny constant) so that
    near-constant features do not explode into huge z-scores.
    """
    values = np.asarray(values, dtype=np.float64)
    med = np.median(values)
    mad = np.median(np.abs(values - med)) * 1.4826
    scale = max(mad, 0.10 * abs(med), 1e-6)
    return (values - med) / scale


def z_to_score(z: np.ndarray) -> np.ndarray:
    """Map |z| to a 0..1 suspicion score (0.5 at config.Z_MIDPOINT)."""
    return 1.0 / (1.0 + np.exp(-config.Z_SLOPE * (np.abs(z) - config.Z_MIDPOINT)))
