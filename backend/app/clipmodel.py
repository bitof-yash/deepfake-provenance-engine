"""Clip-level feature vector + optional trained baseline classifier.

clip_vector() turns the per-window analysis of one clip into a fixed-length vector (37 numbers):
within-clip outlier stats (how weird is the worst window?) AND absolute clip-level feature levels
(needed so a classifier can compare *across* clips). The trained baseline is optional: if
backend/models/baseline.joblib exists it is used for the detection score, otherwise the
heuristic score is used (see pipeline.analyze_file).
"""
import pathlib

import numpy as np

MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "models" / "baseline.joblib"
VECTOR_DIM = 37


def _stats(a) -> list:
    """max, mean, std of the finite values in a (zeros if none)."""
    a = np.asarray(a, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return [0.0, 0.0, 0.0]
    return [float(a.max()), float(a.mean()), float(a.std())]


def _cols(m, fn) -> list:
    m = np.asarray(m, dtype=float)
    out = []
    for k in range(m.shape[1]):
        out += fn(m[:, k])
    return out


def _level(m) -> list:
    """Per-column mean and std of the raw feature across the clip's valid windows."""
    m = np.asarray(m, dtype=float)
    m = m[np.isfinite(m).all(axis=1)]
    if len(m) == 0:
        return [0.0] * 6
    return [float(x) for x in m.mean(axis=0)] + [float(x) for x in m.std(axis=0)]


def clip_vector(vres, ares, lip_scores) -> np.ndarray:
    v = []
    v += _cols(np.abs(vres.z), _stats)                      # 9  video within-clip outliers
    v += _level(vres.feats)                                 # 6  video absolute levels
    if ares is not None:
        v += _cols(np.abs(ares.z), _stats)                  # 9  audio within-clip outliers
        v += _level(ares.feats)                             # 6  audio absolute levels
        voiced = float(np.isfinite(ares.z).all(axis=1).mean()) if len(ares.z) else 0.0
    else:
        v += [0.0] * 15
        voiced = 0.0
    v.append(voiced)                                        # 1  fraction of windows with speech
    lip = np.array([np.nan if s is None else s for s in (lip_scores or [])], dtype=float)
    v += _stats(lip) if lip.size else [0.0] * 3             # 3  lip-sync suspicion
    v.append(float(np.isfinite(lip).mean()) if lip.size else 0.0)  # 1  lip-sync availability
    v.append(1.0 if vres.face_found else 0.0)               # 1
    v.append(float(len(vres.window_scores)))                # 1  number of windows
    out = np.nan_to_num(np.array(v, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    assert out.size == VECTOR_DIM, out.size
    return out


def load_model():
    if not MODEL_PATH.exists():
        return None
    import joblib
    return joblib.load(MODEL_PATH)


def predict(bundle, vec) -> float:
    return float(bundle.predict_proba(np.asarray(vec, dtype=float).reshape(1, -1))[0, 1])
