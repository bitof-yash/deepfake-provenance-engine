"""Cross-modal lip-sync consistency (weak MVP signal).

Correlates lower-face motion with the audio energy envelope over a short
context around each window. Low/negative correlation => higher suspicion.
Returns None when either signal is flat (silence, no face, no motion).
"""
import numpy as np

from . import config


def analyze(mouth_motion: np.ndarray, envelope: np.ndarray, n_windows: int) -> list:
    out = [None] * n_windows
    if mouth_motion.size == 0 or envelope.size == 0:
        return out
    n = min(len(mouth_motion), len(envelope))
    m, e = mouth_motion[:n], envelope[:n]
    spw = int(round(config.WINDOW_SECONDS * config.FRAME_FPS))
    half = int(round(config.LIPSYNC_CONTEXT_SECONDS * config.FRAME_FPS / 2))
    for w in range(n_windows):
        centre = int((w + 0.5) * spw)
        a, b = max(0, centre - half), min(n, centre + half)
        if b - a < 10:
            continue
        ms, es = m[a:b], e[a:b]
        if ms.std() < 1e-3 or es.std() < 1e-4:
            continue
        corr = float(np.corrcoef(ms, es)[0, 1])
        if np.isnan(corr):
            continue
        out[w] = float(np.clip((0.3 - corr) / 0.6, 0.0, 1.0))
    return out
