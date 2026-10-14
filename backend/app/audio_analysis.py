"""Audio-stream analysis: spectral outlier scores per window + energy envelope.

MVP heuristics (no trained weights): within-clip outliers in spectral
flatness, high-band energy ratio and spectral flux. Silent windows return None.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import stft

from . import config
from .scoring import robust_z, z_to_score

SILENCE_RMS = 1e-3


@dataclass
class AudioResult:
    window_scores: list          # list[float | None], one per 1 s window
    envelope: np.ndarray         # RMS per 1/FRAME_FPS second block
    n_windows: int
    feats: np.ndarray            # raw per-window features (n_windows, 3)
    z: np.ndarray                # within-clip z-scores; NaN rows for silent windows


def _envelope(x: np.ndarray) -> np.ndarray:
    block = int(config.AUDIO_SR / config.FRAME_FPS)
    n = len(x) // block
    if n == 0:
        return np.zeros(0)
    blocks = x[: n * block].reshape(n, block)
    return np.sqrt((blocks ** 2).mean(axis=1))


def analyze(x: np.ndarray, n_windows: int) -> AudioResult:
    sr = config.AUDIO_SR
    f, t, Z = stft(x, fs=sr, nperseg=512, noverlap=256)
    P = np.abs(Z) ** 2 + 1e-12
    flat = np.exp(np.log(P).mean(axis=0)) / P.mean(axis=0)
    hfr = P[f >= 4000].sum(axis=0) / P.sum(axis=0)
    flux = np.concatenate([[0.0], np.maximum(np.diff(np.sqrt(P), axis=1), 0).mean(axis=0)])

    feats, silent = [], []
    for w in range(n_windows):
        a, b = w * config.WINDOW_SECONDS, (w + 1) * config.WINDOW_SECONDS
        idx = (t >= a) & (t < b)
        seg = x[int(a * sr): int(b * sr)]
        rms = float(np.sqrt((seg ** 2).mean())) if len(seg) else 0.0
        silent.append(rms < SILENCE_RMS or not idx.any())
        feats.append([np.median(flat[idx]), np.median(hfr[idx]), np.mean(flux[idx])] if idx.any() else [0, 0, 0])
    feats = np.array(feats)
    silent = np.array(silent)

    scores = [None] * n_windows
    z_all = np.full((n_windows, 3), np.nan)
    voiced = np.where(~silent)[0]
    if len(voiced) >= 1:
        z = np.stack([robust_z(feats[voiced, k]) for k in range(3)], axis=1)
        s = z_to_score(np.abs(z).max(axis=1))
        for j, w in enumerate(voiced):
            scores[w] = float(s[j])
        z_all[voiced] = z
    return AudioResult(window_scores=scores, envelope=_envelope(x), n_windows=n_windows,
                       feats=np.where(silent[:, None], np.nan, feats), z=z_all)
