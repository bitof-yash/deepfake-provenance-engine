"""Visual-stream analysis: per-window artifact scores + mouth-motion envelope.

MVP heuristics (no trained weights): within-clip outliers in
  * high-frequency spectral energy of the face region,
  * face-region sharpness (Laplacian variance),
  * temporal frame-difference energy.
A window that deviates strongly from the rest of the same clip is suspicious.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from . import config
from .scoring import robust_z, z_to_score

_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_HANN = np.outer(np.hanning(64), np.hanning(64)).astype(np.float32)
_YY, _XX = np.indices((64, 64))
_RAD = np.hypot(_YY - 32, _XX - 32) / 32.0


@dataclass
class VideoResult:
    window_scores: list          # list[float | None], one per 1 s window
    mouth_motion: np.ndarray     # per-frame lower-face motion (len n_frames) or empty
    face_found: bool
    n_frames: int
    feats: np.ndarray            # raw per-window features (n_windows, 3)
    z: np.ndarray                # robust within-clip z-scores of those features


def _hf_ratio(roi64: np.ndarray) -> float:
    r = roi64.astype(np.float32)
    r = (r - r.mean()) * _HANN
    power = np.abs(np.fft.fftshift(np.fft.fft2(r))) ** 2
    return float(power[_RAD > 0.5].sum() / (power.sum() + 1e-9))


def _detect_face(frame: np.ndarray):
    faces = _CASCADE.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def analyze(frames: np.ndarray) -> VideoResult:
    n, H, W = frames.shape
    bbox = None
    ever_found = False
    hf = np.zeros(n)
    sharp = np.zeros(n)
    tdiff = np.zeros(n)
    mouth = np.zeros(n)
    prev64 = None

    for i in range(n):
        if i % config.FACE_DETECT_EVERY == 0:
            found = _detect_face(frames[i])
            if found is not None:
                bbox = found
                ever_found = True
        if bbox is not None:
            x, y, w, h = bbox
            roi = frames[i, max(y, 0): y + h, max(x, 0): x + w]
        else:  # no face: analyse the central region instead
            roi = frames[i, H // 6: 5 * H // 6, W // 6: 5 * W // 6]
        if roi.size == 0:
            roi = frames[i]
        roi64 = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
        hf[i] = _hf_ratio(roi64)
        sharp[i] = np.log1p(cv2.Laplacian(roi64, cv2.CV_32F).var())
        if prev64 is not None:
            d = np.abs(roi64.astype(np.float32) - prev64.astype(np.float32))
            tdiff[i] = d.mean()
            mouth[i] = d[int(64 * 0.6):, :].mean()
        prev64 = roi64

    spw = int(round(config.WINDOW_SECONDS * config.FRAME_FPS))
    n_windows = max(1, n // spw)
    feats = np.zeros((n_windows, 3))
    for w_i in range(n_windows):
        sl = slice(w_i * spw, (w_i + 1) * spw if w_i < n_windows - 1 else n)
        feats[w_i] = [np.median(hf[sl]), np.median(sharp[sl]), np.mean(tdiff[sl])]

    z = np.stack([robust_z(feats[:, k]) for k in range(3)], axis=1)
    scores = z_to_score(np.abs(z).max(axis=1))
    return VideoResult(
        window_scores=[float(s) for s in scores],
        mouth_motion=mouth if ever_found else np.zeros(0),
        face_found=ever_found,
        n_frames=n,
        feats=feats,
        z=z,
    )
