"""Learned audio detector: wav2vec2-base window embeddings (0.25 s) -> logistic regression.

Loaded lazily, so the API still starts (and falls back to the heuristics) if torch, transformers
or the model file are missing. Mirrors scripts/extract_embeddings.py (same pooling) and
scripts/train_windows.py (same 3-window smoothing). Set DFPE_EMBED=0 to disable it.
"""
import os
import pathlib
import threading

import numpy as np

from . import config

MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "models" / "audio_window.joblib"
W2V_NAME = "facebook/wav2vec2-base"
_lock = threading.Lock()
_state = {}


def enabled() -> bool:
    return os.environ.get("DFPE_EMBED", "1") != "0" and MODEL_PATH.exists()


def _load():
    with _lock:
        if "clf" not in _state:
            import joblib
            import torch
            from transformers import Wav2Vec2Model
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
            _state["torch"], _state["device"] = torch, device
            _state["w2v"] = Wav2Vec2Model.from_pretrained(W2V_NAME).to(device).eval()
            _state["clf"] = joblib.load(MODEL_PATH)
    return _state


def window_embeddings(wav: np.ndarray) -> np.ndarray:
    s = _load()
    torch, w = s["torch"], config.EMBED_WINDOW_SECONDS
    x = (wav - wav.mean()) / (wav.std() + 1e-7)
    t = torch.from_numpy(x.astype(np.float32))[None].to(s["device"])
    with torch.no_grad():
        frames = s["w2v"](t).last_hidden_state[0].float().cpu().numpy()      # (T, 768) at 50 Hz
    n = max(1, int(len(wav) / config.AUDIO_SR // w))
    out = np.zeros((n, frames.shape[1]), dtype=np.float32)
    for i in range(n):
        lo = int(i * w / 0.02)
        hi = max(lo + 1, int((i + 1) * w / 0.02))
        seg = frames[min(lo, len(frames) - 1): min(hi, len(frames))]
        out[i] = (seg if len(seg) else frames[-1:]).mean(axis=0)
    return out.astype(np.float16).astype(np.float32)     # training used float16 embeddings


def detect(wav):
    """Return {score, windows, segments, analyzed} or None if the learned model can't be used."""
    if not enabled() or wav is None or len(wav) < config.AUDIO_SR // 2:
        return None
    wav = np.asarray(wav, dtype=np.float32)[: config.MAX_ANALYZE_SECONDS * config.AUDIO_SR]
    p = _load()["clf"].predict_proba(window_embeddings(wav))[:, 1]
    if len(p) >= 3:
        p = np.convolve(p, np.ones(3) / 3, mode="same")
    thr, w = config.EMBED_FLAG_THRESHOLD, config.EMBED_WINDOW_SECONDS

    segments, start = [], None
    for i, flag in enumerate(list(p > thr) + [False]):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            segments.append({"start": round(start * w, 2), "end": round(i * w, 2), "modality": "audio",
                             "peak_score": round(float(p[start:i].max()), 3)})
            start = None

    per = max(1, int(round(config.WINDOW_SECONDS / w)))
    dur = len(p) * w
    windows = []
    for k in range(int(np.ceil(len(p) / per))):
        v = float(p[k * per:(k + 1) * per].max())
        windows.append({"start": round(k * config.WINDOW_SECONDS, 2),
                        "end": round(min((k + 1) * config.WINDOW_SECONDS, dur), 2),
                        "video": None, "audio": round(v, 3), "lipsync": None, "fused": round(v, 3),
                        "dominant": "audio" if v >= thr else "none", "flagged": v >= thr})
    return {"score": float(p.max()), "windows": windows, "segments": segments, "analyzed": dur}
