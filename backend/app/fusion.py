"""Per-window fusion of video / audio / lip-sync scores and segment localisation.

MVP stand-in for the learned cross-attention fusion module: each window's
modality scores are combined with softmax attention weights driven by how
confident each modality is (distance from 0.5), scaled by modality priors,
plus a boost when several modalities agree. A trained cross-attention model
(e.g. on FakeAVCeleb / LAV-DF features) can replace `fuse_window` later
without changing the rest of the pipeline.
"""
from collections import Counter

import numpy as np

from . import config


def fuse_window(scores: dict):
    """scores: {"video": float|None, "audio": float|None, "lipsync": float|None}"""
    avail = {k: v for k, v in scores.items() if v is not None}
    if not avail:
        return None, "none"
    keys = list(avail)
    conf = np.array([abs(avail[k] - 0.5) * 2 for k in keys])
    logits = np.array([config.MODALITY_PRIORS[k] for k in keys]) * (1.0 + 3.0 * conf)
    w = np.exp(logits - logits.max())
    w /= w.sum()
    fused = float(sum(wi * avail[k] for wi, k in zip(w, keys)))

    high = [k for k in keys if avail[k] >= config.FLAG_THRESHOLD]
    if len(high) >= 2:
        fused = min(1.0, fused + config.AGREEMENT_BOOST * (len(high) - 1))
    # Strong single-modality evidence must not be diluted by clean modalities
    # (e.g. fake audio over real video, or the reverse).
    strong = [avail[k] for k in ("video", "audio") if k in avail and avail[k] >= config.FLAG_THRESHOLD]
    if strong:
        fused = max(fused, max(strong))
    if len(high) == 0:
        dominant = "none"
    elif len(high) == 1:
        dominant = high[0]
    elif "lipsync" in high:
        dominant = "cross-modal"
    else:
        dominant = "audio+video"
    return fused, dominant


def build_windows(video, audio, lipsync, duration: float) -> list:
    n = len(video)
    windows = []
    for i in range(n):
        s = {
            "video": video[i],
            "audio": audio[i] if audio is not None and i < len(audio) else None,
            "lipsync": lipsync[i] if lipsync is not None and i < len(lipsync) else None,
        }
        fused, dominant = fuse_window(s)
        windows.append({
            "start": round(i * config.WINDOW_SECONDS, 2),
            "end": round(min((i + 1) * config.WINDOW_SECONDS, duration) if duration else (i + 1), 2),
            "video": None if s["video"] is None else round(s["video"], 3),
            "audio": None if s["audio"] is None else round(s["audio"], 3),
            "lipsync": None if s["lipsync"] is None else round(s["lipsync"], 3),
            "fused": None if fused is None else round(fused, 3),
            "dominant": dominant,
            "flagged": fused is not None and fused >= config.FLAG_THRESHOLD,
        })
    return windows


def segments_from_windows(windows: list) -> list:
    segments, cur = [], None
    for w in windows:
        if w["flagged"]:
            if cur is None:
                cur = {"start": w["start"], "end": w["end"], "peak": w["fused"], "mods": [w["dominant"]]}
            else:
                cur["end"] = w["end"]
                cur["peak"] = max(cur["peak"], w["fused"])
                cur["mods"].append(w["dominant"])
        elif cur is not None:
            segments.append(cur)
            cur = None
    if cur is not None:
        segments.append(cur)
    out = []
    for s in segments:
        modality = Counter(s["mods"]).most_common(1)[0][0]
        out.append({"start": s["start"], "end": s["end"], "modality": modality, "peak_score": round(s["peak"], 3)})
    return out


def global_score(windows: list) -> float:
    vals = sorted((w["fused"] for w in windows if w["fused"] is not None), reverse=True)
    if not vals:
        return 0.0
    k = max(1, int(np.ceil(0.2 * len(vals))))
    return float(0.5 * vals[0] + 0.5 * np.mean(vals[:k]))
