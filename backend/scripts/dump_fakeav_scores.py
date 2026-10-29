"""Dump per-clip scores (mean of per-window fake prob) for each model, for stratified re-scoring."""
import json, pathlib
import numpy as np
import joblib

OUT = pathlib.Path("~/lavdf_transfer").expanduser()
EMB = OUT / "fakeav_embeddings"
recs = json.load(open(OUT / "fakeav_manifest.json"))


def feats(a, v, modality, clipnorm):
    parts = []
    for key in (("a",) if modality == "audio" else ("v",) if modality == "video" else ("a", "v")):
        x = (a if key == "a" else v).astype(np.float32)
        parts.append(x - x.mean(axis=0, keepdims=True) if clipnorm else x)
    return np.concatenate(parts, axis=1)


preds = {}
for name in ("audio", "audio_clipnorm", "both", "both_clipnorm"):
    modality = name.split("_")[0]
    clipnorm = name.endswith("clipnorm")
    model = joblib.load(OUT / "models" / f"{name}.joblib")
    files, scores, labels_audio = [], [], []
    for r in recs:
        p = EMB / f"{r['key']}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        if not bool(d["has_audio"]) and modality != "video":
            continue
        pw = model.predict_proba(feats(d["a"], d["v"], modality, clipnorm))[:, 1]
        files.append(r["file"])
        scores.append(float(pw.mean()))
        labels_audio.append(int("FakeAudio" in r["type"]))
    preds[name] = {"files": files, "scores": scores, "labels_audio": labels_audio}
    print(name, len(scores))

(OUT / "fakeav_predictions.json").write_text(json.dumps(preds))
print("saved", OUT / "fakeav_predictions.json")
