"""Score FakeAVCeleb with the API's rule (3-window smoothing -> max) using cached audio embeddings.
Speaker-disjoint threshold selection: pick on half A, report on half B."""
import json, pathlib, zlib
import numpy as np, joblib
from sklearn.metrics import roc_auc_score, roc_curve

T = pathlib.Path.home() / "lavdf_transfer"
model = joblib.load(pathlib.Path(__file__).resolve().parent.parent / "models" / "audio_window.joblib")
rows = json.load(open(T / "fakeav_manifest.json"))

y, s_max, s_mean, spk = [], [], [], []
for r in rows:
    f = T / "fakeav_embeddings" / f"{r['key']}.npz"
    if not f.exists():
        continue
    with np.load(f) as d:
        a, has = np.asarray(d["a"], np.float32), bool(np.asarray(d["has_audio"]).item())
    if not has or a.ndim != 2 or len(a) == 0:
        continue
    p = model.predict_proba(a)[:, 1]
    s_mean.append(float(p.mean()))
    if len(p) >= 3:
        p = np.convolve(p, np.ones(3) / 3, mode="same")
    s_max.append(float(p.max()))
    y.append(int("fakeaudio" in r["type"].lower().replace("_", "-").replace(" ", "")))
    spk.append(r["file"].split("/")[-2])
y, s_max, s_mean = np.array(y), np.array(s_max), np.array(s_mean)
half = np.array([zlib.crc32(x.encode()) % 2 for x in spk])
print(f"{len(y)} clips, fake_audio={int(y.sum())}, speakers={len(set(spk))}")
for name, s in (("API rule (smoothed max)", s_max), ("mean prob (old eval)", s_mean)):
    print(f"\n{name}: AUC {roc_auc_score(y, s):.3f}")
    for thr in (0.4, 0.5, 0.6):
        print(f"  thr {thr}: real flagged {np.mean(s[y==0]>=thr):.1%}, fake flagged {np.mean(s[y==1]>=thr):.1%}")

s = s_max
A, B = half == 0, half == 1
print(f"\nSpeaker-disjoint halves: AUC A {roc_auc_score(y[A], s[A]):.3f}, B {roc_auc_score(y[B], s[B]):.3f}")
fpr, tpr, thr = roc_curve(y[A], s[A])
for target in (0.01, 0.05, 0.10):
    t = thr[np.searchsorted(fpr, target, side="right") - 1]
    print(f"threshold {t:.3f} (FPR<={target:.0%} on A) -> on B: real flagged {np.mean(s[B & (y==0)]>=t):.1%}, "
          f"fake flagged {np.mean(s[B & (y==1)]>=t):.1%}")
