"""top20 rule: pick thresholds on FakeAV speaker-half A, report on FakeAV half B and LAV-DF test."""
import json, math, pathlib, zlib
import numpy as np, joblib
from sklearn.metrics import roc_curve

H = pathlib.Path.home(); T = H / "lavdf_transfer"
model = joblib.load(pathlib.Path(__file__).resolve().parent.parent / "models" / "audio_window.joblib")

def top20(a):
    p = model.predict_proba(a)[:, 1]
    if len(p) >= 3:
        p = np.convolve(p, np.ones(3) / 3, mode="same")
    q = np.sort(p)[::-1]
    return float(q[: max(1, math.ceil(0.2 * len(q)))].mean())

ly, ls = [], []
for e in json.load(open(H / "lavdf_big" / "subset.json")):
    if e["split"] != "test" or e["category"] not in ("real", "fake_audio", "fake_av"):
        continue
    f = H / "lavdf_big" / "emb" / "test" / (pathlib.Path(e["file"]).stem + ".npz")
    if f.exists():
        ly.append(int(e["category"] != "real")); ls.append(top20(np.load(f)["a"].astype(np.float32)))
ly, ls = np.array(ly), np.array(ls)

fy, fs, spk = [], [], []
for r in json.load(open(T / "fakeav_manifest.json")):
    f = T / "fakeav_embeddings" / f"{r['key']}.npz"
    if not f.exists():
        continue
    with np.load(f) as d:
        a, has = np.asarray(d["a"], np.float32), bool(np.asarray(d["has_audio"]).item())
    if not has or a.ndim != 2 or len(a) == 0:
        continue
    fy.append(int("fakeaudio" in r["type"].lower().replace("_", "-").replace(" ", "")))
    fs.append(top20(a)); spk.append(r["file"].split("/")[-2])
fy, fs = np.array(fy), np.array(fs)
half = np.array([zlib.crc32(x.encode()) % 2 for x in spk]); A, B = half == 0, half == 1

rate = lambda s, y, t: f"{np.mean(s[y==0]>=t):5.1%} / {np.mean(s[y==1]>=t):5.1%}"
print("target FPR on A | thr | FakeAV B (real/fake flagged) | LAV-DF test (real/fake flagged)")
fpr, _, thr = roc_curve(fy[A], fs[A])
for tgt in (0.01, 0.02, 0.05, 0.10):
    t = thr[np.searchsorted(fpr, tgt, side="right") - 1]
    print(f"{tgt:>5.0%} | {t:.3f} | {rate(fs[B], fy[B], t)} | {rate(ls, ly, t)}")
