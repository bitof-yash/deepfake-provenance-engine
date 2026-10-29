"""Balanced FakeAVCeleb subsample -> identical re-encode -> metadata shortcut check -> score LAV-DF models.
Resumable. Run: python -m scripts.fakeav_normalized"""
import collections, hashlib, json, pathlib, random, subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from scripts.extract_fakeav_embeddings import Encoders, extract_one

SRC = pathlib.Path("~/fakeavceleb/FakeAVCeleb_v1.2").expanduser()
OUT = pathlib.Path("~/lavdf_transfer").expanduser()
NORM, EMB = OUT / "fakeav_norm", OUT / "fakeav_norm" / "emb"
PER_TYPE, DUR = 500, 4.0
TYPES = ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]


def probe(p):
    d = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(p)],
                                  capture_output=True, text=True).stdout or "{}")
    v = next((s for s in d.get("streams", []) if s["codec_type"] == "video"), {})
    a = next((s for s in d.get("streams", []) if s["codec_type"] == "audio"), {})
    f = lambda x: float(x) if x not in (None, "N/A") else np.nan
    dur = f(d.get("format", {}).get("duration"))
    return dict(a_sr=f(a.get("sample_rate")), a_bitrate=f(a.get("bit_rate")), channels=f(a.get("channels")),
                a_dur=f(a.get("duration")), v_dur=f(v.get("duration")), v_bitrate=f(v.get("bit_rate")),
                dur_gap=f(a.get("duration")) - f(v.get("duration")),
                size_per_s=p.stat().st_size / max(dur, 0.1) if dur == dur else np.nan,
                dur=dur, has_a=bool(a), has_v=bool(v))


def normalise(r):
    dst = NORM / f"{r['key']}.mp4"
    if not dst.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(SRC / r["file"]), "-t", str(DUR), "-map_metadata", "-1",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-r", "25",
                        "-vf", "scale=-2:360", "-c:a", "aac", "-ar", "16000", "-ac", "1", "-b:a", "64k", str(dst)],
                       capture_output=True)
    return r, dst


def auc(y, s):
    y = np.asarray(y)
    return roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")


def main():
    NORM.mkdir(parents=True, exist_ok=True); EMB.mkdir(exist_ok=True)
    recs = json.load(open(OUT / "fakeav_manifest.json"))
    by = collections.defaultdict(list)
    for r in recs:
        by[r["type"]].append(r)
    sub = []
    for t in TYPES:
        random.Random(0).shuffle(by[t]); sub += by[t][:PER_TYPE]
    print(f"subsample: {len(sub)} clips, {PER_TYPE} per type")

    # 1. normalise
    with ThreadPoolExecutor(4) as ex:
        done = list(ex.map(normalise, sub))
    rows = []
    for r, dst in done:
        if not dst.exists():
            continue
        m = probe(dst)
        if m["has_a"] and m["has_v"] and m["dur"] >= DUR - 0.2:
            rows.append({**r, **m, "norm": str(dst), "source": pathlib.Path(r["file"]).parent.name,
                         "fake_audio": int("FakeAudio" in r["type"])})
    print(f"{len(rows)} usable normalised clips (dropped {len(sub) - len(rows)} too short / broken)")
    print("per type:", dict(collections.Counter(r["type"] for r in rows)))

    # 2. metadata shortcut check on NORMALISED clips (should be ~0.5)
    feats = ["a_sr", "a_bitrate", "channels", "a_dur", "v_dur", "dur_gap", "v_bitrate", "size_per_s"]
    X = np.nan_to_num(np.array([[r[f] for f in feats] for r in rows], float))
    y = np.array([r["fake_audio"] for r in rows]); g = np.array([r["source"] for r in rows])
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        s[te] = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    print(f"\nMETADATA-ONLY AUC on normalised clips: {auc(y, s):.3f}   (was 0.993 raw; ~0.5 = shortcut gone)")

    # 3. embeddings
    enc, todo = None, [r for r in rows if not (EMB / f"{r['key']}.npz").exists()]
    print(f"\nembeddings: {len(rows) - len(todo)} cached, {len(todo)} to go")
    if todo:
        enc = Encoders(dry_run=False)
        for i, r in enumerate(todo, 1):
            try:
                a, v, ha, ff = extract_one(pathlib.Path(r["norm"]), enc)
                np.savez_compressed(EMB / f"{r['key']}.npz", a=a, v=v, has_audio=ha, face_frac=ff, window=0.25)
            except Exception as e:
                print("ERROR", r["file"], e)
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}")

    # 4. score with LAV-DF-trained RAW models (clipnorm is structurally wrong for whole-clip fakes)
    print("\nTRANSFER AUC on normalised FakeAVCeleb (target = fake audio)")
    for name, mod in (("audio", "audio"), ("both", "both")):
        model = joblib.load(OUT / "models" / f"{name}.joblib")
        sc, keep = [], []
        for r in rows:
            p = EMB / f"{r['key']}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            x = d["a"].astype(np.float32) if mod == "audio" else np.concatenate([d["a"], d["v"]], 1).astype(np.float32)
            sc.append(float(model.predict_proba(x)[:, 1].mean())); keep.append(r)
        yy = np.array([r["fake_audio"] for r in keep]); ss = np.array(sc); tt = np.array([r["type"] for r in keep])
        fv = np.char.startswith(tt, "FakeVideo"); rv = ~fv
        print(f"  {name:<6} ALL {auc(yy, ss):.3f} | FakeVideo FVFA vs FVRA {auc(yy[fv], ss[fv]):.3f} "
              f"| RealVideo RVFA vs RVRA {auc(yy[rv], ss[rv]):.3f}  (n={len(yy)})")
    json.dump([{k: v for k, v in r.items()} for r in rows], open(NORM / "norm_rows.json", "w"), default=str)


if __name__ == "__main__":
    main()
