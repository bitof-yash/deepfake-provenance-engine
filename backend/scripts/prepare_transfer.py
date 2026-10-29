import csv
import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Existing project dataset paths
LAVDF_SUBSET = Path("~/lavdf_big/subset_small.json").expanduser()
LAVDF_EMB = Path("~/lavdf_norm/emb").expanduser()
FAKEAV_ROOT = Path("~/fakeavceleb/FakeAVCeleb_v1.2").expanduser()
FAKEAV_CSV = FAKEAV_ROOT / "meta_data.csv"
OUT = Path("~/lavdf_transfer").expanduser()

MAX_WINDOWS = 120_000
RNG = np.random.default_rng(0)


def overlap(start, end, periods):
    return sum(
        max(0.0, min(end, p1) - max(start, p0))
        for p0, p1 in periods
    )


def load_lavdf_train():
    entries = json.loads(LAVDF_SUBSET.read_text())
    clips = []
    missing = 0

    for entry in entries:
        if entry.get("split") != "train":
            continue

        emb_path = LAVDF_EMB / "train" / (
            Path(entry["file"]).stem + ".npz"
        )
        if not emb_path.exists():
            missing += 1
            continue

        with np.load(emb_path) as data:
            audio = data["a"].astype(np.float32)
            video = data["v"].astype(np.float32)
            window = float(data["window"])

        periods = [
            tuple(map(float, p))
            for p in entry.get("fake_periods", [])
        ]
        labels = np.array([
            int(
                overlap(i * window, (i + 1) * window, periods)
                >= 0.5 * window
            )
            for i in range(len(audio))
        ], dtype=np.int64)

        if len(audio) != len(video) or len(audio) != len(labels):
            raise ValueError(f"Length mismatch: {entry['file']}")

        clips.append({
            "file": entry["file"],
            "audio": audio,
            "video": video,
            "labels": labels,
        })

    if not clips:
        raise RuntimeError(
            "No LAV-DF train embeddings found. Check the dataset paths."
        )

    print(f"LAV-DF train clips loaded: {len(clips)}")
    print(f"Missing embeddings: {missing}")
    return clips


def features(clip, modality, clipnorm):
    keys = {
        "audio": ["audio"],
        "video": ["video"],
        "both": ["audio", "video"],
    }[modality]

    arrays = []
    for key in keys:
        x = clip[key]
        if clipnorm:
            x = x - x.mean(axis=0, keepdims=True)
        arrays.append(x)

    return np.concatenate(arrays, axis=1)


def train_models():
    clips = load_lavdf_train()
    model_dir = OUT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for modality in ("audio", "video", "both"):
        for clipnorm in (False, True):
            name = modality + ("_clipnorm" if clipnorm else "")
            X_parts, y_parts = [], []

            for clip in clips:
                X_parts.append(features(clip, modality, clipnorm))
                y_parts.append(clip["labels"])

            X = np.concatenate(X_parts, axis=0)
            y = np.concatenate(y_parts, axis=0)

            if len(y) > MAX_WINDOWS:
                keep = RNG.choice(len(y), MAX_WINDOWS, replace=False)
                X, y = X[keep], y[keep]

            if len(np.unique(y)) < 2:
                raise RuntimeError(f"{name}: only one label class found")

            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=300,
                    C=0.1,
                    class_weight="balanced",
                ),
            )

            print(
                f"Training {name}: {len(y)} windows, "
                f"{int(y.sum())} fake windows"
            )
            model.fit(X, y)

            destination = model_dir / f"{name}.joblib"
            joblib.dump(model, destination)
            print(f"Saved: {destination}")

    metadata = {
        "training_dataset": "LAV-DF",
        "training_split": "train only",
        "fakeavceleb_used_for_training": False,
        "window_seconds": 0.25,
        "max_training_windows_per_model": MAX_WINDOWS,
        "models": [
            "audio", "audio_clipnorm",
            "video", "video_clipnorm",
            "both", "both_clipnorm",
        ],
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )


def prepare_fakeav_manifest():
    if not FAKEAV_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {FAKEAV_CSV}")

    # Group duplicate CSV entries by their actual video path.
    grouped = defaultdict(list)

    with FAKEAV_CSV.open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            folder = (row.get("") or "").replace("\\", "/")
            folder = folder.removeprefix("FakeAVCeleb/").strip("/")
            relpath = (Path(folder) / row["path"]).as_posix()
            fullpath = (FAKEAV_ROOT / relpath).resolve()

            if not fullpath.is_file():
                raise FileNotFoundError(f"Video missing: {fullpath}")

            grouped[str(fullpath)].append({
                "file": relpath,
                "type": row["type"],
                "method": row.get("method", ""),
            })

    records, conflicts = [], []

    for fullpath, rows in sorted(grouped.items()):
        labels = {row["type"] for row in rows}

        # Do not silently pick a label if duplicate rows disagree.
        if len(labels) != 1:
            conflicts.append({
                "path": fullpath,
                "types": sorted(labels),
                "rows": len(rows),
            })
            continue

        relpath = sorted({row["file"] for row in rows})[0]
        key = hashlib.sha1(relpath.encode()).hexdigest()[:16]

        records.append({
            "key": key,
            "file": relpath,
            "type": rows[0]["type"],
            "method": rows[0]["method"],
            "duplicate_rows": len(rows),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fakeav_manifest.json").write_text(
        json.dumps(records, indent=2)
    )
    (OUT / "fakeav_conflicts.json").write_text(
        json.dumps(conflicts, indent=2)
    )

    print("\nFakeAVCeleb manifest prepared")
    print(f"Unique videos retained: {len(records)}")
    print(f"Conflicting-label videos excluded: {len(conflicts)}")
    print("Category counts:")
    for category, count in sorted(
        Counter(r["type"] for r in records).items()
    ):
        print(f"  {category}: {count}")

    print(f"\nManifest: {OUT / 'fakeav_manifest.json'}")
    print(f"Models:   {OUT / 'models'}")


if __name__ == "__main__":
    train_models()
    prepare_fakeav_manifest()
    print("\nPreparation complete.")
