import json
import time
from pathlib import Path

import cv2
import numpy as np

from app import media
from scripts.extract_embeddings import Encoders, face_crops

ROOT = Path("~/fakeavceleb/FakeAVCeleb_v1.2").expanduser()
OUT = Path("~/lavdf_transfer").expanduser()
MANIFEST = OUT / "fakeav_manifest.json"
EMB_DIR = OUT / "fakeav_embeddings"
WINDOW = 0.25


def duration_seconds(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"Cannot read video duration: {path}")

    return frames / fps


def extract_one(path, enc):
    duration = duration_seconds(path)
    n = max(1, int(duration // WINDOW))
    centres = (np.arange(n) + 0.5) * WINDOW

    crops, face_frac = face_crops(str(path), centres)
    video = enc.video(crops).astype(np.float16)

    wav = media.extract_audio(str(path))
    audio = np.zeros((n, 768), dtype=np.float32)
    has_audio = wav is not None and len(wav) > 0

    if has_audio:
        audio_frames = enc.audio(wav)
        for i in range(n):
            lo = int(i * WINDOW / 0.02)
            hi = max(lo + 1, int((i + 1) * WINDOW / 0.02))
            segment = audio_frames[
                min(lo, len(audio_frames) - 1):min(hi, len(audio_frames))
            ]
            if len(segment) == 0:
                segment = audio_frames[-1:]
            audio[i] = segment.mean(axis=0)

    return audio.astype(np.float16), video, has_audio, float(face_frac)


def main():
    records = json.loads(MANIFEST.read_text())
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    todo = [
        r for r in records
        if not (EMB_DIR / f"{r['key']}.npz").exists()
    ]

    print(f"Total manifest videos: {len(records)}")
    print(f"Already cached: {len(records) - len(todo)}")
    print(f"Remaining: {len(todo)}")

    if not todo:
        print("All embeddings are already cached.")
        return

    enc = Encoders(dry_run=False)
    print(f"Encoder device: {enc.device}")

    errors_file = OUT / "fakeav_extraction_errors.jsonl"
    start = time.time()
    errors = 0

    with errors_file.open("a") as errors_out:
        for i, record in enumerate(todo, 1):
            path = ROOT / record["file"]

            try:
                a, v, has_audio, face_frac = extract_one(path, enc)
                np.savez_compressed(
                    EMB_DIR / f"{record['key']}.npz",
                    a=a,
                    v=v,
                    has_audio=has_audio,
                    face_frac=face_frac,
                    window=WINDOW,
                )
            except Exception as exc:
                errors += 1
                errors_out.write(json.dumps({
                    "file": record["file"],
                    "error": f"{type(exc).__name__}: {exc}",
                }) + "\n")
                errors_out.flush()
                print(f"ERROR: {record['file']}: {exc}")

            if i % 25 == 0 or i == len(todo):
                elapsed = max(time.time() - start, 0.001)
                rate = i / elapsed
                eta_min = (len(todo) - i) / rate / 60
                print(
                    f"{i}/{len(todo)} | errors={errors} | "
                    f"{rate:.2f} videos/sec | ETA {eta_min:.1f} min"
                )

    print("Extraction run finished.")
    print(f"Errors logged in: {errors_file}")
    print("Rerun this script to resume any videos that were not cached.")


if __name__ == "__main__":
    main()
