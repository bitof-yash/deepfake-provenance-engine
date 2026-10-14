"""Run the existing pipeline over the extracted LAV-DF subset and cache results (resumable).

Run from the backend folder, venv active:
  python -m scripts.extract_features --subset ~/lavdf/subset.json --root ~/lavdf/LAV-DF \
         --out ~/lavdf/features.jsonl --workers 4
Safe to Ctrl+C and rerun: finished clips are skipped.
"""
import argparse
import json
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np


def _init():
    import cv2
    cv2.setNumThreads(1)


def work(entry, root):
    from app import pipeline
    rec = {k: entry[k] for k in ("file", "split", "category", "n_fakes", "fake_periods", "duration")}
    rec["label"] = int(entry["n_fakes"] > 0)
    try:
        c = pipeline.compute(str(pathlib.Path(root) / entry["file"]))
        try:
            vec = np.asarray(c["clip_vector"], dtype=float).ravel().tolist()
        except Exception:
            vec = None
        rec.update(heuristic_score=float(c["heuristic_score"]), clip_vector=vec,
                   windows=c["windows"], analyzed=float(c["analyzed"]), notes=c["notes"])
    except Exception as e:  # keep going; record the failure
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    entries = json.load(open(pathlib.Path(a.subset).expanduser()))
    root = str(pathlib.Path(a.root).expanduser())
    out = pathlib.Path(a.out).expanduser()
    done = set()
    if out.exists():
        done = {json.loads(l)["file"] for l in open(out) if l.strip()}
    todo = [e for e in entries if e["file"] not in done]
    print(f"{len(done)} already done, {len(todo)} to go")
    if not todo:
        return

    # determinism check: same file twice must give the same score
    r1, r2 = work(todo[0], root), work(todo[0], root)
    if "error" in r1:
        raise SystemExit(f"first clip failed: {r1['error']}")
    same = r1["heuristic_score"] == r2["heuristic_score"] and r1["clip_vector"] == r2["clip_vector"]
    print("DETERMINISM CHECK:", "PASS (identical)" if same else
          f"FAIL ({r1['heuristic_score']} vs {r2['heuristic_score']}) - fix before trusting any numbers")

    t0, n_done, n_err = time.time(), 0, 0
    with open(out, "a") as f, ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"),
                                                  initializer=_init) as ex:
        futs = [ex.submit(work, e, root) for e in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n_done += 1
            n_err += "error" in rec
            if n_done % 25 == 0 or n_done == len(todo):
                rate = n_done / (time.time() - t0)
                print(f"{n_done}/{len(todo)}  errors={n_err}  {rate:.2f} clips/s  "
                      f"ETA {(len(todo) - n_done) / rate / 60:.0f} min")


if __name__ == "__main__":
    main()
