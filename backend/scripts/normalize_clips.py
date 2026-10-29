"""Re-encode every clip (real and fake) with identical ffmpeg settings so container and codec
fingerprints cannot separate real from fake. Resumable: rerun it and it skips finished clips."""
import argparse, json, pathlib, subprocess, time
from concurrent.futures import ThreadPoolExecutor


def run(e, root, out):
    src, dst = root / e["file"], out / e["file"]
    if dst.exists() and dst.stat().st_size > 0:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?",
           "-t", f"{e['duration']:.3f}", "-shortest",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "64k", "-ar", "16000", "-ac", "1", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists():
        return "ERR " + r.stderr[-200:]
    tmp.rename(dst)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    entries = json.load(open(pathlib.Path(a.subset).expanduser()))
    root, out = pathlib.Path(a.root).expanduser(), pathlib.Path(a.out).expanduser()
    t0, errs = time.time(), 0
    with ThreadPoolExecutor(a.workers) as ex:
        for k, res in enumerate(ex.map(lambda e: run(e, root, out), entries), 1):
            if res.startswith("ERR"):
                errs += 1
                print(res)
            if k % 100 == 0 or k == len(entries):
                rate = k / (time.time() - t0)
                print(f"{k}/{len(entries)} errors={errs} {rate:.1f} clips/s ETA {(len(entries)-k)/rate/60:.0f} min")


if __name__ == "__main__":
    main()
