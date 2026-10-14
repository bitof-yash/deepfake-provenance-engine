"""Pick a balanced, stratified LAV-DF subset from metadata.json.

Writes subset.json (the chosen metadata entries) and filelist.txt (tar member
names) so only the chosen videos need to be extracted from the 24 GB tar.

Run:  python3 make_subset.py --meta ~/lavdf/LAV-DF/metadata.json --out ~/lavdf
"""
import argparse
import collections
import json
import pathlib
import random


def category(m):
    if m["n_fakes"] == 0:
        return "real"
    mv, ma = m["modify_video"], m["modify_audio"]
    if mv and ma:
        return "fake_av"
    if mv:
        return "fake_video"
    if ma:
        return "fake_audio"
    return "fake_other"


def pick(entries, n, rng):
    by = collections.defaultdict(list)
    for m in entries:
        by[category(m)].append(m)
    n_real = n // 2
    fake_cats = sorted(c for c in by if c != "real")
    chosen = rng.sample(by["real"], min(n_real, len(by["real"])))
    n_fake = n - n_real
    for i, c in enumerate(fake_cats):
        share = n_fake // len(fake_cats) + (1 if i < n_fake % len(fake_cats) else 0)
        chosen += rng.sample(by[c], min(share, len(by[c])))
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    meta = json.load(open(a.meta))
    rng = random.Random(a.seed)
    splits = collections.defaultdict(list)
    for m in meta:
        splits[m["split"]].append(m)
    print({k: len(v) for k, v in splits.items()}, "entries per split")

    chosen = pick(splits["train"], a.n_train, rng) + pick(splits["test"], a.n_test, rng)
    for m in chosen:
        m["category"] = category(m)
    out = pathlib.Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    slim = [{k: m[k] for k in ("file", "split", "category", "n_fakes", "fake_periods", "duration",
                                "modify_video", "modify_audio")} for m in chosen]
    json.dump(slim, open(out / "subset.json", "w"))
    open(out / "filelist.txt", "w").write("".join(f"LAV-DF/{m['file']}\n" for m in chosen))
    for s in ("train", "test"):
        c = collections.Counter(m["category"] for m in slim if m["split"] == s)
        print(s, dict(c))
    print(f"wrote {out/'subset.json'} and {out/'filelist.txt'} ({len(chosen)} videos)")


if __name__ == "__main__":
    main()
