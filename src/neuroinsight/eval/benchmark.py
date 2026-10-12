"""Benchmark the saliency model against MIT1003 human fixation data.

Usage:
    python -m neuroinsight.eval.benchmark --limit 200 --out results/eval.csv

Dataset: Judd, Ehinger, Durand & Torralba (ICCV 2009), "Learning to Predict Where
Humans Look". 1003 natural photographs with eye-tracking from 15 viewers each.
Obtained from https://people.csail.mit.edu/tjudd/WherePeopleLook/

Ground truth used here is the authors' precomputed `_fixMap.jpg` fixation-density
images. That is a legitimate but not maximal choice: these maps are already
Gaussian-blurred, so exact fixation coordinates are unavailable from this archive and
the metric set is slightly less sharp than a raw-coordinate evaluation. We state this
rather than implying we reproduced a canonical benchmark score.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from ..vision import saliency as sal_mod
from . import metrics as M

STIM = "data/eval/ALLSTIMULI"
FIXMAP = "data/eval/ALLFIXATIONMAPS"


def pairs(limit: int | None) -> list[tuple[str, Path, Path]]:
    """Pair each stimulus with its fixation map by basename."""
    stims = sorted(Path(STIM).glob("*.jpeg"))
    out = []
    for s in stims:
        fm = Path(FIXMAP) / f"{s.stem}_fixMap.jpg"
        if fm.exists():
            out.append((s.stem, s, fm))
        if limit and len(out) >= limit:
            break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="neuroinsight.eval.benchmark")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N images (default: all)")
    ap.add_argument("--out", default="results/eval.csv")
    ap.add_argument("--json", default="results/eval_summary.json")
    args = ap.parse_args(argv)

    if not Path(STIM).exists():
        print("ERROR: dataset not found at", STIM)
        print("Fetch it first:")
        print("  bash scripts/fetch_eval_data.sh")
        return 1

    data = pairs(args.limit)
    print(f"paired {len(data)} stimuli with fixation maps\n")
    if not data:
        print("ERROR: no matching pairs found")
        return 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()

    for i, (name, sp, fp) in enumerate(data, 1):
        img = cv2.imread(str(sp), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        if gt.shape[:2] != img.shape[:2]:
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_LINEAR)

        sal, _meta = sal_mod.compute_saliency(img)
        s = M.normalise_saliency(sal)
        sc = M.score(s, gt)
        rows.append({"image": name, **sc})

        if i % 50 == 0 or i == len(data):
            el = time.perf_counter() - t0
            print(f"  {i:4d}/{len(data)}  {el:6.1f}s elapsed")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "auc", "cc", "nss"])
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------------ summary
    def col(k):
        return np.array([r[k] for r in rows if not np.isnan(r[k])], dtype=float)

    summary = {
        "n_images": len(rows),
        "dataset": "MIT1003 (Judd et al., ICCV 2009)",
        "ground_truth": "authors' precomputed _fixMap.jpg fixation-density maps",
        "model": "OpenCV Spectral Residual + Fine-Grained (Hou & Zhang 2007)",
        "runtime_seconds": round(time.perf_counter() - t0, 1),
        "metrics": {},
    }
    print(f"\n{'=' * 62}\nRESULTS  ({len(rows)} images, MIT1003)\n{'=' * 62}")
    print(f"{'metric':8s}{'mean':>10s}{'sd':>9s}{'median':>10s}   interpretation")
    for k in ("auc", "cc", "nss"):
        v = col(k)
        if not len(v):
            continue
        mean = float(v.mean())
        entry = {
            "mean": round(mean, 4),
            "sd": round(float(v.std()), 4),
            "median": round(float(np.median(v)), 4),
            "ci95_halfwidth": round(1.96 * float(v.std()) / np.sqrt(len(v)), 4),
            "interpretation": M.interpret(k, mean),
        }
        summary["metrics"][k] = entry
        print(f"{k.upper():8s}{mean:10.3f}{v.std():9.3f}{np.median(v):10.3f}   "
              f"{entry['interpretation']}")

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out} and {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())