"""Checks the cost budget against the results on disk.

  1. A cell with a score is not also recorded as refused.
  2. A cell that ran is not projected to exceed the memory cap.
  3. Fewer workers always means more time and less memory.

    PYTHONPATH=. python script_funcs/budget_sanity.py [--diff-budget HOURS]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from src.resultdirs import MAIN

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from src.budget import MEMORY_MARGIN, default_memory_cap


def load(result_dir):
    for path in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        name = os.path.basename(path)[:-5]
        if name.startswith("_"):
            continue
        with open(path) as f:
            yield name, json.load(f)


def check_no_shadowing(name, data, failures):
    """A method is refused on a dataset or it is not; a per-fold split verdict
    leaves a mean over a different number of folds from the row beside it."""
    for model, folds in data.get("_skipped", {}).items():
        for fold in folds:
            if fold in data.get(model, {}):
                failures.append(
                    f"{name}/{model}/{fold}: hem skoru var hem reddedilmis")
        scored = data.get(model, {})
        if scored and folds:
            failures.append(
                f"{name}/{model}: {len(scored)} fold kosmus ama {len(folds)} "
                f"fold reddedilmis -- karar fold basina degil hucre basina olmali")


def check_memory_model(name, data, cap, failures):
    """A cell that ran cannot be impossible. Re-derives the projection from the
    probe stored with it, so a later change to the memory model is caught."""
    for model, folds in data.get("_metrics", {}).items():
        for fold, rec in folds.items():
            probe = rec.get("projection", {}).get("probe", {})
            if "fixed_gb" not in probe:
                continue
            gb = probe["fixed_gb"] + MEMORY_MARGIN * probe["grows_gb"]
            if gb > cap:
                failures.append(
                    f"{name}/{model}/{fold}: kosmus ama bellek modeli tek "
                    f"isciyle bile {gb:.1f} GB diyor, kap {cap:.1f} GB")


def check_throttling_monotone(name, data, failures):
    for model, folds in data.get("_metrics", {}).items():
        for fold, rec in folds.items():
            probe = rec.get("projection", {}).get("probe", {})
            if "grid_seconds_serial" not in probe:
                continue
            times, mems = [], []
            for workers in (1, 2, 5):
                times.append(probe["grid_seconds_serial"] / workers
                             + probe["serial_seconds"])
                mems.append(probe["fixed_gb"]
                            + MEMORY_MARGIN * probe["grows_gb"] * workers)
            if not (times[0] >= times[1] >= times[2]):
                failures.append(f"{name}/{model}/{fold}: sure isci sayisiyla "
                                f"monoton azalmiyor: {times}")
            if not (mems[0] <= mems[1] <= mems[2]):
                failures.append(f"{name}/{model}/{fold}: bellek isci sayisiyla "
                                f"monoton artmiyor: {mems}")


def diff_budget(result_dir, budget):
    """Which cells that ran would a different time budget have refused."""
    print(f"\n{budget} saatlik butce ile reddedilecek olanlar:")
    hit = 0
    for name, data in load(result_dir):
        for model, folds in data.get("_metrics", {}).items():
            for fold, rec in folds.items():
                hours = rec.get("projection", {}).get("projected_hours")
                if hours is not None and hours > budget:
                    hit += 1
                    print(f"  {name}/{model}/{fold}: projeksiyon {hours:.2f} saat")
    if not hit:
        print("  hicbiri")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=os.path.basename(MAIN))
    ap.add_argument("--memory-cap-gb", type=float, default=default_memory_cap())
    ap.add_argument("--diff-budget", type=float, default=None)
    args = ap.parse_args()
    result_dir = os.path.join(REPO_ROOT, "logs", args.results_dir)

    failures: list[str] = []
    problems = 0
    for name, data in load(result_dir):
        problems += 1
        check_no_shadowing(name, data, failures)
        check_memory_model(name, data, args.memory_cap_gb, failures)
        check_throttling_monotone(name, data, failures)

    print(f"{args.results_dir}: {problems} problem, bellek kapi "
          f"{args.memory_cap_gb} GB")
    if args.diff_budget is not None:
        diff_budget(result_dir, args.diff_budget)
    if failures:
        print(f"\nBASARISIZ: {len(failures)} degismez ihlali")
        for f in failures[:20]:
            print(f"  {f}")
        return 1
    print("gecti: skor golgeleme yok, kosmus hucre imkansiz ilan edilmiyor, "
          "isci-kisma monoton")
    return 0


if __name__ == "__main__":
    sys.exit(main())
