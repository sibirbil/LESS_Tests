"""Heatmap of method ranks per problem with the mean MSE in each cell.

Problems are taken from src.problems. The figure goes to
plots/main/rankings_with_mse.png and the raw table to logs/evidence/.

    PYTHONPATH=. python tools/plot_ranks_mse.py
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.problems import ALL_PROBLEMS
from src.resultdirs import MAIN

METHODS = ["LESS-B", "LESS-A", "TSK", "LGBM", "XGB", "RF", "MLP", "KNN",
           "LOCR", "MAG", "GPR", "ANFIS"]
DEFAULT_OUT = "plots/main/rankings_with_mse.png"


def short(v):
    """A score readable in a 2 cm cell, across targets spanning 1e-2 to 1e9."""
    if v is None or not np.isfinite(v):
        return ""
    if v >= 1e5 or (v < 1e-2 and v > 0):
        m, e = f"{v:.1e}".split("e")
        return f"{m}e{int(e)}"
    if v >= 100:
        return f"{v:.0f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}"


def load(problems):
    rows = {}
    for problem in problems:
        path = os.path.join(MAIN, f"{problem}.json")
        if not os.path.exists(path):
            continue
        data = json.load(open(path))
        scores = {}
        for method in METHODS:
            folds = data.get(method)
            if isinstance(folds, dict) and len(folds) == 4:
                scores[method] = float(np.mean(list(folds.values())))
        if scores:
            rows[problem] = scores
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", default=list(ALL_PROBLEMS),
                    choices=list(ALL_PROBLEMS))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--title", default="Rank among the methods scored on each "
                                       "problem, with mean MSE over four folds")
    args = ap.parse_args()
    OUT = args.out
    rows = load(args.problems)
    problems = list(rows)
    mse = pd.DataFrame(index=problems, columns=METHODS, dtype=float)
    for p, s in rows.items():
        for m, v in s.items():
            mse.loc[p, m] = v
    # the rank is taken among the methods that ran on that problem, so a column
    # of eight and a column of twelve are both ranked 1..n rather than padded
    ranks = mse.rank(axis=1)

    # a method that ran on none of the selected problems would otherwise occupy
    # a column of dashes with a nan average
    mse = mse.dropna(axis=1, how="all")
    ranks = ranks[mse.columns]
    order = ranks.mean().sort_values().index.tolist()
    mse, ranks = mse[order], ranks[order]
    coverage = mse.notna().sum()

    fig, ax = plt.subplots(figsize=(1.05 * len(order) + 3.5, 0.62 * len(problems) + 2.4))
    im = ax.imshow(ranks.values.astype(float), cmap="Blues_r", aspect="auto",
                   vmin=1, vmax=np.nanmax(ranks.values))

    for i, p in enumerate(problems):
        for j, m in enumerate(order):
            r, v = ranks.loc[p, m], mse.loc[p, m]
            if not np.isfinite(r):
                ax.text(j, i, "—", ha="center", va="center", color="#999", fontsize=11)
                continue
            dark = r <= np.nanmax(ranks.values) / 2
            colour = "white" if dark else "#222"
            ax.text(j, i - 0.17, f"{int(r)}", ha="center", va="center",
                    color=colour, fontsize=11, fontweight="bold")
            ax.text(j, i + 0.22, short(v), ha="center", va="center",
                    color=colour, fontsize=7)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{m}\n({coverage[m]})" for m in order], fontsize=9)
    ax.set_yticks(range(len(problems)))
    ax.set_yticklabels([p.replace("_GROUPED", "").lower() for p in problems], fontsize=9)
    ax.set_xlabel("method (problems scored)", fontsize=10)

    avg = ranks.mean()
    for j, m in enumerate(order):
        ax.text(j, -0.85, f"{avg[m]:.2f}", ha="center", va="center",
                fontsize=9, fontweight="bold")
    ax.text(-0.9, -0.85, "avg rank", ha="right", va="center", fontsize=9)

    ax.set_title(args.title, fontsize=11, pad=26)
    ax.set_xticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(problems), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    fig.colorbar(im, ax=ax, shrink=0.55, label="rank")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"yazildi: {OUT}  ({len(problems)} problem, {len(order)} yontem)")

    print("\nortalama sira (kapsama):")
    for m in order:
        wins = int((ranks[m] == 1).sum())
        print(f"  {m:8s} {avg[m]:5.2f}  ({coverage[m]:2d} veri, {wins} birincilik)")
    csv = OUT.replace("plots/main/", "logs/evidence/").replace(".png", ".csv")
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    mse.to_csv(csv)
    print(f"\nham tablo: {csv}")


if __name__ == "__main__":
    main()
