"""Statistical comparison of LESS-B with the other methods.

Friedman test on the methods that ran on every problem, then paired t-tests and
Wilcoxon signed-rank tests of LESS-B against each method, with Holm correction.

    PYTHONPATH=. python script_funcs/posthoc_table.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "script_funcs"))

from anova_many_to_one import BASE, holm, load_observations

from src.problems import ALL_PROBLEMS

OUT = os.path.join(REPO_ROOT, "logs", "posthoc_less_vs_all.txt")
ORDER = ["LESS-A", "ANFIS", "TSK", "RF", "KNN", "MLP", "LOCR", "MAG", "GPR", "XGB", "LGBM"]


def p_str(v):
    if not np.isfinite(v):
        return "   N/A"
    return "<0.0001" if v < 1e-4 else f"{v:7.4f}"


def main():
    df = load_observations("all")
    dropped_problems = sorted(set(df["problem"].unique()) - set(ALL_PROBLEMS))
    df = df[df["problem"].isin(ALL_PROBLEMS)]

    means = (
        df.groupby(["problem", "model"], as_index=False)["mse_scaled"]
        .mean()
        .pivot(index="problem", columns="model", values="mse_scaled")
    )
    n_problems = means.shape[0]
    lines = []
    w = lines.append

    w(f"veri seti: {n_problems}  |  kontrol: {BASE}  |  birim: fold ortalamasi, problem bazinda olcekli MSE")
    if dropped_problems:
        w(f"suite disinda birakilanlar: {', '.join(dropped_problems)}")
    cover = means.notna().sum().sort_values()
    w("yontem kapsamasi: " + ", ".join(f"{m} {int(c)}/{n_problems}" for m, c in cover.items()))
    w("")

    # ---- step 1: Friedman, on the complete submatrix ------------------------
    complete = means.dropna(axis=1)
    excluded = [c for c in means.columns if c not in complete.columns]
    ranks = complete.rank(axis=1)
    k, N = complete.shape[1], complete.shape[0]
    chi2, p_omni = friedmanchisquare(*[ranks[c].values for c in complete.columns])
    F = (N - 1) * chi2 / (N * (k - 1) - chi2)
    df1, df2 = k - 1, (k - 1) * (N - 1)
    from scipy.stats import f as fdist

    p_F = 1 - fdist.cdf(F, df1, df2)
    avg = ranks.mean().sort_values()

    w("ADIM 1 -- Friedman omnibus")
    w(f"  siralanan yontemler ({k}): " + ", ".join(avg.index))
    if excluded:
        w(f"  eksik hucre nedeniyle siralamaya girmeyenler: {', '.join(excluded)}")
    w(f"  chi2({k - 1}) = {chi2:.4f}, p = {p_omni:.3g}")
    w(f"  Iman-Davenport F({df1},{df2}) = {F:.4f}, p = {p_F:.3g}")
    w("  ortalama siralar: " + ", ".join(f"{m}={v:.2f}" for m, v in avg.items()))
    w("")

    # ---- step 2a / 2b: many-to-one against the control ----------------------
    others = [m for m in ORDER if m in means.columns]
    rows = []
    for m in others:
        pair = means[[BASE, m]].dropna()
        n = pair.shape[0]
        if n < 3:
            rows.append((m, n, np.nan, np.nan, np.nan))
            continue
        d = pair[m] - pair[BASE]  # positive = LESS-B better
        p_t = ttest_rel(pair[m], pair[BASE]).pvalue
        try:
            p_w = wilcoxon(d).pvalue
        except ValueError:
            p_w = np.nan
        rows.append((m, n, float(d.mean()), float(p_t), float(p_w)))

    made = [i for i, r in enumerate(rows) if np.isfinite(r[3])]
    p_t_holm = np.full(len(rows), np.nan)
    p_w_holm = np.full(len(rows), np.nan)
    p_t_holm[made] = holm([rows[i][3] for i in made])
    p_w_holm[made] = holm([rows[i][4] for i in made])

    w(f"ADIM 2 -- {BASE}'ye karsi ikili karsilastirmalar   (Holm, {len(made)} karsilastirma uzerinden)")
    w("  fark > 0  =>  LESS-B daha iyi")
    w("")
    w(f"  {'yontem':8s} {'N':>3s} {'fark':>8s} | {'t: p':>8s} {'Holm':>8s} | {'W: p':>8s} {'Holm':>8s}")
    w("  " + "-" * 62)
    for i, (m, n, diff, p_t, p_w) in enumerate(rows):
        dstr = f"{diff:8.4f}" if np.isfinite(diff) else "     N/A"
        w(f"  {m:8s} {n:3d} {dstr} | {p_str(p_t)} {p_str(p_t_holm[i])} | "
          f"{p_str(p_w)} {p_str(p_w_holm[i])}")
    w("  " + "-" * 62)
    missing = [r[0] for r in rows if not np.isfinite(r[3])]
    absent = [m for m in ORDER if m not in means.columns]
    if missing or absent:
        w("  karsilastirilamayanlar: " + ", ".join(missing + absent) + " (yeterli hucre yok)")

    text = "\n".join(lines)
    print(text)
    with open(OUT, "w") as f:
        f.write(text + "\n")
    print(f"\nyazildi: {os.path.relpath(OUT, REPO_ROOT)}")


if __name__ == "__main__":
    main()
