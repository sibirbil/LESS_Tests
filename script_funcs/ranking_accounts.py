"""Average ranks of the methods, computed three ways because some methods
did not run on every problem:

  A   only the methods that ran everywhere (Friedman test),
  B   every method, a missing cell gets the worst rank,
  SM  Skillings-Mack test, which handles missing cells directly.

    PYTHONPATH=. python script_funcs/ranking_accounts.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2dist
from scipy.stats import f as fdist
from scipy.stats import friedmanchisquare, norm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "script_funcs"))

from anova_many_to_one import BASE, holm, load_observations

OUT = os.path.join(REPO_ROOT, "logs", "ranking_accounts.txt")


def friedman_from_ranks(ranks: pd.DataFrame):
    """chi2, Iman-Davenport F and their p-values, from a block x method rank matrix."""
    N, k = ranks.shape
    stat, p_chi = friedmanchisquare(*[ranks[c].values for c in ranks.columns])
    F = (N - 1) * stat / (N * (k - 1) - stat)
    df1, df2 = k - 1, (k - 1) * (N - 1)
    return stat, p_chi, F, df1, df2, 1 - fdist.cdf(F, df1, df2)


def many_to_one(ranks: pd.DataFrame, base: str):
    """Demsar's rank comparison against a control: z, Bonferroni-Dunn and Holm."""
    N, k = ranks.shape
    avg = ranks.mean()
    se = np.sqrt(k * (k + 1) / (6.0 * N))
    others = [c for c in ranks.columns if c != base]
    z = np.array([(avg[c] - avg[base]) / se for c in others])
    p_raw = 2 * (1 - norm.cdf(np.abs(z)))
    return others, avg, z, np.minimum(p_raw * len(others), 1.0), holm(p_raw)


def worst_rank_matrix(means: pd.DataFrame) -> pd.DataFrame:
    """Observed cells ranked 1..m within the block; missing cells share the tail ranks."""
    k = means.shape[1]
    out = pd.DataFrame(index=means.index, columns=means.columns, dtype=float)
    for prob, row in means.iterrows():
        obs = row.dropna()
        out.loc[prob, obs.index] = obs.rank()
        miss = [c for c in means.columns if c not in obs.index]
        if miss:
            # the ranks that are left over, averaged so the block still sums to k(k+1)/2
            out.loc[prob, miss] = np.mean(np.arange(len(obs) + 1, k + 1))
    return out


def mid_rank_matrix(means: pd.DataFrame) -> pd.DataFrame:
    """Missing cells get the block's middle rank; the observed ones are centred on it.

    The neutral counterpart of worst-rank scoring: a cell with no score is placed
    where it carries no evidence either way, (k+1)/2. The observed ranks are then
    shifted by (k-m)/2 so the block still averages (k+1)/2 and sums to k(k+1)/2,
    which is what Friedman's statistic assumes. The price is power: a block with
    many missing cells has its observed ranks squeezed toward the middle, so the
    spread that the test reads as evidence shrinks.
    """
    k = means.shape[1]
    mid = (k + 1) / 2.0
    out = pd.DataFrame(index=means.index, columns=means.columns, dtype=float)
    for prob, row in means.iterrows():
        obs = row.dropna()
        m = len(obs)
        out.loc[prob, obs.index] = obs.rank() + (k - m) / 2.0
        miss = [c for c in means.columns if c not in obs.index]
        if miss:
            out.loc[prob, miss] = mid
    return out


def skillings_mack(means: pd.DataFrame):
    """Skillings-Mack (1981) statistic for an incomplete block design."""
    cols = list(means.columns)
    k = len(cols)
    A = np.zeros(k)
    S = np.zeros((k, k))
    for _, row in means.iterrows():
        obs = row.dropna()
        ki = len(obs)
        if ki < 2:
            continue
        r = obs.rank()
        c = np.sqrt(12.0 / (ki + 1))
        for name, val in r.items():
            A[cols.index(name)] += c * (val - (ki + 1) / 2.0)
        idx = [cols.index(n) for n in obs.index]
        for a in idx:
            S[a, a] += ki - 1
            for b in idx:
                if a != b:
                    S[a, b] -= 1
    T = float(A @ np.linalg.pinv(S) @ A)
    df = k - 1
    return T, df, 1 - chi2dist.cdf(T, df)


def block(title, ranks, note, lines):
    w = lines.append
    stat, p_chi, F, df1, df2, p_F = friedman_from_ranks(ranks)
    others, avg, z, p_bonf, p_holm = many_to_one(ranks, BASE)
    order = avg.sort_values()
    w("")
    w("=" * 74)
    w(title)
    w("=" * 74)
    w(note)
    w(f"  {ranks.shape[0]} veri x {ranks.shape[1]} yontem")
    w(f"  Friedman chi2({ranks.shape[1] - 1}) = {stat:.4f}, p = {p_chi:.3g}")
    w(f"  Iman-Davenport F({df1},{df2}) = {F:.4f}, p = {p_F:.3g}")
    w("  ortalama siralar: " + ", ".join(f"{m}={v:.2f}" for m, v in order.items()))
    w("")
    w(f"  {BASE}'ye karsi (rank tabanli, Demsar many-to-one)")
    w(f"  {'yontem':8s}{'sira':>7s}{'fark':>7s}{'z':>7s}{'p (B-Dunn)':>12s}{'p (Holm)':>10s}")
    w("  " + "-" * 51)
    for i, m in enumerate(others):
        w(f"  {m:8s}{avg[m]:7.2f}{avg[m] - avg[BASE]:7.2f}{z[i]:7.2f}"
          f"{p_bonf[i]:12.4f}{p_holm[i]:10.4f}")
    w("  " + "-" * 51)


def main():
    df = load_observations("all")
    means = (
        df.groupby(["problem", "model"], as_index=False)["mse_scaled"]
        .mean()
        .pivot(index="problem", columns="model", values="mse_scaled")
    )
    lines = []
    w = lines.append
    n = means.shape[0]
    cover = means.notna().sum().sort_values()
    w(f"veri seti: {n}   kontrol: {BASE}   birim: fold ortalamasi, problem bazinda olcekli MSE")
    w("kapsama: " + ", ".join(f"{m} {int(c)}/{n}" for m, c in cover.items()))

    complete = means.dropna(axis=1)
    dropped = [c for c in means.columns if c not in complete.columns]
    block("HESAP A -- yalnizca tam kosan yontemler",
          complete.rank(axis=1),
          f"  disarida birakilanlar: {', '.join(dropped)}", lines)

    block("HESAP B -- kosulamayan hucre en kotu sira",
          worst_rank_matrix(means),
          "  eksik hucreler bloklarinin en kotu siralarini paylasiyor", lines)

    block("HESAP C -- kosulamayan hucre ortalama sira",
          mid_rank_matrix(means),
          "  eksik hucreler bloklarinin orta sirasini aliyor, gozlenenler onun etrafina kaydiriliyor",
          lines)

    T, dfree, p_sm = skillings_mack(means)
    w("")
    w("=" * 74)
    w("SKILLINGS-MACK -- eksik bloklu tasarim icin omnibus (atama yok)")
    w("=" * 74)
    w(f"  T = {T:.4f}, df = {dfree}, p = {p_sm:.3g}")

    text = "\n".join(lines)
    print(text)
    with open(OUT, "w") as f:
        f.write(text + "\n")
    print(f"\nyazildi: {os.path.relpath(OUT, REPO_ROOT)}")


if __name__ == "__main__":
    main()
