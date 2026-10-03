"""Compares LESS-B with every other method using a two-way ANOVA on the fold
scores (dataset + method effects), with Dunnett simultaneous confidence
intervals. Bonferroni and unadjusted intervals are printed alongside.

    PYTHONPATH=. python script_funcs/anova_many_to_one.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t as tdist

from src.resultdirs import MAIN

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from config import RANDOM_STATE

SYNTHETIC = ("FRIEDMAN1_", "REGRESSION_")
BASE = "LESS-B"
# One results directory now holds every problem, small and large, scored the same
# way, so the two sources the split used to need have collapsed into one. The
# name comes from src/resultdirs so a rerun written elsewhere does not leave this
# test reading the superseded directory -- which is what it was doing.
SOURCES = {
    "all": (
        MAIN if os.path.isabs(MAIN) else os.path.join(REPO_ROOT, MAIN),
        True,
        ["LESS-A", "TSK", "RF", "KNN", "MLP", "LOCR", "MAG", "GPR", "XGB", "LGBM"],
    ),
}
N_DRAWS = 400_000
SEED = 16252329


def load_observations(source: str) -> pd.DataFrame:
    """One row per (dataset, method, fold): the fold-level MSE the model is fitted on."""
    log_dir, scalar_mse, _ = SOURCES[source]
    rows = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*.json"))):
        problem = os.path.basename(path)[:-5]
        if problem.startswith(SYNTHETIC):
            continue
        with open(path) as f:
            data = json.load(f)
        for model, folds in data.items():
            # The runners keep their own bookkeeping beside the scores under
            # keys that start with an underscore -- per-fold metrics, refused
            # cells, cost projections. They are not methods and must not
            # become a column here.
            if model.startswith("_") or not isinstance(folds, dict):
                continue
            for fold, rec in folds.items():
                mse = rec if scalar_mse else rec["mse"]
                rows.append(
                    {"problem": problem, "model": model, "fold": int(fold), "mse": float(mse)}
                )
    df = pd.DataFrame(rows)
    # A cell with fewer than four folds is a run in progress, not a score. Left in,
    # its mean is taken over whatever folds happened to finish and the method is
    # compared on a different test set from everyone else -- the same trap the
    # figure generators had. Partial cells are dropped and named.
    counts = df.groupby(["problem", "model"])["fold"].transform("count")
    partial = df[counts < 4][["problem", "model"]].drop_duplicates()
    if len(partial):
        names = ", ".join(f"{r.model}/{r.problem}" for r in partial.itertuples())
        print(f"kismi hucre atlandi ({len(partial)}): {names}", file=sys.stderr)
    df = df[counts == 4]
    # QUERY and ROAD3D were dropped from the problem suite; their result files stay
    # on disk, so without this filter every test here would still be run over 20
    # datasets while the tables and figures report 18.
    from src.problems import ALL_PROBLEMS

    df = df[df["problem"].isin(ALL_PROBLEMS)]
    # Per-problem scaling by the problem's mean MSE. The additive model assumes one
    # sigma^2 for every cell, and raw MSE spans 12 (housing) to 1e9 (cadata), so
    # without this the residual variance is set by the largest-scale dataset alone.
    df["mse_scaled"] = df.groupby("problem")["mse"].transform(lambda s: s / s.mean())
    return df


def dunnett_quantile(corr: np.ndarray, df_resid: int, alpha: float, rng) -> float:
    """Two-sided equicoordinate quantile of the max-|t| distribution.

    t_k = Z_k / sqrt(X / df) with Z ~ N(0, corr) shared across comparisons and
    X ~ chi2(df); the same denominator for every k is what makes the comparisons
    dependent, and ignoring it is what makes Bonferroni conservative.
    """
    k = corr.shape[0]
    chol = np.linalg.cholesky(corr + 1e-12 * np.eye(k))
    z = rng.standard_normal((N_DRAWS, k)) @ chol.T
    chi = rng.chisquare(df_resid, size=N_DRAWS) / df_resid
    max_abs_t = np.abs(z / np.sqrt(chi)[:, None]).max(axis=1)
    return float(np.quantile(max_abs_t, 1 - alpha))


def max_abs_t_pvalues(corr, df_resid, tstats, rng) -> np.ndarray:
    """P(max_k |T_k| >= |t_j|) -- the p-value that matches the simultaneous intervals."""
    k = corr.shape[0]
    chol = np.linalg.cholesky(corr + 1e-12 * np.eye(k))
    z = rng.standard_normal((N_DRAWS, k)) @ chol.T
    chi = rng.chisquare(df_resid, size=N_DRAWS) / df_resid
    max_abs_t = np.abs(z / np.sqrt(chi)[:, None]).max(axis=1)
    return np.array([(max_abs_t >= abs(t)).mean() for t in tstats])


def paired_t_current(df: pd.DataFrame, other: str, alpha=0.05):
    """What create_latex_table_2.py does now, for side-by-side comparison."""
    d = (
        df[df["model"].isin([BASE, other])]
        .groupby(["problem", "model"], as_index=False)["mse_scaled"]
        .mean()
        .pivot(index="problem", columns="model", values="mse_scaled")
        .dropna()
    )
    if d.shape[0] < 2 or other not in d.columns:
        return np.nan, np.nan, np.nan
    diff = d[other] - d[BASE]
    n = len(diff)
    se = diff.std(ddof=1) / np.sqrt(n)
    tcrit = tdist.ppf(1 - alpha / 2, n - 1)
    tstat = diff.mean() / se if se > 0 else np.inf
    p = 2 * (1 - tdist.cdf(abs(tstat), n - 1))
    return diff.mean() - tcrit * se, diff.mean() + tcrit * se, p


def holm(pvals):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def demsar_route(df, models):
    """Friedman omnibus, then pairwise exact Wilcoxon against the control with Holm.

    This is a defensible family of pairwise comparisons, but it is NOT Demsar's
    many-to-one post-hoc: that one derives z from the average rank differences
    and is in demsar_many_to_one() below. Demsar uses Wilcoxon signed-rank for
    comparing *two* methods; applying it once per competitor is a different
    procedure, and the published Table 2 did exactly that without any
    multiplicity correction.

    Note also that Wilcoxon here is not scale-free the way the rank analysis is.
    It ranks the magnitudes of the differences across datasets, so scaling the
    MSEs per problem changes its p-values (only the signs are preserved); the
    ranks inside a dataset, and therefore Friedman and demsar_many_to_one(), are
    untouched by it.

    Only methods that ran on every dataset can be ranked, so ANFIS, GPR, LOCR
    and TSK drop out; dropna(axis=1) removes columns, so N stays at 16.
    """
    from scipy.stats import friedmanchisquare, wilcoxon

    means = (
        df.groupby(["problem", "model"], as_index=False)["mse_scaled"]
        .mean()
        .pivot(index="problem", columns="model", values="mse_scaled")
    )
    complete = means.dropna(axis=1)  # methods present on every dataset
    dropped = [c for c in means.columns if c not in complete.columns]
    ranks = complete.rank(axis=1)
    stat, p_omni = friedmanchisquare(*[ranks[c].values for c in complete.columns])
    others = [m for m in models if m in complete.columns]
    praw = []
    for m in others:
        try:
            praw.append(wilcoxon(complete[m] - complete[BASE])[1])
        except ValueError:
            praw.append(np.nan)
    return complete, ranks, dropped, stat, p_omni, others, np.array(praw)


def rank_permutation_pvalues(r, n_perm=200_000, seed=RANDOM_STATE):
    """FWER p-values for the many-to-one rank comparisons, without the normal approximation.

    Demsar's post-hoc uses z = (R_j - R_0) / sqrt(k(k+1)/(6N)), whose null
    distribution is asymptotic in N. With N = 16 datasets that approximation is
    worth checking, so this draws the exact null instead: under the omnibus null
    every within-dataset ranking is equally likely, so each replicate assigns an
    independent random permutation of 1..k per dataset. Taking the maximum |z|
    over the k-1 comparisons in each replicate gives single-step FWER control
    (Westfall-Young), which is the permutation analogue of Bonferroni-Dunn.
    """
    rng = np.random.default_rng(seed)
    obs = np.abs(np.asarray(r["z"], float))
    N, k, se = r["N"], r["k"], r["se"]
    hits = np.zeros(len(obs), dtype=np.int64)
    block = 10_000
    done = 0
    while done < n_perm:
        b = min(block, n_perm - done)
        sums = np.zeros((b, k))
        for _ in range(N):  # one random ranking per dataset
            u = rng.random((b, k))
            sums += np.argsort(np.argsort(u, axis=1), axis=1) + 1
        avg = sums / N
        z = (avg[:, 1:] - avg[:, :1]) / se  # control is column 0 by symmetry
        mx = np.max(np.abs(z), axis=1)
        hits += (mx[:, None] >= obs[None, :]).sum(axis=0)
        done += b
    return (hits + 1) / (n_perm + 1)  # add-one, so never exactly zero


def demsar_many_to_one(df, base=BASE, alpha=0.05):
    """Demsar (2006), section 3.2.2: comparing one method with all the others.

    Step 1  Friedman omnibus on the per-dataset ranks, reported with the
            Iman-Davenport F correction, which Demsar prefers because the chi^2
            form of the statistic is undesirably conservative.
    Step 2  If the null is rejected, a post-hoc test against the control. Demsar
            gives Bonferroni-Dunn, whose critical difference is
            CD = q_alpha * sqrt(k(k+1)/(6N)) with q_alpha the two-tailed normal
            quantile at alpha/(k-1).
    Step 3  Holm's step-down procedure, which Demsar notes is more powerful than
            Bonferroni-Dunn and rejects at least as much.

    A method has to be present on every dataset to be ranked, so any method with
    missing runs is dropped and reported separately.
    """
    from scipy.stats import chi2, norm
    from scipy.stats import f as fdist

    means = (
        df.groupby(["problem", "model"], as_index=False)["mse_scaled"]
        .mean()
        .pivot(index="problem", columns="model", values="mse_scaled")
    )
    complete = means.dropna(axis=1)
    dropped = [c for c in means.columns if c not in complete.columns]
    ranks = complete.rank(axis=1)
    N, k = ranks.shape
    avg = ranks.mean(axis=0)

    chi_F = 12 * N / (k * (k + 1)) * ((avg - (k + 1) / 2) ** 2).sum()
    p_chi = 1 - chi2.cdf(chi_F, k - 1)
    F_ID = (N - 1) * chi_F / (N * (k - 1) - chi_F)
    p_ID = 1 - fdist.cdf(F_ID, k - 1, (k - 1) * (N - 1))

    se = np.sqrt(k * (k + 1) / (6 * N))
    others = [c for c in complete.columns if c != base]
    z = np.array([(avg[c] - avg[base]) / se for c in others])
    p_raw = 2 * (1 - norm.cdf(np.abs(z)))
    q_alpha = norm.ppf(1 - alpha / (2 * len(others)))  # Bonferroni-Dunn
    cd = q_alpha * se
    p_bonf = np.minimum(p_raw * len(others), 1.0)
    p_holm = holm(p_raw)

    return {
        "N": N,
        "k": k,
        "dropped": dropped,
        "avg": avg,
        "others": others,
        "se": se,
        "chi_F": chi_F,
        "p_chi": p_chi,
        "F_ID": F_ID,
        "p_ID": p_ID,
        "z": z,
        "p_raw": p_raw,
        "q": q_alpha,
        "cd": cd,
        "p_bonf": p_bonf,
        "p_holm": p_holm,
        "base": base,
    }


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else next(iter(SOURCES))
    if source not in SOURCES:
        raise SystemExit(
            f"kaynak {' veya '.join(repr(k) for k in SOURCES)} olmali, verilen: {source}")
    print(f"### kaynak: {source}\n")
    df = load_observations(source)
    models = [m for m in SOURCES[source][2] if m in set(df["model"])]
    n_problems = df["problem"].nunique()
    print(
        f"gozlem {len(df)} satir | {n_problems} veri seti | {len(models) + 1} yontem "
        f"| fold {sorted(df['fold'].unique())}"
    )
    missing = [
        (m, n_problems - df[df["model"] == m]["problem"].nunique())
        for m in models
        if df[df["model"] == m]["problem"].nunique() < n_problems
    ]
    if missing:
        print(
            "eksik veri seti olan yontemler:",
            ", ".join(f"{m} ({k} eksik)" for m, k in missing),
            "-> dengesiz tasarim",
        )

    # --- the model the paper states: all methods at once, fold-level observations
    fit = smf.ols("mse_scaled ~ C(problem) + C(model)", data=df).fit()
    sigma2 = fit.mse_resid
    df_resid = int(fit.df_resid)
    print(
        f"\niki yonlu toplamsal model: sigma^2 = {sigma2:.6f}, "
        f"artik serbestlik derecesi = {df_resid}"
    )

    # contrasts nu_j - nu_1 straight out of the fitted parameters
    names = fit.params.index.tolist()
    contrasts, valid = [], []
    for m in models:
        key = f"C(model)[T.{m}]"
        base_key = f"C(model)[T.{BASE}]"
        if key not in names:
            continue
        c = np.zeros(len(names))
        c[names.index(key)] = 1.0
        if base_key in names:
            c[names.index(base_key)] = -1.0
        contrasts.append(c)
        valid.append(m)
    C = np.array(contrasts)
    cov = C @ fit.cov_params().values @ C.T
    est = C @ fit.params.values
    se = np.sqrt(np.diag(cov))
    corr = cov / np.outer(se, se)
    tstats = est / se

    rng = np.random.default_rng(SEED)
    d_crit = dunnett_quantile(corr, df_resid, 0.05, rng)
    p_sim = max_abs_t_pvalues(corr, df_resid, tstats, rng)
    t_unadj = tdist.ppf(0.975, df_resid)
    t_bonf = tdist.ppf(1 - 0.025 / len(valid), df_resid)
    p_unadj = 2 * (1 - tdist.cdf(np.abs(tstats), df_resid))

    print(
        f"Dunnett kritik degeri (simultane, %95) = {d_crit:.4f}   "
        f"Bonferroni = {t_bonf:.4f}   duzeltmesiz = {t_unadj:.4f}"
    )
    print("\nPozitif fark = LESS-B daha iyi. Aralik 0'i icermiyorsa fark anlamli.\n")
    print(
        f"{'yontem':<8}{'fark':>9}{'simultane %95 CI':>24}{'p (sim)':>10}"
        f"{'p (duzeltmesiz)':>17}   |  {'MEVCUT KOD: CI':>24}{'p':>10}"
    )
    print("-" * 118)
    for m, e, s, _ts, ps, pu in zip(valid, est, se, tstats, p_sim, p_unadj, strict=False):
        lo, hi = e - d_crit * s, e + d_crit * s
        cl, cr, cp = paired_t_current(df, m)
        sig_new = "" if (lo <= 0 <= hi) else "*"
        sig_old = "" if (not np.isfinite(cl) or cl <= 0 <= cr) else "*"
        print(
            f"{m:<8}{e:>9.4f}[{lo:>9.4f},{hi:>8.4f}]{ps:>10.4f}{pu:>17.4f}{sig_new:<2}"
            f" | [{cl:>9.4f},{cr:>8.4f}]{cp:>10.4f}{sig_old}"
        )
    print("-" * 118)
    print("* = 0'i icermeyen aralik (anlamli). Sagdaki blok mevcut create_latex_table_2.py cikti")

    # --- option (a): keep the pairwise blocked t-test, add the multiplicity
    # correction the paper already promises with the word "simultaneous"
    praw = np.array([paired_t_current(df, m)[2] for m in valid])
    padj = holm(praw)
    print(f"\n(a) MEVCUT YAKLASIM + Holm duzeltmesi ({len(valid)} karsilastirma)")
    print(f"{'yontem':<8}{'p (ham)':>10}{'p (Holm)':>10}   sonuc")
    print("-" * 44)
    for m, pr, pa in zip(valid, praw, padj, strict=False):
        print(f"{m:<8}{pr:>10.4f}{pa:>10.4f}   {'anlamli' if pa < 0.05 else '-'}")

    # --- option (c): the route the paper cites but does not take
    complete, ranks, dropped, stat, p_omni, others, praw_w = demsar_route(df, models)
    padj_w = holm(praw_w)
    print("\n(c) DEMSAR (2006) YOLU: Friedman omnibus + Holm duzeltmeli Wilcoxon")
    if dropped:
        print(f"    siralama icin her veri setinde bulunmayanlar cikarildi: {', '.join(dropped)}")
    print(
        f"    Friedman: chi2 = {stat:.4f}, p = {p_omni:.6f} "
        f"({'yontemler arasi fark var' if p_omni < 0.05 else 'omnibus reddedilemedi'})"
    )
    print(
        "    ortalama siralar: "
        + ", ".join(
            f"{c}={ranks[c].mean():.2f}"
            for c in sorted(complete.columns, key=lambda c: ranks[c].mean())
        )
    )
    print(f"\n{'yontem':<8}{'p (ham)':>10}{'p (Holm)':>10}   sonuc")
    print("-" * 44)
    for m, pr, pa in zip(others, praw_w, padj_w, strict=False):
        print(f"{m:<8}{pr:>10.4f}{pa:>10.4f}   {'anlamli' if pa < 0.05 else '-'}")

    # --- the post-hoc Demsar prescribes, plus a permutation check on its
    # asymptotic z, which is what small N makes questionable
    r = demsar_many_to_one(df)
    p_perm = rank_permutation_pvalues(r)
    print(
        f"\n    Iman-Davenport F({r['k'] - 1},{(r['k'] - 1) * (r['N'] - 1)}) = "
        f"{r['F_ID']:.4f}, p = {r['p_ID']:.3e}   Bonferroni-Dunn CD = {r['cd']:.4f}"
    )
    print(
        f"\n{'yontem':<8}{'sira':>7}{'z':>8}{'p (B-Dunn)':>12}{'p (Holm)':>10}"
        f"{'p (permutasyon)':>17}"
    )
    print("-" * 62)
    order = np.argsort(r["z"])
    for i in order:
        print(
            f"{r['others'][i]:<8}{float(r['avg'][r['others'][i]]):>7.2f}{r['z'][i]:>8.2f}"
            f"{r['p_bonf'][i]:>12.4f}{r['p_holm'][i]:>10.4f}{p_perm[i]:>17.5f}"
        )
    print("-" * 62)
    print(f"permutasyon: {200_000} tekrar, tek adimli max-|z|, tohum {RANDOM_STATE}")


if __name__ == "__main__":
    main()
