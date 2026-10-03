"""Scores Magging with random groups and with the data's own environments,
on the problems that have a usable environment label.

    PYTHONPATH=. python script_funcs/mag_environments.py [--problems ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_PATH, N_OUT_CV, RANDOM_STATE
from src.models.magging import Magging
from src.problems import ALL_PROBLEMS, ROW_IDENTITY, row_identity_groups

LOG_DIR = "logs"
RESULT_DIR = os.path.join(LOG_DIR, "mag_env")

# problem -> the file holding one environment label per row
# ctslices is listed so the refusal is recorded rather than the problem being
# quietly absent: its 97 scans are environments by every other measure, but with
# 384 features they are far too small to fit a group model in.
ENVIRONMENTS = {
    "LASROSAS": "lasrosas_env.csv",       # landscape position, 4
    "EDWARDS": "edwards_env.csv",         # trial environment, 34
    "AIRQUALITY": "airquality_env.csv",   # month, 12
    "SRDB": "srdb_env.csv",               # biome, 6
    "CTSLICES_GROUPED": "ctslices_groups.csv",  # scan, 97 -- refused, see below
}

ARMS = ["rand", "env"]

logger = logging.getLogger(__name__)


def _atomic_write(path, payload):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", default=list(ENVIRONMENTS),
                    choices=list(ENVIRONMENTS))
    ap.add_argument("--log-file", default="mag_env.log")
    args = ap.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(os.path.join(LOG_DIR, args.log_file)),
                  logging.StreamHandler()],
    )

    for name in args.problems:
        loader, group_spec = ALL_PROBLEMS[name]
        X, y = loader(DATA_PATH)
        env = pd.read_csv(os.path.join(DATA_PATH, ENVIRONMENTS[name]),
                          header=None).iloc[:, 0].values
        codes = pd.factorize(env)[0].astype(float)

        sizes = pd.Series(codes).value_counts()
        need = 5 * X.shape[1]
        logger.info(
            f"{name}: n={len(X)} d={X.shape[1]} environments={sizes.size} "
            f"rows/group {sizes.min()}-{sizes.max()} median {int(sizes.median())} "
            f"(rule: >= {need})"
        )
        # The rule was stated but not applied, and ctslices walked straight into
        # what it exists to prevent: 384 features against a median environment of
        # 499 rows leaves every group OLS underdetermined, and the smallest (66
        # rows) made scipy's lstsq fail outright. A refusal with the numbers
        # behind it is the result here -- an environment too small to fit a model
        # in is not an environment magging can aggregate over.
        if sizes.median() < need:
            refusal = {
                "reason": (f"median environment holds {int(sizes.median())} rows "
                           f"against {X.shape[1]} features; a group model needs "
                           f"at least {need} to be identified, and the smallest "
                           f"environment ({int(sizes.min())} rows) fails to fit "
                           f"at all"),
                "n_environments": int(sizes.size),
                "rows_per_group": [int(sizes.min()), int(sizes.median()),
                                   int(sizes.max())],
                "n_features": int(X.shape[1]),
                "required": int(need),
            }
            _atomic_write(os.path.join(RESULT_DIR, f"{name}.json"),
                          {"_refused": refusal})
            logger.info(f"{name}: refused -- {refusal['reason']}")
            continue

        # the fold policy stays the one the tables use: the leakage grouping,
        # not the environment. Splitting on the environment would score a
        # different question from every other row in the table.
        if group_spec is None:
            groups = None
        elif group_spec is ROW_IDENTITY:
            groups = row_identity_groups(X)
        else:
            groups = group_spec(DATA_PATH)

        outer = (GroupKFold(n_splits=N_OUT_CV) if groups is not None
                 else KFold(n_splits=N_OUT_CV, shuffle=True,
                            random_state=RANDOM_STATE))

        result_path = os.path.join(RESULT_DIR, f"{name}.json")
        results = defaultdict(dict)
        if os.path.exists(result_path):
            results.update(json.load(open(result_path)))

        for fold, (tr, te) in enumerate(outer.split(X, y, groups=groups), start=1):
            for arm in ARMS:
                if str(fold) in results.get(arm, {}):
                    continue
                model = TransformedTargetRegressor(
                    regressor=Pipeline([
                        ("scaler", StandardScaler()),
                        ("regressor", Magging(random_state=RANDOM_STATE)),
                    ]),
                    transformer=StandardScaler(),
                )
                # the environment rides in as a fit parameter, which is how it
                # reaches an estimator wrapped in a Pipeline inside a
                # TransformedTargetRegressor without switching metadata routing
                # on for the whole suite
                fit_params = ({"regressor__env": codes[tr]} if arm == "env" else {})
                model.fit(X[tr], y[tr], **fit_params)
                pred = model.predict(X[te])
                mse = mean_squared_error(y[te], pred)
                inner = model.regressor_.named_steps["regressor"]

                # V_g is computed on the scaled design the estimator was fitted
                # on, not on raw units, so it is comparable across arms
                Xs = model.regressor_.named_steps["scaler"].transform(X[te])
                ys = model.transformer_.transform(y[te].reshape(-1, 1)).ravel()
                V = inner.explained_variance(Xs, ys, codes[te])
                per_env = {
                    str(int(c)): float(mean_squared_error(y[te][codes[te] == c],
                                                          pred[codes[te] == c]))
                    for c in np.unique(codes[te]) if (codes[te] == c).sum() > 1
                }

                results.setdefault(arm, {})[str(fold)] = mse
                results.setdefault("_detail", {}).setdefault(arm, {})[str(fold)] = {
                    "mse": float(mse),
                    "worst_env_mse": max(per_env.values()) if per_env else None,
                    "min_explained_variance": min(V.values()) if V else None,
                    "n_groups": int(inner.n_groups_),
                    "solver": inner.solver_,
                    "max_weight": float(np.max(inner.w)),
                    "nonzero_weights": int(np.sum(inner.w > 1e-8)),
                }
                logger.info(
                    f"{name} fold {fold} {arm}: MSE {mse:.4f} "
                    f"({inner.n_groups_} groups, max w {np.max(inner.w):.3f}, "
                    f"min_g V {min(V.values()):.4f})" if V else
                    f"{name} fold {fold} {arm}: MSE {mse:.4f}"
                )
                _atomic_write(result_path, dict(results))

        for arm in ARMS:
            vals = list(results.get(arm, {}).values())
            det = (results.get("_detail", {}).get(arm) or {}).values()
            worst = [d["worst_env_mse"] for d in det if d.get("worst_env_mse")]
            minv = [d["min_explained_variance"] for d in det
                    if d.get("min_explained_variance") is not None]
            if vals:
                logger.info(
                    f"{name} {arm}: mean MSE {np.mean(vals):.4f}"
                    + (f" | worst environment {np.mean(worst):.4f}" if worst else "")
                    + (f" | min_g V {np.mean(minv):.4f}" if minv else ""))


if __name__ == "__main__":
    main()
