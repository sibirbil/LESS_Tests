"""Main benchmark: scores LESS and the competing models on every problem.

Each problem is evaluated with nested cross-validation (outer folds for the test
score, inner folds for GridSearchCV). Results are written per problem to
logs/<results-dir>/<PROBLEM>.json, and finished cells are skipped on a rerun.

    PYTHONPATH=. python main.py [--problems ...] [--only ...] [--no-budget]
"""

import argparse
import json
import logging
import os
import time
import warnings
from collections import defaultdict

import numpy as np
from less import LESSARegressor, LESSBRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import (
    DATA_PATH,
    GPR_GRID,
    KNN_GRID,
    LESSA_GRID,
    LESSB_GRID,
    LGBM_GRID,
    LOCR_GRID,
    MAG_GRID,
    MLP_GRID,
    N_IN_CV,
    N_OUT_CV,
    RANDOM_STATE,
    RF_GRID,
    SCORING,
    TSK_GRID,
    XGB_GRID,
)
from src.budget import (
    budget_check,
    default_memory_cap,
    load_probe_cache,
)
from src.environments import environment_for
from src.models.locreg import LocalRegression
from src.models.magging import Magging

# from src.models.anfis import SklearnAnfisRegressor
from src.models.tsk import TSKRegressorPipeline
from src.problems import ALL_PROBLEMS, ROW_IDENTITY, SMALL_PROBLEMS, row_identity_groups
from src.protocol import check as protocol_check

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

LOG_DIR = "logs"

# The inner CV selects on config.SCORING (negative MSE), the same quantity the
# tables report. The published results were produced before this was set, with
# GridSearchCV falling back to the estimator's own R^2, so --results-dir lets a
# corrected run be written next to them instead of over them.

ap = argparse.ArgumentParser()
ap.add_argument("--problems", nargs="+", default=list(SMALL_PROBLEMS),
                choices=list(ALL_PROBLEMS),
                help="default: the seven problems the paper reports")
ap.add_argument("--results-dir", default="results",
                help="subdirectory of logs/ to write to")
ap.add_argument("--skip", nargs="*", default=[],
                help="estimator names to leave out of this run")
ap.add_argument("--only", nargs="*", default=None,
                help="run just these estimators, leaving the other cells alone")
ap.add_argument("--log-file", default="training.log")
# The larger problems are out of reach for some of the methods. Rather than a
# hand-written list of which, the cost of each cell is measured on small
# subsamples and projected; a projection over these limits refuses the cell and
# records the number behind the refusal. See src/budget.py.
ap.add_argument("--no-budget", action="store_true",
                help="skip the cost projection and run every cell. The probe "
                     "exists to keep a sweep from starting a fit that never "
                     "returns; on a rerun of methods already known to finish on "
                     "these problems it is pure overhead, and a single-threaded "
                     "probe of an expensive candidate can cost more than the "
                     "search it guards")
ap.add_argument("--grid-n-jobs", type=int, default=5,
                help="GridSearchCV workers; also what the cost projection "
                     "assumes will share the work")
ap.add_argument("--budget-hours", type=float, default=20.0,
                help="refuse a cell whose projected nested search exceeds this")
ap.add_argument("--memory-cap-gb", type=float, default=default_memory_cap(),
                help="the memory one cell may use; defaults to 60%% of this "
                     "machine's RAM")
args = ap.parse_args()

RESULT_DIR = os.path.join(LOG_DIR, args.results_dir)
args.result_dir = RESULT_DIR
os.makedirs(LOG_DIR, exist_ok=True)
# Stops instead of appending if this directory was produced under other settings.
# One policy for every problem and every runner, so the manifest is the same
# string everywhere and a directory written under the old unshuffled inner loop
# is refused instead of being appended to.
protocol_check(RESULT_DIR, {"inner_cv": "same policy as the outer split"})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, args.log_file)),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def save_results(path, payload):
    """Write the results file atomically.

    A long unattended run gets killed sooner or later -- out of memory, a
    laptop asleep, a stray signal -- and json.dump straight onto the real path
    leaves a half-written file that the next attempt cannot read, losing every
    fold already computed. Writing beside it and renaming makes the swap
    indivisible: the file on disk is always one complete version or the other.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

if __name__ == "__main__":
    logger.info(f"scoring={SCORING} -> {RESULT_DIR}"
                + (f", skipping {args.skip}" if args.skip else "")
                + (f", only {args.only}" if args.only else ""))
    # One probe per (estimator, rows, features), reused across folds and runs.
    probes = load_probe_cache(RESULT_DIR)

    for problem_name in args.problems:
        problem, group_spec = ALL_PROBLEMS[problem_name]
        logger.info(f"Starting problem: {problem_name}")

        result_path = os.path.join(RESULT_DIR, f"{problem_name}.json")

        # Var olan sonuçları yükle
        if os.path.exists(result_path):
            with open(result_path) as f:
                previous_results = json.load(f)
        else:
            previous_results = {}

        X, y = problem(DATA_PATH)
        # A grouping keeps every row of one patient, one road segment or one
        # repeated record on a single side of the split. Without it a plain
        # KFold scores memorisation: on ctslices it put 94 identical (X, y)
        # records on both sides and lifted R^2 from 0.933 to 0.993.
        if group_spec is None:
            groups = None
        elif group_spec is ROW_IDENTITY:
            groups = row_identity_groups(X)
        else:
            groups = group_spec(DATA_PATH)

        cachedir = os.path.join(DATA_PATH, "cache")

        scores = defaultdict(list)

        # Every estimator runs single-threaded: the grid above them is what gets
        # the cores, its candidates are independent, and nesting a library's own
        # threads inside ten grid workers only makes them fight for the machine.
        # Measured on this box, the nested arrangement used 235% of ten cores
        # and the flat one 846%.
        grid_models = [
            ("LESS-B", LESSB_GRID, LESSBRegressor(local_n_jobs=1)),
            ("LESS-A", LESSA_GRID, LESSARegressor(local_n_jobs=1)),
            ("TSK", TSK_GRID, TSKRegressorPipeline(consequent=RandomForestRegressor())),
            ("RF", RF_GRID, RandomForestRegressor(n_jobs=1)),
            ("KNN", KNN_GRID, KNeighborsRegressor(n_jobs=1)),
            ("MLP", MLP_GRID, MLPRegressor(random_state=RANDOM_STATE)),
            ("LOCR", LOCR_GRID, LocalRegression()),
            ("MAG", MAG_GRID, Magging()),
            ("GPR", GPR_GRID, GaussianProcessRegressor()),
            ("XGB", XGB_GRID, XGBRegressor(n_jobs=1)),
            ("LGBM", LGBM_GRID, LGBMRegressor(n_jobs=1)),
        ]

        # GroupKFold is left unshuffled: sklearn assigns groups to folds
        # deterministically and takes no random_state.
        outer_fold = (
            GroupKFold(n_splits=N_OUT_CV) if groups is not None
            else KFold(n_splits=N_OUT_CV, shuffle=True, random_state=RANDOM_STATE)
        )

        grid_models = [g for g in grid_models if g[0] not in args.skip]
        if args.only:
            grid_models = [g for g in grid_models if g[0] in args.only]

        # Magging is the one method here that is defined over environments rather
        # than over a sample: where the data carries them, they are what it gets.
        # Everywhere else it keeps the random disjoint partition, and the reason a
        # label was refused is recorded beside the score.
        env_codes, env_note = environment_for(problem_name, DATA_PATH, X.shape[1])
        if env_codes is not None:
            logger.info(f"{problem_name}: MAG will use the data's own environment "
                        f"({len(np.unique(env_codes))} groups)")
            previous_results.setdefault("_mag_environment", {})["groups"] = int(
                len(np.unique(env_codes)))
        elif env_note:
            previous_results.setdefault("_mag_environment", {})["refused"] = env_note

        for fold_num, (train_index, test_index) in enumerate(
            outer_fold.split(X, y, groups=groups), start=1
        ):
            logger.info(f"Fold {fold_num} for {problem_name}")

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            g_train = groups[train_index] if groups is not None else None

            for estimator_name, grid, estimator in grid_models:
                # Eğer bu fold ve estimator daha önce çalıştıysa atla
                if str(fold_num) in previous_results.get(estimator_name, {}):
                    logger.info(
                        f"Skipping {estimator_name} fold {fold_num} (already computed)."
                    )
                    continue

                skipped = previous_results.setdefault("_skipped", {})
                if str(fold_num) in skipped.get(estimator_name, {}):
                    logger.info(f"Skipping {estimator_name} fold {fold_num} "
                                f"(refused earlier, see _skipped).")
                    continue

                if args.no_budget:
                    reason, projection = None, {"skipped_budget": True}
                else:
                    reason, projection = budget_check(
                        estimator_name, lambda e=estimator: clone(e), grid,
                        X_train, y_train, len(X_test), args, probes)
                # A verdict belongs to a method on a dataset, not to one fold of
                # it. The projection is measured per fold and noisy enough that
                # road3d refused LESS-A on fold 4 after running it three times,
                # leaving a row whose mean covers a different number of folds
                # from its neighbours. Evidence that the cell runs outranks a
                # projection that says it cannot.
                if reason is not None and previous_results.get(estimator_name):
                    logger.info(
                        f"{estimator_name} fold {fold_num}: projection says "
                        f"'{reason}', but folds "
                        f"{sorted(previous_results[estimator_name])} already ran "
                        f"on this problem -- running it"
                    )
                    reason = None
                if reason is not None:
                    logger.info(f"{estimator_name} fold {fold_num}: skipped "
                                f"-- {reason}")
                    # Kept out of the score dict so every reader still sees
                    # plain per-fold numbers there, and next to it so the gap
                    # in the table has a reason attached.
                    skipped.setdefault(estimator_name, {})[str(fold_num)] = {
                        "reason": reason, "projection": projection,
                    }
                    save_results(result_path, previous_results)
                    continue

                logger.info(f"Training {estimator_name} on fold {fold_num}...")

                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("regressor", estimator),
                    ],
                )

                model = TransformedTargetRegressor(
                    regressor=pipe,
                    transformer=StandardScaler(),
                )

                # With an environment in hand, frac has nothing left to choose:
                # the partition comes from the labels, so the three grid points
                # would fit the same model three times and record a meaningless
                # winner.
                cell_grid = grid
                if estimator_name == "MAG" and env_codes is not None:
                    cell_grid = {k: v for k, v in grid.items() if k != "frac"}

                gcv = GridSearchCV(
                    estimator=model,
                    param_grid={
                        f"regressor__regressor__{key}": val
                        for key, val in cell_grid.items()
                    },
                    # The budget may have throttled this cell to fewer workers
                    # to keep it inside the memory cap; running it serially is a
                    # smaller loss than leaving it out of the table.
                    n_jobs=projection.get("n_jobs", args.grid_n_jobs),
                    # The inner split follows the outer policy: the grouping
                    # has to hold at both levels, and on ungrouped data the
                    # shuffle matters because several of these files are sorted.
                    cv=(GroupKFold(N_IN_CV) if g_train is not None
                        else KFold(n_splits=N_IN_CV, shuffle=True,
                                   random_state=RANDOM_STATE)),
                    scoring=SCORING,
                    verbose=1,
                    refit=True,
                )

                try:
                    start = time.perf_counter()
                    fit_params = {}
                    if estimator_name == "MAG" and env_codes is not None:
                        # sklearn slices a fit param of the right length with the
                        # fold, so the estimator sees its own rows' labels.
                        fit_params["regressor__env"] = env_codes[train_index]
                    gcv.fit(X_train, y_train, groups=g_train, **fit_params)
                    grid_time = time.perf_counter() - start

                    start = time.perf_counter()
                    gcv_pred = gcv.best_estimator_.predict(X_test)
                    predict_time = time.perf_counter() - start
                except Exception as exc:
                    # One estimator failing must not take the sweep with it. A
                    # run of this length will hit a solver that does not
                    # converge or a library that raises on some fold, and losing
                    # the remaining problems to it costs hours. The failure is
                    # written next to the scores so it shows up as a gap with a
                    # cause, and a later attempt retries the cell.
                    logger.exception(
                        f"{estimator_name} fold {fold_num} raised, continuing"
                    )
                    previous_results.setdefault("_failed", {}).setdefault(
                        estimator_name, {})[str(fold_num)] = {
                            "error": f"{type(exc).__name__}: {exc}"[:400],
                        }
                    save_results(result_path, previous_results)
                    continue

                mse = mean_squared_error(gcv_pred, y_test)
                logger.info(f"{estimator_name} MSE on fold {fold_num}: {mse:.4f}")

                # Sonucu kaydet
                if estimator_name not in previous_results:
                    previous_results[estimator_name] = {}
                previous_results[estimator_name][str(fold_num)] = mse
                previous_results.get("_failed", {}).get(
                    estimator_name, {}).pop(str(fold_num), None)

                # The score dict stays a plain number per fold, because every
                # reader in the repository expects that. What the tables were
                # missing sits beside it: which configuration the inner CV chose
                # -- the paper reports grids but never which point of them won --
                # and what the fit cost, alongside the projection the budget rule
                # made, so the rule can be checked against what happened.
                previous_results.setdefault("_metrics", {}).setdefault(
                    estimator_name, {})[str(fold_num)] = {
                        "mse": float(mse),
                        "rmse": float(np.sqrt(mse)),
                        "mae": float(mean_absolute_error(y_test, gcv_pred)),
                        "r2": float(r2_score(y_test, gcv_pred)),
                        "refit_time": float(gcv.refit_time_),
                        "grid_time": float(grid_time),
                        "predict_time": float(predict_time),
                        "best_params": {
                            k.replace("regressor__regressor__", ""): (
                                v if isinstance(v, (int, float, str, bool,
                                                    type(None)))
                                else repr(v))
                            for k, v in gcv.best_params_.items()
                        },
                        "n_train": int(X_train.shape[0]),
                        "n_test": int(X_test.shape[0]),
                        "n_features": int(X_train.shape[1]),
                        "grouped": groups is not None,
                        "projection": projection,
                    }

                # JSON olarak dosyaya yaz
                save_results(result_path, previous_results)
