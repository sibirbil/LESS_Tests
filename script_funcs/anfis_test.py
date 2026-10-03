"""Scores ANFIS (scikit-anfis) on the problems it can handle.

Run it on its own: scikit-anfis writes a checkpoint with a fixed file name to
the working directory on every fit, so two ANFIS jobs in the same directory
would read each other's model.
"""
import argparse
import json
import logging
import os
import warnings
from itertools import product

import numpy as np
from skanfis import scikit_anfis
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from config import (
    ANFIS_GRID,
    DATA_PATH,
    N_IN_CV,
    N_OUT_CV,
    RANDOM_STATE,
)
from src.problems import ALL_PROBLEMS, ROW_IDENTITY, row_identity_groups
from src.protocol import check as protocol_check

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

LOG_DIR = "logs"

# ANFIS wrote into logs/results unconditionally, so a corrected run overwrote
# the published one and there was no way to keep the two apart. It now takes the
# same --results-dir as every other runner and declares the same protocol, which
# also stops it appending to a directory produced under different settings.
ap = argparse.ArgumentParser()
ap.add_argument("--problems", nargs="+", default=["ABALONE", "AIRFOIL", "CCPP", "HOUSING"],
                choices=list(ALL_PROBLEMS),
                help="default: the four the paper reports a number for. Table 1 "
                     "prints N/A for cadata, cpusmallscale and energy without "
                     "saying why; the rule count explains energy (2^26 rules) "
                     "but not cadata (2^8) or cpusmallscale (2^12), both of "
                     "which need fewer rules than housing, which did run")
ap.add_argument("--log-file", default="anfis_training.log",
                help="log file under logs/; a rerun should not append to the "
                     "log of the run it replaces")
ap.add_argument("--results-dir", default="results",
                help="subdirectory of logs/ to write to")
args = ap.parse_args()

RESULT_DIR = os.path.join(LOG_DIR, args.results_dir)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
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

def train_anfis_with_cv(X_train, y_train, X_test, y_test, grid, inner_fold,
                        g_train=None):
    """ANFIS için manuel grid search cross-validation"""
    param_names = list(grid.keys())
    param_values = list(grid.values())
    param_combinations = list(product(*param_values))
    
    best_score = float('inf')
    best_params = None
    
    for param_combo in param_combinations:
        params = dict(zip(param_names, param_combo, strict=False))
        logger.info(f"Testing ANFIS params: {params}")
        
        cv_scores = []
        
        for inner_train_idx, inner_val_idx in inner_fold.split(X_train, groups=g_train):
            X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
            y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]
            
            # Scaling
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_inner_train_scaled = scaler_X.fit_transform(X_inner_train)
            X_inner_val_scaled = scaler_X.transform(X_inner_val)
            y_inner_train_scaled = scaler_y.fit_transform(y_inner_train.reshape(-1, 1)).ravel()
            
            # ANFIS model training
            anfis_model = scikit_anfis(data=X_inner_train_scaled, label="r", **params)
            anfis_model.fit(X_inner_train_scaled, y_inner_train_scaled)
            
            # Prediction and evaluation
            y_pred_scaled = anfis_model.predict(X_inner_val_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            mse = mean_squared_error(y_inner_val, y_pred)
            cv_scores.append(mse)
        
        avg_cv_score = np.mean(cv_scores)
        logger.info(f"ANFIS params {params} - Avg CV MSE: {avg_cv_score:.4f}")
        
        if avg_cv_score < best_score:
            best_score = avg_cv_score
            best_params = params
    
    # Train final model with best parameters
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    
    best_estimator = scikit_anfis(data=X_train_scaled, label="r", **best_params)
    best_estimator.fit(X_train_scaled, y_train_scaled)
    
    y_pred_scaled = best_estimator.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    mse = mean_squared_error(y_pred, y_test)
    
    return mse, best_params

# main.py groups ccpp by the repeated record -- the file ships 41 pairs of
# identical (X, y) rows -- so the same grouping applies here, or the anfis row
# would rest on a split main.py refuses.
GROUP_SPECS = {"CCPP": ROW_IDENTITY}

if __name__ == "__main__":
    for problem_name in args.problems:
        problem, group_spec = ALL_PROBLEMS[problem_name]
        logger.info(f"Starting ANFIS experiment for problem: {problem_name}")

        result_path = os.path.join(RESULT_DIR, f"{problem_name}.json")

        # Load existing results if available
        if os.path.exists(result_path):
            with open(result_path) as f:
                previous_results = json.load(f)
        else:
            previous_results = {"ANFIS": {}}

        X, y = problem(DATA_PATH)
        # 2^p rules, one per combination of the two membership functions per
        # feature, so the width of the problem decides feasibility long before
        # its length does
        logger.info(f"{problem_name}: n={len(X)} p={X.shape[1]} "
                    f"rules=2^{X.shape[1]}={2 ** min(X.shape[1], 62)}")
        if group_spec is None:
            groups = None
        elif group_spec is ROW_IDENTITY:
            groups = row_identity_groups(X)
        else:
            groups = group_spec(DATA_PATH)

        outer_fold = (
        GroupKFold(n_splits=N_OUT_CV) if groups is not None
        else KFold(n_splits=N_OUT_CV, shuffle=True, random_state=RANDOM_STATE)
    )
        # The inner seed was RANDOM_STATE + 1 with nothing in the code or the
        # paper to justify the offset. The inner loop splits a subset of the
        # outer training rows, so it has no reason to differ from the outer
        # seed, and every other runner uses RANDOM_STATE.
        inner_fold = (
        GroupKFold(n_splits=N_IN_CV) if groups is not None
        else KFold(n_splits=N_IN_CV, shuffle=True, random_state=RANDOM_STATE)
    )

        for fold_num, (train_index, test_index) in enumerate(
            outer_fold.split(X, y, groups=groups), start=1
        ):
            logger.info(f"ANFIS Fold {fold_num} for {problem_name}")

            # Skip if already computed
            if str(fold_num) in previous_results.get("ANFIS", {}):
                logger.info(f"Skipping ANFIS fold {fold_num} (already computed).")
                continue

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            logger.info(f"Training ANFIS on fold {fold_num}...")
            
            mse, best_params = train_anfis_with_cv(
                X_train, y_train, X_test, y_test, ANFIS_GRID, inner_fold,
            groups[train_index] if groups is not None else None
            )

            logger.info(f"ANFIS MSE on fold {fold_num}: {mse:.4f}")

            # Save results
            if "ANFIS" not in previous_results:
                previous_results["ANFIS"] = {}
            previous_results["ANFIS"][str(fold_num)] = mse
            # train_anfis_with_cv already picks the grid point; without this the
            # score is all that survives and which hybrid/epoch won cannot be
            # recovered from the file.
            previous_results.setdefault("_best_params", {}).setdefault(
                "ANFIS", {})[str(fold_num)] = best_params

            # Write to JSON file
            # A run this long gets killed sooner or later; json.dump straight onto the
            # real path would leave a half-written file that the next attempt cannot
            # read. Write beside it and rename, so the file on disk is always one
            # complete version or the other.
            _tmp = f"{result_path}.tmp"
            with open(_tmp, "w") as f:
                json.dump(previous_results, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(_tmp, result_path)

        # Calculate and log average performance
        if "ANFIS" in previous_results:
            mse_scores = list(previous_results["ANFIS"].values())
            avg_mse = np.mean(mse_scores)
            std_mse = np.std(mse_scores)
            logger.info(f"ANFIS Average MSE for {problem_name}: {avg_mse:.4f} ± {std_mse:.4f}")