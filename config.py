"""Shared settings: data path, random seed, cross-validation sizes, scoring and
the hyperparameter grid of every model.
"""

# PATH
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

# Resolved against this file, so the repository works wherever it is cloned.
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasetsR")

# SEED
RANDOM_STATE = 16252329
N_OUT_CV = 4
N_IN_CV = 4

# The inner cross-validation selects hyperparameters by the same quantity the
# tables report. Left unset, GridSearchCV falls back to the estimator's own
# score(), which for a regressor is R^2 = 1 - MSE / mean((y - ybar)^2). Within
# one validation fold that denominator is the same for every candidate, so the
# two criteria rank them identically; the fold scores are then averaged, and
# there each fold is divided by its own target variance, which is the same as
# weighting the fold MSEs by the inverse of that variance. Measured across the
# nine problems, this changed the selected configuration on eight of them.
SCORING = "neg_mean_squared_error"

# The grid the paper reports: subset structure, kernel width and the local
# learner. What it does not search is capacity. LESS keeps the package
# defaults -- 100 stages, learning_rate 0.1 for LESS-B, local trees capped at
# 31 leaves with min_child_weight 2, and 25 trees in each global aggregator
# (less.py lines 371-424) -- so its capacity is fixed while xgboost's and
# lightgbm's is tuned over depth and rounds. That cuts both ways and the paper
# has to say so: LESS cannot grow with n the way the boosters can, but it also
# carries 25 trees in every one of its 100 stages without spending a grid
# point on them. These are method-specific grids, not an equal tuning budget.
LESSB_GRID = {
    "n_subsets": [10, 20],
    "kernel_coeff": [0.1, 0.01],
    "local_estimator": ["linear", "tree"],
    "random_state": [RANDOM_STATE],
}
# LESS-A searches the same three axes as LESS-B. It averages its local
# predictions where LESS-B boosts them, so it has no learning rate to fix.
LESSA_GRID = {
    "n_subsets": [10, 20],
    "kernel_coeff": [0.1, 0.01],
    "local_estimator": ["linear", "tree"],
    "random_state": [RANDOM_STATE],
}

# The forest is held to the same depth the boosters work within. Left
# unbounded it is the only method in the table without a cap, and on a problem
# that rewards deep trees that is the whole result: one outer fold of sgemm
# gives the unbounded forest MSE 16.4 at an average depth of 24, the same
# forest at depth 8 gives 2486, and XGB allowed the same depth gives 16.8. A
# comparison where one method may grow and the others may not measures the
# grids, not the methods.
# The forest is searched over the same three round counts as the boosting
# baselines. Theory says a forest's expected error converges as trees are added
# rather than rising, so the count could have been fixed high -- but convergence
# of the expectation is not monotonicity of the test error on a finite sample,
# and nothing here measured that, so the count stays in the search where the
# other ensembles keep theirs.
RF_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [6, 8],
    "random_state": [RANDOM_STATE],
}

KNN_GRID = {
    "n_neighbors": [3, 5, 10, 20, 50, 100],
}
# max_iter is a ceiling that does not bind: adam stops on its own
# (n_iter_no_change=10, tol=1e-4) after 22 to 395 iterations across these
# problems, and the sweep raised no ConvergenceWarning at 1000. Raising it
# to 5000 therefore leaves every stored score unchanged; it is insurance
# against a harder fit, not a change of protocol.
MLP_GRID = {"alpha": [0.0001, 0.001, 0.01], "max_iter": [5000]}
LOCR_GRID = {"frac": [0.01, 0.05, 0.10, 0.20]}
MAG_GRID = {"frac": [0.10, 0.20, 0.30], "random_state": [RANDOM_STATE]}

GPR_GRID = {"alpha": [1.0e-1, 1.0e-4, 1.0e-8], "random_state": [RANDOM_STATE]}
# LightGBM grows leafwise, so the leaf count is its capacity dial and max_depth
# only bites once the leaves allow it. The three values are the leaf counts a
# depth-wise tree of the depths XGBoost searches would reach: a full binary tree
# of depth d holds 2^d leaves, so 6, 8 and 10 map to 63, 255 and 1023. LightGBM
# reaches those leaves along deeper, narrower paths than XGBoost does, so this
# matches the capacity rather than the shape.
LGBM_GRID = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.1],
    # Matched to xgboost's grid rather than translated into leaves. lightgbm
    # grows leaf-wise, so num_leaves alone does not bound depth: measured on
    # query, num_leaves=255 with no depth cap produced trees averaging 13.2
    # levels and reaching 18, against xgboost's 8 -- a freedom xgboost does not
    # have on the same grid. Capping depth and leaving num_leaves as a ceiling
    # that does not bind reproduces xgboost's capacity almost exactly: at
    # max_depth 6 and 8 lightgbm builds 63.1 and 237.1 leaves against xgboost's
    # 63.9 and 247.2, at identical depth.
    "max_depth": [6, 8],
    "num_leaves": [255],
    "random_state": [RANDOM_STATE],
    "verbose": [-1],
}

# Eight candidates, the same count LESS-B, LESS-A and LGBM get, and the same
# depth axis LGBM carries. xgboost grows depthwise and leaves max_leaves
# unset, so max_depth alone bounds the tree: measured on query it fills the
# bound, 63.9 leaves at depth 6 and 247.2 at depth 8 out of 64 and 256.
XGB_GRID = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.1],
    "max_depth": [6, 8],
    "random_state": [RANDOM_STATE],
    "verbosity": [0],
}

# The two tree consequents are seeded, and so is the fuzzy clustering that
# feeds them: without it the same fold and the same grid point did not give the
# same answer twice, which makes a recorded score unreproducible.
TSK_GRID = {
    "consequent": [Ridge(),
                   DecisionTreeRegressor(random_state=RANDOM_STATE),
                   RandomForestRegressor(random_state=RANDOM_STATE)],
    "n_cluster": [5, 10, 20],
    "fuzzy_index": [2.0, "auto"],
    "random_state": [RANDOM_STATE],
}

ANFIS_GRID = {"hybrid": [True, False], "epoch": [25, 50]}
