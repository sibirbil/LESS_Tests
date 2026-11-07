"""Shared settings: data path, random seed, cross-validation sizes, scoring and
the hyperparameter grid of every model.
"""

# PATH
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
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

# Every method is tuned over a full-factorial grid of exactly N_GRID
# configurations, the same grid on every problem and fold, and every grid
# contains the estimator's default configuration. main.py refuses a grid of any
# other size. Equal size means an equal number of evaluated configurations, not
# equal compute and not equal coverage of each method's space.
N_GRID = 48

# LESS-B and LESS-A: locality (number of subsets m), local learner, kernel
# coefficient and number of stages b. kernel_coeff=None is lambda = m^-2, which
# for these m never coincides with 0.1 or 1, so all 48 points are distinct.
# LESS-B's learning rate (rho) is fixed at the package default.
LESSB_GRID = {
    "n_subsets": [5, 10, 20, 50],
    "local_estimator": ["linear", "tree"],
    "kernel_coeff": [None, 0.1, 1.0],
    "n_estimators": [100, 500],
    "random_state": [RANDOM_STATE],
    "learning_rate": [0.1],
}
LESSA_GRID = {
    "n_subsets": [5, 10, 20, 50],
    "local_estimator": ["linear", "tree"],
    "kernel_coeff": [None, 0.1, 1.0],
    "n_estimators": [100, 500],
    "random_state": [RANDOM_STATE],
}

# Leaf-wise growth with no depth cap: num_leaves is the capacity axis.
LGBM_GRID = {
    "n_estimators": [100, 500, 2000],
    "learning_rate": [0.03, 0.1],
    "num_leaves": [31, 127, 255, 1023],
    "min_child_samples": [20, 100],
    "max_depth": [-1],
    "random_state": [RANDOM_STATE],
    "verbose": [-1],
}

# Depth-wise growth: rounds, learning rate and depth.
XGB_GRID = {
    "n_estimators": [100, 500, 2000],
    "learning_rate": [0.01, 0.03, 0.1, 0.3],
    "max_depth": [3, 6, 9, 12],
    "random_state": [RANDOM_STATE],
    "verbosity": [0],
}

# Unbounded depth is in the grid; max_features=1.0 is the sklearn default.
RF_GRID = {
    "n_estimators": [100, 500],
    "max_depth": [None, 12],
    "max_features": [0.33, 0.5, 1.0],
    "min_samples_leaf": [1, 2, 5, 10],
    "random_state": [RANDOM_STATE],
}

KNN_GRID = {
    "n_neighbors": [1, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
}

# max_iter=200 is the sklearn default; 5000 lets adam run to its own stopping
# rule on the larger problems.
MLP_GRID = {
    "hidden_layer_sizes": [(100,), (200,), (100, 100)],
    "alpha": [1e-6, 1e-4, 1e-2, 1e-1],
    "learning_rate_init": [1e-3, 1e-2],
    "max_iter": [200, 5000],
}

LOCR_GRID = {"frac": [j / 100 for j in range(1, 49)]}

# Random partition into q = 2..49 groups (frac = 1/q); 0.2 is the default.
MAG_GRID = {"frac": [1 / q for q in range(2, 50)], "random_state": [RANDOM_STATE]}

# Kernel hyperparameters are fixed so that each grid point is the model it
# names; RBF(1.0) with alpha=1e-10 is the sklearn default.
_GPR_LENGTHS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
GPR_GRID = {
    "alpha": [1e-10, 1e-6, 1e-3, 1e-1],
    "kernel": [RBF(length_scale=ls, length_scale_bounds="fixed") for ls in _GPR_LENGTHS]
    + [Matern(length_scale=ls, length_scale_bounds="fixed", nu=2.5)
       for ls in _GPR_LENGTHS],
    "random_state": [RANDOM_STATE],
}

# The tree consequents are seeded, and so is the fuzzy clustering that feeds
# them, so a fold and a grid point always give the same answer.
TSK_GRID = {
    "consequent": [Ridge(),
                   DecisionTreeRegressor(random_state=RANDOM_STATE),
                   RandomForestRegressor(random_state=RANDOM_STATE)],
    "n_cluster": [5, 10, 20, 40],
    "fuzzy_index": [1.6, 2.5, 4.0, "auto"],
    "random_state": [RANDOM_STATE],
}

ANFIS_GRID = {"hybrid": [True, False], "epoch": [25, 50]}
