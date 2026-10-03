#!/usr/bin/env python3
"""Magging (maximin aggregation): one linear model per group, combined with
weights that maximise the worst-case explained variance.

@author: sibirbil
"""


import warnings

import numpy as np
from qpsolvers import available_solvers, solve_qp
from scipy.sparse import csc_matrix
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

# Tried in order. clarabel is the interior-point solver the package ships with;
# the rest are alternatives for the ill-conditioned H a near-collinear set of
# group coefficients produces.
_SOLVERS = [s for s in ("clarabel", "quadprog", "osqp", "scs", "cvxopt")
            if s in available_solvers]


class SklearnEstimator:
    """
    This base class is dummy, just for guideline. Sklearn does not
    provide a base class that has both fit and predict
    """

    def fit(self, X: np.array, y: np.array):
        raise NotImplementedError("Needs to implement fit(X, y)")

    def predict(self, X0: np.array):
        raise NotImplementedError("Needs to implement predict(X, y)")


class Magging(RegressorMixin, BaseEstimator, SklearnEstimator):
    """Maximin aggregation over environments.

    Parameters
    ----------
        frac: fraction of the sample that forms one group when no environment
            is given, so the number of groups is round(1 / frac) (default is
            0.2, i.e. five groups). Must be in (0, 1].
        random_state: seed for that fallback partition.

    fit() also takes an ``env`` argument: one environment label per row.
    Magging is defined over data that arrives in known environments -- sites,
    batches, seasons, field positions -- each of which may carry its own
    regression coefficients. The labels belong to the sample rather than to the
    estimator, which is why they are a fit argument; inside a GridSearchCV they
    travel as ``regressor__env=...`` and sklearn slices them with the fold.

    Do not pass the suite's fold-grouping array here. Those labels exist to keep
    a duplicated record on one side of a split -- ccpp has 9527 of them for 9568
    rows -- and an environment holding one row is not an environment.

    Without ``env`` the fallback is a random disjoint partition of the rows,
    which is what the main table's MAG column is scored on. Groups drawn that
    way all estimate the same coefficient vector, so their aggregate collapses
    towards the pooled least-squares fit: measured on lasrosas, MAG 201.7
    against OLS 200.2, with the two coefficient vectors at cosine 0.98-0.99.
    That is a property of the setting, not a failure of the solver, and it is
    why the column has to be labelled for what it is.

    Notes
    -----
    Follows Buhlmann and Meinshausen. Each environment is fitted with its own
    intercept and only the slopes enter the aggregate: the maximin effect is a
    slope vector, and a level difference read as a slope is manufactured
    heterogeneity. The aggregate is B w*, where w* minimises w' B' Sigma B w
    over the simplex with Sigma the covariance of the training design, and the
    prediction intercept is refitted as ybar - xbar' coef -- aggregating the
    group intercepts with the same weights would impose the level of whichever
    environment the weight landed on upon all the others, and the optimisation
    solved for the slopes does not ask for it.

    The groups are NOT read off the feature space. Building them from nearest
    neighbours of random anchors would make the subsets local and overlapping,
    which is the construction LESS uses, and the maximin objective would then be
    aggregating neighbourhood fits rather than environments -- a different
    method with the same name.

    Source
    ------
    P. Buhlmann and N. Meinshausen. Magging: maximin aggregation for
    inhomogeneous large-scale data. arXiv:1409.2638, 2014.
    N. Meinshausen and P. Buhlmann. Maximin effects in inhomogeneous large-scale
    data. The Annals of Statistics, 43(4):1801-1830, 2015.
    """

    def __init__(self, frac=0.2, random_state=None):
        self.frac = frac
        self.random_state = random_state

    def _partition(self, n: int, env) -> list:
        if env is not None:
            labels = np.asarray(env)
            if len(labels) != n:
                raise ValueError(f"env has {len(labels)} labels for {n} rows")
            return [np.flatnonzero(labels == u) for u in np.unique(labels)]
        if not 0.0 < self.frac <= 1.0:
            raise ValueError(f"frac must be in (0, 1], got {self.frac}")
        n_groups = max(2, round(1.0 / self.frac))
        rng = np.random.default_rng(self.random_state)
        return np.array_split(rng.permutation(n), n_groups)

    def fit(self, X: np.array, y: np.array, env=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(X)

        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = y.mean()
        Xc = X - self.x_mean_

        parts = [p for p in self._partition(n, env) if len(p) > 0]
        # Groups smaller than the number of features leave the fit
        # underdetermined; LinearRegression returns the minimum-norm solution,
        # which is a legitimate group estimate but a noisy one.
        self.local_models = [LinearRegression().fit(X[p], y[p]) for p in parts]
        B = np.column_stack([m.coef_ for m in self.local_models])
        n_groups = B.shape[1]

        H = B.T @ ((Xc.T @ Xc) / n) @ B
        H = (H + H.T) / 2.0
        # With more groups than features the coefficient vectors are necessarily
        # dependent and H is singular. That is not a degenerate problem -- the
        # objective stays convex, the minimiser just stops being unique -- and
        # pruning groups to restore strict definiteness would silently drop
        # environments from the aggregate. A jitter scaled to H's own trace lets
        # the solver factorise it while staying far below the values it perturbs.
        scale = np.trace(H) / n_groups
        if scale > 0:
            H = H + (1e-10 * scale) * np.eye(n_groups)

        self.w = None
        self.solver_ = None
        for solver in _SOLVERS:
            try:
                w = solve_qp(
                    P=csc_matrix(H),
                    q=np.zeros(n_groups),
                    A=csc_matrix(np.ones((1, n_groups))),
                    b=np.array([1.0]),
                    lb=np.zeros(n_groups),
                    solver=solver,
                )
            except Exception:
                continue
            if isinstance(w, np.ndarray) and np.all(np.isfinite(w)):
                w = np.clip(w, 0.0, None)
                # a solver that satisfies the sum-to-one constraint only
                # approximately can return a vector whose components are all at
                # or below zero; normalising that divides by zero and yields nan
                # coefficients without raising anything
                if w.sum() > 0:
                    self.w, self.solver_ = w / w.sum(), solver
                    break

        if self.w is None:
            # Every solver refused H. Equal weights are feasible but they are
            # not the maximin solution, so the fallback is recorded rather than
            # passed off as one.
            warnings.warn(
                f"no QP solver returned a usable weight vector ({n_groups} "
                "groups); falling back to equal weights, which is not the "
                "maximin aggregate",
                RuntimeWarning, stacklevel=2,
            )
            self.w = np.ones(n_groups) / n_groups
            self.solver_ = "equal-weight fallback"

        self.n_groups_ = n_groups
        self.coef_ = B @ self.w
        self.intercept_ = float(self.y_mean_ - self.x_mean_ @ self.coef_)
        self.is_fitted_ = True

        return self

    def predict(self, X0: np.array):
        if getattr(self, "w", None) is None:
            raise ValueError("You need to fit the Magging model first")

        X0 = np.asarray(X0, dtype=float)
        return self.intercept_ + X0 @ self.coef_

    def explained_variance(self, X, y, env):
        """Per-environment explained variance of the aggregate slope.

        What magging maximises is min_g V_g(b) with
        V_g(b) = 2 b' Cov_g(X, y) - b' Cov_g(X) b, so mean squared error over a
        pooled test set measures something else: V_g is a gain over predicting
        the environment's own mean, and that baseline differs between
        environments. The covariances are taken within the environment, which
        makes this a statement about the slope rather than about the level.

        Returns {environment: V_g}. A negative value means the aggregate slope
        does worse in that environment than the zero vector -- the quantity that
        exposed the old global-centring fit, which reached -169 on lasrosas
        where the per-environment intercepts reach +15.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        labels = np.asarray(env)
        out = {}
        for u in np.unique(labels):
            idx = np.flatnonzero(labels == u)
            if len(idx) < 2:
                continue
            Xg = X[idx] - X[idx].mean(axis=0)
            yg = y[idx] - y[idx].mean()
            cov_xy = Xg.T @ yg / len(idx)
            cov_xx = Xg.T @ Xg / len(idx)
            out[u] = float(2 * self.coef_ @ cov_xy - self.coef_ @ cov_xx @ self.coef_)
        return out
