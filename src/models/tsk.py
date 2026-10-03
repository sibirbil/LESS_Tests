"""TSK fuzzy regressor: fuzzy c-means antecedents with a configurable
scikit-learn consequent model.
"""

import numpy as np
from pytsk.cluster import FuzzyCMeans
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted


class TSKRegressorPipeline(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        consequent,
        n_cluster=10,
        fuzzy_index="auto",
        sigma_scale="auto",
        init="random",
        tol_iter=100,
        error=1e-6,
        dist="euclidean",
        verbose=0,
        order=1,
        random_state=None,
    ):
        if consequent is None:
            raise ValueError("You must pass a regressor as 'consequent'.")
        
        self.n_cluster = n_cluster
        self.fuzzy_index = fuzzy_index
        self.sigma_scale = sigma_scale
        self.init = init
        self.tol_iter = tol_iter
        self.error = error
        self.dist = dist
        self.verbose = verbose
        self.order = order
        self.consequent = consequent
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        # Step 1: Fuzzy C-Means as antecedent
        self.antecedent_ = FuzzyCMeans(
            n_cluster=self.n_cluster,
            fuzzy_index=self.fuzzy_index,
            sigma_scale=self.sigma_scale,
            init=self.init,
            tol_iter=self.tol_iter,
            error=self.error,
            dist=self.dist,
            verbose=self.verbose,
            order=self.order,
        )

        # FuzzyCMeans takes no random_state: with init="random" it draws the
        # initial membership grid from the global NumPy state, so the same fold
        # at the same grid point did not give the same answer twice and a
        # recorded score could not be reproduced. Handing it a pre-drawn grid
        # would be the clean fix and the library documents that path, but it is
        # unreachable -- its own check reads `if self.init == "random"`, which
        # raises on an array before the array branch is tested. So the legacy
        # global state is seeded around the one call that consumes it and put
        # back afterwards, which leaves the rest of the process untouched.
        state = np.random.get_state()
        try:
            if self.random_state is not None:
                np.random.seed(self.random_state % (2 ** 32))
            X_trans = self.antecedent_.fit_transform(X)
        finally:
            np.random.set_state(state)

        # Step 2: Fit consequent. Cloned rather than fitted in place, so one
        # grid point cannot leave a fitted model behind for the next.
        self.consequent_ = clone(self.consequent)
        self.consequent_.fit(X_trans, y.ravel())
        return self

    def predict(self, X):
        check_is_fitted(self, ["antecedent_", "consequent_"])
        X = np.asarray(X, dtype=np.float32)
        X_trans = self.antecedent_.transform(X)
        return self.consequent_.predict(X_trans)
