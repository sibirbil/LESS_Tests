"""scikit-learn wrapper around the ANFISpy ANFIS model."""

import numpy as np
import torch
from ANFISpy import ANFIS
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class SklearnAnfisRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_sets=3,
        mf_type="gaussian",
        lr=0.001,
        epochs=100,
        batch_size=32,
        verbose=False,
        n_features=3,
        and_operator=torch.prod,
        # None means nn.Identity(); it is built in fit() rather than here so the
        # default is not a shared mutable module instance across estimators
        output_activation=None,
        mean_rule_activation=False,
    ):
        self.n_sets = n_sets
        self.mf_type = mf_type
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.n_features = n_features
        self.and_operator = and_operator
        self.output_activation = output_activation
        self.mean_rule_activation = mean_rule_activation

        var_names = [f"x{i}" for i in range(n_features)]
        self.variables = {
            "inputs": {
                "n_sets": [n_sets] * n_features,
                "uod": [(0, 1)] * n_features,
                "var_names": var_names,
                "mf_names": [[]] * n_features,
            },
            "output": {
                "var_names": ["y"],
                "n_classes": 1,
            },
        }

        self.is_fitted_ = False

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        dataset = DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(y)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.model = ANFIS(
            self.variables,
            self.mf_type,
            and_operator=self.and_operator,
            output_activation=(
                self.output_activation if self.output_activation is not None else nn.Identity()
            ),
            mean_rule_activation=self.mean_rule_activation,
        )
        self.model.train()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in dataset:
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {total_loss:.4f}")

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            preds = self.model(torch.tensor(X)).numpy()
        return preds.ravel()

    def score(self, X, y):
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)
