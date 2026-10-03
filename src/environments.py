"""Environment labels for the problems that have one.

Used by the Magging model, which fits one model per environment. A label is only
used when it gives at least three groups and enough rows per group to fit a
model in each.
"""
from __future__ import annotations

import os

import pandas as pd

# problem -> file holding one environment label per row
ENVIRONMENT_FILES = {
    "LASROSAS": "lasrosas_env.csv",      # landscape position, 4
    "AIRQUALITY": "airquality_env.csv",  # month, 12
    "SRDB": "srdb_env.csv",              # biome, 6
    "EDWARDS": "edwards_env.csv",        # trial, 34 -- refused by the size rule
}

MIN_GROUPS = 3
ROWS_PER_FEATURE = 5


def environment_for(problem: str, data_path: str, n_features: int):
    """Integer environment codes for `problem`, or None when it has no usable one.

    Returns (codes, note). `note` says why a label was refused, so the refusal
    can be recorded beside the score instead of looking like an oversight.
    """
    filename = ENVIRONMENT_FILES.get(problem)
    if filename is None:
        return None, None

    path = os.path.join(data_path, filename)
    if not os.path.exists(path):
        return None, f"{filename} not found"

    labels = pd.read_csv(path, header=None).iloc[:, 0].values
    codes = pd.factorize(labels)[0].astype(float)
    sizes = pd.Series(codes).value_counts()
    need = ROWS_PER_FEATURE * n_features

    if sizes.size < MIN_GROUPS:
        return None, f"only {sizes.size} environments"
    if sizes.median() < need:
        return None, (f"median environment holds {int(sizes.median())} rows "
                      f"against {n_features} features; a group model needs "
                      f"at least {need}")
    return codes, None
