"""Downloads and builds the lasrosas, edwards, airquality and srdb CSVs in
datasetsR/, together with their group (*_groups.csv) and environment
(*_env.csv) files.

    PYTHONPATH=. python script_funcs/prepare_hetero_datasets.py [--check]
"""

from __future__ import annotations

import os
import sys
import tarfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from src.problems import merge_group_labels

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "datasetsR")
CACHE = os.path.join(OUT_DIR, "_downloads")

AGRIDAT_URL = "https://cran.r-project.org/src/contrib/agridat_1.26.tar.gz"
AIRQUALITY_URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
SRDB_URL = "https://raw.githubusercontent.com/bpbond/srdb/master/srdb-data.csv"


def _agridat(member):
    os.makedirs(CACHE, exist_ok=True)
    tar_path = os.path.join(CACHE, "agridat_1.26.tar.gz")
    if not os.path.exists(tar_path):
        urllib.request.urlretrieve(AGRIDAT_URL, tar_path)
    with tarfile.open(tar_path) as tar:
        with tar.extractfile(member) as fh:
            return pd.read_csv(fh, sep="\t")


def build_lasrosas():
    df = _agridat("agridat/data/lasrosas.corn.txt")
    env = df["topo"].astype(str).values
    out = df[["year", "lat", "long", "nitro", "bv"]].astype(float).copy()
    out["yield"] = df["yield"].values
    return out, None, env


def build_edwards():
    df = _agridat("agridat/data/edwards.oats.txt")

    # 1211 of the 1235 (environment, genotype) cells hold exactly three rows --
    # the trial's three replicate blocks. Same genotype, same location, same
    # year: near-identical feature rows, and a plain KFold would put a plot's
    # replicates on both sides of the split. The cell is the unit that has to
    # stay together, the way ccpp's repeated record does.
    groups = pd.factorize(df["eid"].astype(str) + "_" + df["gen"].astype(str))[0]
    env = df["eid"].astype(str).values

    # block is a design label with no meaning for a plot that has not been sown
    # yet, so it is not a feature. testwt is a trait measured on the same plot
    # at the same harvest -- kept because it is the only continuous covariate
    # the file carries, and flagged here because using a co-outcome as an input
    # is a modelling choice rather than a neutral one.
    out = pd.concat([
        df[["year", "testwt"]].astype(float),
        pd.get_dummies(df["loc"], prefix="loc", dtype=float),
        pd.get_dummies(df["gen"], prefix="gen", dtype=float),
    ], axis=1)
    out["yield"] = df["yield"].values
    return out, groups, env


def build_airquality():
    os.makedirs(CACHE, exist_ok=True)
    zip_path = os.path.join(CACHE, "air_quality.zip")
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(AIRQUALITY_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        with z.open("AirQualityUCI.csv") as fh:
            df = pd.read_csv(fh, sep=";", decimal=",")
    df = df.dropna(axis=1, how="all").dropna(how="all")

    # the file codes a missing reading as -200, and NMHC(GT) is missing in 8443
    # of 9357 rows, so that column goes rather than the rows that carry it
    target = "NOx(GT)"
    feats = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)",
             "PT08.S5(O3)", "T", "RH", "AH"]
    keep = df[feats + [target]].replace(-200, np.nan).dropna()
    df = df.loc[keep.index]

    # the five PT08 channels are the sensor array's own responses and the
    # reference analyser is the target: the task is calibrating one to the
    # other, which is what the drift claim is about. The other ground-truth
    # columns are left out -- they are co-measurements, not inputs.
    date = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    env = date.dt.month.astype(str).values

    # one uninterrupted hourly series: neighbouring hours are nearly the same
    # reading, so a shuffled fold would score memorisation. The day holds them
    # together, the way metro's own grouping does.
    groups = pd.factorize(date.dt.strftime("%Y-%m-%d"))[0]

    out = keep[feats].astype(float).reset_index(drop=True)
    out[target] = keep[target].astype(float).values
    return out, groups, env


def build_srdb():
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "srdb-data.csv")
    if not os.path.exists(path):
        urllib.request.urlretrieve(SRDB_URL, path)
    df = pd.read_csv(path, low_memory=False)

    cols = ["Study_number", "Latitude", "Longitude", "MAT", "MAP",
            "Study_midyear", "Biome", "Rs_annual"]
    df = df[cols]
    # one row carries Rs_annual = -114, which a respiration flux cannot be
    df = df[df["Rs_annual"].notna() & (df["Rs_annual"] > 0)].dropna()
    # Arctic holds 22 rows, too few to estimate anything in, and it would be the
    # environment every group-wise statement then rests on
    df = df[df["Biome"] != "Arctic"].reset_index(drop=True)

    # 5877 rows come from 1391 distinct coordinates -- one site appears 130
    # times -- and from 1225 studies, and the two overlap without nesting. A
    # shuffled fold would put a site's other visits in training and hand a
    # local method the answer. What has to stay together is the connected
    # component of the two relations, the way road3d merges way and coordinate.
    site = pd.factorize(df["Latitude"].astype(str) + "_" + df["Longitude"].astype(str))[0]
    study = pd.factorize(df["Study_number"].astype(str))[0]
    groups = merge_group_labels(site, study)

    env = df["Biome"].astype(str).values
    out = df[["Latitude", "Longitude", "MAT", "MAP", "Study_midyear"]].astype(float)
    out["Rs_annual"] = df["Rs_annual"].values
    return out, groups, env


BUILDERS = {
    "lasrosas": build_lasrosas,
    "srdb": build_srdb,
    "edwards": build_edwards,
    "airquality": build_airquality,
}


def main() -> None:
    check = "--check" in sys.argv
    for name, build in BUILDERS.items():
        frame, groups, env = build()
        X = frame.iloc[:, :-1]
        print(f"{name}: n={len(frame)} d={X.shape[1]} "
              f"target={frame.columns[-1]} "
              f"env={len(set(env))} groups "
              f"({pd.Series(env).value_counts().min()}-"
              f"{pd.Series(env).value_counts().max()} rows each)")
        if groups is not None:
            sizes = pd.Series(groups).value_counts()
            print(f"  fold grouping: {sizes.size} units, "
                  f"{(sizes > 1).sum()} of them hold more than one row")

        paths = {
            f"{name}.csv": frame,
            f"{name}_env.csv": pd.Series(env),
        }
        if groups is not None:
            paths[f"{name}_groups.csv"] = pd.Series(groups)

        for filename, obj in paths.items():
            path = os.path.join(OUT_DIR, filename)
            if check:
                if not os.path.exists(path):
                    print(f"  {filename}: MISSING")
                    continue
                on_disk = pd.read_csv(path, header=None)
                same = on_disk.shape == (len(obj), obj.shape[1] if obj.ndim > 1 else 1)
                print(f"  {filename}: {'shape matches' if same else 'SHAPE DIFFERS'}")
            else:
                obj.to_csv(path, header=False, index=False)
                print(f"  wrote {filename}")


if __name__ == "__main__":
    main()
