"""Cost projection run before each model/problem cell in main.py.

The model is fitted on a few small subsamples, a growth curve is fitted to the
measured time and memory, and the cost of the full nested search is projected
from it. A cell whose projection exceeds the time budget or the memory cap is
skipped and the reason is recorded. Measurements are cached in _probes.json.

Memory is measured in a fresh child process: run_probe re-runs this file with
--probe-child.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import itertools
import json
import logging
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from config import N_IN_CV, N_OUT_CV, RANDOM_STATE

# Not every method reaches these row counts, and finding that out by starting a
# fit that never returns wastes a sweep. Rather than hard-coding which ones, the
# cost of each (model, problem) pair is measured: the model is fitted at growing
# subsample sizes until one of them takes long enough to extrapolate from, a
# power law is fitted to those timings, and the whole nested search is projected
# from it. A projection over the budget refuses the cell and records the number,
# so the table prints a dash next to a figure anyone can check. The same rule
# covers a method added later -- there is nothing per-model to keep in sync.
PROBE_SHARES = (1 / 16, 1 / 8, 1 / 4)   # probe at these shares of the fold
PROBE_MIN_ROWS = 200        # ... but never below this
PROBE_MEMORY_SHARE = 0.25   # a probe point may use this much of the cell's cap
PROBE_MAX_ROWS = 20_000     # and never more rows than this, whatever the fold
PROBE_TIMEOUT = 300.0       # a probe child that runs longer than this is killed
MEMORY_MARGIN = 1.5         # projections are a floor, not a ceiling
MEMORY_FLOOR_GB = 0.01      # below this the probe is measuring noise


def default_memory_cap() -> float:
    """What one cell may take: a share of the machine, not a fixed constant.

    A cap has to come from somewhere real or it is just a number that happens
    to refuse things. Sixty per cent leaves room for the operating system and
    for the parent process holding the fold while the workers hold their copies.
    """
    try:
        import psutil
        return round(0.6 * psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:
        return 8.0
PROBE_MIN_POINTS = 3        # never extrapolate from fewer than this many sizes

logger = logging.getLogger(__name__)


def _peak_rss_gb() -> float:
    """This process's high-water mark. ru_maxrss is bytes on macOS, kB on Linux."""
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024 ** 3 if sys.platform == "darwin" else peak / 1024 ** 2


def probe_child(args) -> None:
    """Fit one configuration in this fresh process and report cost on stdout.

    Peak memory has to be measured in a process that started clean: an RSS delta
    taken inside the runner counts the allocator's high-water mark from every
    fit before it, which made a model that holds nothing look like it holds
    gigabytes. A child that only ever loads this subsample and fits once gives a
    figure whose growth across sizes is the model's own.
    """
    payload = np.load(args.probe_data)
    X, y = payload["X"], payload["y"]
    params = json.loads(args.probe_params)
    module_name, _, class_name = args.probe_estimator.rpartition(".")
    estimator_class = getattr(importlib.import_module(module_name), class_name)
    n_q = max(1, len(X) // 4)

    def build():
        return make_model(estimator_class(**json.loads(args.probe_init))
                          .set_params(**params))

    # One throwaway fit first. Before it, the high-water mark covers only the
    # interpreter and the data; after it, also whatever the library allocates
    # once -- xgboost and lightgbm initialise on first use and the allocator
    # keeps its arenas. Charging that to the model made a fixed 150 MB look
    # like memory that grows with n.
    # Enough rows that a candidate with a minimum sample requirement is happy
    # -- KNN with n_neighbors=100 refuses 64 -- and tolerant of the ones that
    # are still unhappy, since a failed warm-up only costs accuracy in the
    # memory baseline, while a failed probe costs the whole cell.
    warm_n = min(len(X), 512)
    try:
        warm = build()
        warm.fit(X[:warm_n], y[:warm_n])
        warm.predict(X[: min(8, len(X))])
        del warm
    except Exception as exc:
        print(f"WARMUP-SKIPPED {type(exc).__name__}", file=sys.stderr)

    baseline = _peak_rss_gb()
    model = build()
    start = time.perf_counter()
    model.fit(X, y)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    model.predict(X[:n_q])
    predict_seconds = time.perf_counter() - start

    print("PROBE " + json.dumps({
        "n": len(X),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "predict_rows": n_q,
        "peak_gb": max(_peak_rss_gb() - baseline, 0.0),
    }))


class NotProbeable(Exception):
    """The child cannot rebuild this estimator from a JSON description."""


def estimator_spec(estimator):
    """(import path, constructor kwargs) for the probe child to rebuild this.

    The child starts from a clean interpreter, so it cannot be handed the
    estimator itself; it is handed what it takes to build an equivalent one,
    which also keeps the probe honest about constructor arguments that change
    the cost, such as how many cores a forest is allowed. An estimator whose
    constructor demands an argument, or whose grid carries other estimators as
    values, cannot be described this way -- TSK takes a consequent model -- and
    says so instead of failing inside the child.
    """
    cls = type(estimator)
    try:
        defaults = cls().get_params(deep=False)
    except TypeError as exc:
        raise NotProbeable(
            f"{cls.__name__} cannot be constructed without arguments: {exc}"
        ) from exc
    kwargs = {k: v for k, v in estimator.get_params(deep=False).items()
              if k in defaults and repr(v) != repr(defaults[k])
              and isinstance(v, (int, float, str, bool, type(None)))}
    return f"{cls.__module__}.{cls.__qualname__}", kwargs


def run_probe(estimator, params, X, y):
    """One probe point, measured in a child process. None if the child died."""
    path_to_class, init_kwargs = estimator_spec(estimator)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as fh:
        path = fh.name
    try:
        np.savez(path, X=X, y=y)
        # A probe that outlives the cap has already answered the question it was
        # asked: this configuration is too slow at a fraction of the real size.
        out = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--probe-child",
             "--probe-estimator", path_to_class, "--probe-data", path,
             "--probe-init", json.dumps(init_kwargs),
             "--probe-params", json.dumps(params)],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": REPO_ROOT},
            check=False, timeout=PROBE_TIMEOUT,
        )
        for line in out.stdout.splitlines():
            if line.startswith("PROBE "):
                return json.loads(line[6:])
        logger.warning(f"probe child failed for {path_to_class} at n={len(X)}: "
                        f"{out.stderr.strip().splitlines()[-1:] or out.returncode}")
        return None
    except subprocess.TimeoutExpired:
        logger.warning(f"probe child for {path_to_class} at n={len(X)} exceeded "
                        f"{PROBE_TIMEOUT:.0f}s and was killed")
        return {"timed_out": True, "n": len(X)}
    finally:
        os.unlink(path)


def _cost_curve(sizes, values, default_exponent=1.0):
    """Fit cost(n) = t0 + a * n^p and return a callable plus (a, p, t0).

    A pure power law in log space reads a model's fixed start-up cost as growth:
    on the smallest subsamples LESS spends most of its time on overhead that does
    not scale, which flattens or steepens the exponent depending on where the
    probe happens to sit. Fitting an offset separates the two. With too few
    points to identify three parameters the caller's assumption is used instead
    of a silently linear one -- fit cost grows with n, cost per predicted row
    usually does not.
    """
    sizes = np.asarray(sizes, float)
    values = np.maximum(np.asarray(values, float), 1e-12)
    if len(sizes) < 2:
        a, p, t0 = float(values[0] / sizes[0] ** default_exponent), default_exponent, 0.0
    elif len(sizes) < 3:
        p, log_a = np.polyfit(np.log(sizes), np.log(values), 1)
        a, p, t0 = float(np.exp(log_a)), float(p), 0.0
    else:
        best = None
        # The exponent can only be as steep as the measurements support. Three
        # points over a fourfold range cannot tell a quadratic from a noisy
        # flat line, and the scan will happily pick whichever fits best: on
        # road3d it read 0.113, 0.137 and 0.257 GB as growth of order n^2.5 and
        # projected 158 GB for a cell that had already run three times in
        # 0.1 hours. The cap is the slope between the extreme points, with a
        # little room, so an exponent above what was seen is never invented.
        # The slope is read off consecutive pairs, and only where both ends are
        # above the noise floor: a probe point that measured zero turns a log
        # ratio into whatever the floor happens to be, which is how an exponent
        # of 3.5 came out of timings that were all but flat.
        floor = max(1e-6, 1e-3 * max(values))
        slopes = [np.log(values[i + 1] / values[i]) / np.log(sizes[i + 1] / sizes[i])
                  for i in range(len(sizes) - 1)
                  if values[i] > floor and values[i + 1] > floor]
        p_cap = min(3.5, max(1.0, max(slopes) * 1.25)) if slopes else 1.0
        # p is not identifiable jointly with t0 by least squares in closed form,
        # so it is scanned over the range these methods actually live in and the
        # linear part solved exactly at each step.
        for p_try in np.arange(0.25, p_cap + 1e-9, 0.05):
            basis = np.column_stack([np.ones_like(sizes), sizes ** p_try])
            coef, *_ = np.linalg.lstsq(basis, values, rcond=None)
            if coef[0] < 0:
                # A negative fixed cost is not a cost. Clipping it to zero and
                # keeping the slope would report a curve that was never fitted,
                # so the slope is solved again with the offset held at zero.
                slope = float(np.sum(values * sizes ** p_try)
                              / max(np.sum(sizes ** (2 * p_try)), 1e-30))
                coef = np.array([0.0, slope])
            resid = float(np.sum((basis @ coef - values) ** 2))
            if coef[1] > 0 and (best is None or resid < best[0]):
                best = (resid, float(coef[1]), float(p_try), float(coef[0]))
        if best is None:
            p, log_a = np.polyfit(np.log(sizes), np.log(values), 1)
            a, p, t0 = float(np.exp(log_a)), float(p), 0.0
        else:
            _, a, p, t0 = best
    return (lambda n: t0 + a * n ** p), (a, p, t0)


def _candidate_params(grid):
    """Every point of the grid, as kwargs for set_params."""
    keys = list(grid)
    values = [grid[k] for k in keys]
    for k, vs in zip(keys, values, strict=True):
        for v in vs:
            if not isinstance(v, (int, float, str, bool, type(None))):
                raise NotProbeable(
                    f"grid value {k}={v!r} does not survive a JSON round trip"
                )
    return [dict(zip(keys, combo, strict=True))
            for combo in itertools.product(*values)]


def probe_cost(make_estimator, grid, X, y, n_train, n_test, n_jobs=1,
               n_outer=N_OUT_CV, memory_cap_gb=None):
    """Project what one cell of the nested search costs, by measuring it small.

    Two stages. First every candidate is fitted on the smallest subsample, which
    gives their cost relative to one another -- a grid that mixes 100 and 200
    estimators does not cost the same per candidate, and timing only the default
    configuration would miss that. Then the slowest candidate is grown over
    several sizes and a power law is fitted to it, so both exponents come from
    the configuration that dominates the bill. Every measurement runs in a fresh
    child process, which is what makes the memory figure mean anything.

    Returns (projected_seconds, projected_peak_gb, samples); samples carries the
    raw probe so the projection can be audited instead of trusted.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    probe_memory_ceiling = PROBE_MEMORY_SHARE * (memory_cap_gb
                                                 or default_memory_cap())
    # Sizes are a fraction of the fold rather than a fixed 1000, so the largest
    # probe is always a quarter of the real thing: three points spanning a factor
    # of four, and never more than a 4x extrapolation. A fixed start could not
    # reach three points under the ceiling on a mid-sized fold, and the fallback
    # that kicked in read fixed cost as growth -- it refused LESS on a 20k-row
    # problem over a projected 11.6 GB it does not allocate.
    # The points are fractions of the largest probe, not of the fold, so the
    # row cap cannot collapse them onto each other. It did on road3d: n/16, n/8
    # and n/4 were all above 20,000, all three clipped to the cap, and the set
    # left one point. With one point there is no slope to measure, the defaults
    # took over -- cost per predicted row held constant -- and LOCR, whose
    # predict walks the whole training set for every test row, was projected at
    # 1.05 h for a cell that measurement puts near fifty.
    top = min(n_train, PROBE_MAX_ROWS, max(PROBE_MIN_ROWS,
                                           int(n_train * max(PROBE_SHARES))))
    plan = sorted({max(PROBE_MIN_ROWS, int(top * f))
                   for f in (0.25, 0.5, 1.0)})
    plan = [n for n in plan if n <= n_train]
    n0 = plan[0]
    idx0 = rng.choice(len(X), n0, replace=False)

    candidates = _candidate_params(grid)
    # Relative candidate cost has to include predict: for a memory-based method
    # the fit is nearly free and the whole bill is at prediction time, so ranking
    # candidates by fit alone would model LOCR exactly backwards.
    #
    # This stage runs in-process. Only the memory figure needs a clean
    # interpreter, and that comes from the size sweep below; timing eighteen
    # candidates in eighteen children spent more on process start-up than the
    # whole grid search it was protecting -- on a 506-row problem the probe was
    # the slow part.
    costs = []
    for params in candidates:
        model = make_model(make_estimator().set_params(**params))
        start = time.perf_counter()
        model.fit(X[idx0], y[idx0])
        n_q = max(1, n0 // 4)
        model.predict(X[idx0][:n_q])
        costs.append(time.perf_counter() - start)
        del model
    release()
    if not np.all(np.isfinite(costs)) or max(costs) <= 0:
        return np.inf, np.inf, {"sizes": [n0], "child_failed": True,
                                "n_candidates": len(candidates)}
    slowest = candidates[int(np.argmax(costs))]
    ratio_sum = float(np.sum(costs) / max(costs))

    sizes, fit_times, per_row, mem = [], [], [], []
    timed_out_at = None
    incomplete = False
    for size in plan:
        idx = rng.choice(len(X), size, replace=False)
        got = run_probe(make_estimator(), slowest, X[idx], y[idx])
        if got is None:
            # A child that died after the first point is evidence too: the cost
            # is somewhere past what the smaller points show, so the cell is
            # refused rather than priced from the points that did return.
            incomplete = True
            break
        if got.get("timed_out"):
            incomplete = True
            # The completed points are what the curves are fitted on; the
            # timeout itself is recorded, being the strongest evidence there is.
            timed_out_at = size
            break
        sizes.append(size)
        fit_times.append(got["fit_seconds"])
        per_row.append(got["predict_seconds"] / got["predict_rows"])
        mem.append(got["peak_gb"])
        # Stop growing once a probe point is itself heavy. A method whose
        # memory goes as n^2 reaches the machine's limit inside the probe: GPR
        # on a 150k fold was measured at a quarter of that, 37,500 rows, which
        # is an 11 GB kernel matrix -- the probe drove the machine into swap
        # before it could report that the real thing does not fit. Two points
        # already show a quadratic, and refusing on them is the same answer.
        if got["peak_gb"] > probe_memory_ceiling:
            logger.info(f"probe stopped at {size} rows: {got['peak_gb']:.1f} GB "
                         f"is over the {probe_memory_ceiling:.1f} GB a probe "
                         f"point may use")
            break

    if not sizes:
        return np.inf, np.inf, {"child_failed": True,
                                "n_candidates": len(candidates)}
    if len(sizes) < 2 and n_train > sizes[0] * 2:
        # One measurement gives a value, not a trend. Extrapolating it means
        # assuming the shape of the curve, and the assumption is what went
        # wrong before, so the cell is refused for want of evidence instead.
        return np.inf, np.inf, {
            "sizes": sizes, "single_point": True,
            "fit_seconds": [round(v, 4) for v in fit_times],
            "peak_gb": [round(v, 4) for v in mem],
            "n_candidates": len(candidates),
        }

    fit_at, fit_coef = _cost_curve(sizes, fit_times, default_exponent=1.0)
    # cost per predicted row: flat for a model that carries parameters, growing
    # for one that carries the training set
    pred_row_at, pred_coef = _cost_curve(sizes, per_row, default_exponent=0.0)
    # Memory is only fitted when the probe actually saw some. A method that holds
    # nothing measurable at these sizes produced peaks of a megabyte or two, and
    # a curve through those reads their noise as growth -- which had LOCR, whose
    # memory is the training set it already holds, refused over a projected
    # 12 GB it will never allocate. Below the floor the largest measurement is
    # scaled linearly instead, and the projection records that it did.
    measurable = [(n, g) for n, g in zip(sizes, mem, strict=True)
                  if g >= MEMORY_FLOOR_GB]
    mem_measurable = len(measurable) >= min(PROBE_MIN_POINTS, len(sizes))
    if mem_measurable:
        mem_at, mem_coef = _cost_curve([n for n, _ in measurable],
                                       [g for _, g in measurable],
                                       default_exponent=1.0)
    else:
        # The steepest point, not the biggest: memory per row is what
        # extrapolates, and the largest measurement can come from the largest
        # subsample while another one grew faster per row.
        slope_gb = max(g / n for n, g in zip(sizes, mem, strict=True))
        mem_coef = (slope_gb, 1.0, 0.0)

        def mem_at(n, _slope=mem_coef[0]):
            return _slope * n

    n_inner = int(n_train * (N_IN_CV - 1) / N_IN_CV)
    n_val = n_train - n_inner
    # The budget is spent on a whole model-by-dataset cell, which is every outer
    # fold. Inside a fold the grid's candidates run in parallel, so that part is
    # shared between the workers; the winner's refit and the test prediction
    # that follow are a single model on one core and are not.
    # An estimator built with n_jobs=-1 already saturates the machine, so the
    # grid's workers do not multiply the throughput -- counting both would make
    # the projection optimistic by exactly the factor that matters most.
    _, probe_init = estimator_spec(make_estimator())
    # xgboost and lightgbm thread across every core unless told otherwise, so
    # their default is as saturating as an explicit -1; counting five grid
    # workers on top of that would make the projection optimistic by the factor
    # that matters most.
    # Only the gradient-boosting libraries take the whole machine when left
    # alone: xgboost and lightgbm read n_jobs=None as "every core", while
    # scikit-learn reads it as one thread. Treating the scikit-learn forest as
    # saturating dropped the grid to a single worker and left RF-D crawling on
    # one core for hours.
    threads_by_default = ("XGB", "LGBM")
    probe_class = type(make_estimator()).__name__
    saturating = (probe_init.get("n_jobs") in (-1, 0)
                  or (probe_init.get("n_jobs") is None
                      and any(t in probe_class for t in threads_by_default)))
    workers = 1 if saturating else max(1, min(n_jobs, len(candidates)))
    grid_serial = N_IN_CV * ratio_sum * (fit_at(n_inner)
                                        + pred_row_at(n_inner) * n_val)
    serial_part = fit_at(n_train) + pred_row_at(n_train) * n_test
    seconds = n_outer * (grid_serial / workers + serial_part)
    # A projection is a floor: it misses the allocator's fragmentation, the
    # copies a library makes internally, and the workers holding a fold each.
    # A fixed cost stays fixed however many workers there are; only the part
    # that grows with the fold is paid per worker. Multiplying the whole
    # projection by five turned a 0.15 GB measurement into 11.6 GB.
    fixed_gb = min(mem)
    grows_gb = max(mem_at(n_train) - fixed_gb, max(mem) - fixed_gb, 0.0)
    peak_gb = fixed_gb + MEMORY_MARGIN * grows_gb * workers

    samples = {
        "sizes": [int(v) for v in sizes],
        "fit_seconds": [round(v, 4) for v in fit_times],
        "peak_gb": [round(v, 4) for v in mem],
        "fit_curve": {"a": fit_coef[0], "exponent": round(fit_coef[1], 3),
                      "offset": round(fit_coef[2], 4)},
        "predict_row_exponent": round(pred_coef[1], 3),
        "memory_exponent": round(mem_coef[1], 3),
        "memory_measurable": mem_measurable,
        "extrapolation_factor": round(n_train / max(sizes), 1),
        "probe_timed_out_at": timed_out_at,
        "probe_incomplete": incomplete,
        "n_candidates": len(candidates),
        "grid_workers": workers,
        "outer_folds": n_outer,
        # The pieces the decision is made from, so the same measurement can be
        # re-costed at another worker count without probing again. Memory is
        # what parallelism buys and pays for: a cell too big for five workers is
        # often perfectly feasible for one.
        "grid_seconds_serial": grid_serial,
        "serial_seconds": serial_part,
        "fixed_gb": fixed_gb,
        "grows_gb": grows_gb,
        "estimator_saturates_cores": saturating,
        "candidate_cost_ratio_sum": round(ratio_sum, 2),
        "memory_margin": MEMORY_MARGIN,
        "slowest_candidate": {k: (v if isinstance(v, (int, float, str, bool))
                                  else repr(v)) for k, v in slowest.items()},
    }
    return float(seconds), float(peak_gb), samples


def probe_cache_path(result_dir: str) -> str:
    return os.path.join(result_dir, "_probes.json")


def load_probe_cache(result_dir: str) -> dict:
    """Probes cost real minutes and are deterministic in what the key names, so
    a resumed run reads them back instead of measuring again."""
    path = probe_cache_path(result_dir)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def budget_check(model_name, make_estimator, grid, X_train, y_train, n_test,
                 args, cache):
    """(reason to refuse this cell, the projection behind it) or (None, projection)."""
    # The key carries everything the measurement depends on: the same model on
    # the same rows still costs differently under another grid, and the test
    # size drives the predict term. The budget is deliberately not part of it --
    # the probe measures cost, the budget only decides what to do with it, so
    # raising the budget reuses the measurement and reconsiders the refusal.
    def bucket(n):
        """Fold sizes differ by a row or two, which is not worth a fresh probe.

        Rounding alone would let two different datasets of a similar shape share
        a measurement, so the key carries a fingerprint of the data as well.
        """
        return int(float(f"{n:.3g}"))

    fingerprint = hashlib.sha256()
    fingerprint.update(np.ascontiguousarray(X_train[:64]).tobytes())
    fingerprint.update(np.ascontiguousarray(y_train[:64]).tobytes())
    fingerprint.update(str((X_train.shape, float(y_train.mean()))).encode())
    key = "|".join([
        model_name, str(bucket(X_train.shape[0])), str(X_train.shape[1]),
        str(bucket(n_test)), str(args.grid_n_jobs),
        repr(sorted((k, str(v)) for k, v in grid.items())),
        fingerprint.hexdigest()[:16],
        # the probe's own settings and the cap change what was measured, so a
        # cached measurement taken under other settings must not be reused
        f"probe={PROBE_SHARES}/{PROBE_MAX_ROWS}/{PROBE_MEMORY_SHARE}"
        f"/{MEMORY_MARGIN}/{args.memory_cap_gb}",
    ])
    if key not in cache:
        logger.info(f"{model_name}: cost probe on {X_train.shape[0]} rows")
        try:
            cache[key] = list(probe_cost(
                make_estimator, grid, X_train, y_train, X_train.shape[0], n_test,
                n_jobs=args.grid_n_jobs,
                n_outer=1 if getattr(args, "one_outer_fold", False) else N_OUT_CV,
                memory_cap_gb=args.memory_cap_gb))
        except NotProbeable as exc:
            # Not a refusal. The method simply cannot be described to a child
            # process, so this cell runs without a projection instead of being
            # thrown away on a technicality.
            logger.info(f"{model_name}: not probeable ({exc}), running unbudgeted")
            cache[key] = [0.0, 0.0, {"not_probeable": str(exc)}]
        with open(probe_cache_path(args.result_dir), "w") as f:
            json.dump(cache, f, indent=2)

    _, _, samples = cache[key]

    if samples.get("not_probeable"):
        return None, {"probe": samples, "n_jobs": args.grid_n_jobs}
    if samples.get("child_failed"):
        rows = samples.get("sizes", [0])[0]
        return (f"the probe could not complete one fit at {rows} rows inside "
                f"the {PROBE_TIMEOUT:.0f}s timeout"), {"probe": samples}
    if samples.get("single_point"):
        return (f"the probe collapsed to one size ({samples['sizes'][0]} rows) "
                f"against a fold of {X_train.shape[0]}, so no growth rate was "
                f"measured and the cost cannot be bounded"), {"probe": samples}
    if samples.get("probe_incomplete"):
        # The probe stopped early because a child timed out or died. The points
        # that did return describe a cheaper problem than the real one, so
        # re-pricing from them would turn evidence of cost into acceptance.
        done = samples.get("sizes", [])
        return (f"the probe stopped at {done[-1] if done else 0} rows "
                f"({'timed out' if samples.get('probe_timed_out_at') else 'child died'}), "
                f"so the cost at {'the full fold'} is not bounded"), {"probe": samples}

    # Memory is bought with parallelism, so before refusing a cell the same
    # measurement is re-costed at fewer workers: GPR on cadata needs about 12 GB
    # per worker, which does not fit five ways on this machine but fits once.
    # The largest worker count that stays under the cap wins; running one cell
    # serially is a smaller loss than dropping it from the table.
    n_outer = samples.get("outer_folds", N_OUT_CV)
    requested = samples.get("grid_workers", 1)
    chosen = None
    for workers in range(requested, 0, -1):
        gb = samples["fixed_gb"] + MEMORY_MARGIN * samples["grows_gb"] * workers
        hours = n_outer * (samples["grid_seconds_serial"] / workers
                           + samples["serial_seconds"]) / 3600
        if gb <= args.memory_cap_gb:
            chosen = (workers, gb, hours)
            break
    if chosen is None:
        gb = samples["fixed_gb"] + MEMORY_MARGIN * samples["grows_gb"]
        projection = {
            "projected_hours": None, "projected_peak_gb": round(gb, 2),
            "budget_hours": args.budget_hours,
            "memory_cap_gb": args.memory_cap_gb,
            "n_jobs": 1, "probe": samples,
        }
        return (f"projected peak memory {gb:.1f} GB even with a single grid "
                f"worker, over the {args.memory_cap_gb:.0f} GB cap "
                f"(probe exponent {samples.get('memory_exponent', '?')})"), projection

    workers, peak_gb, hours = chosen
    projection = {
        "projected_hours": round(hours, 2),
        "projected_peak_gb": round(peak_gb, 2),
        "budget_hours": args.budget_hours,
        "memory_cap_gb": args.memory_cap_gb,
        "n_jobs": workers,
        "probe": samples,
    }
    if hours > args.budget_hours:
        return (f"projected {hours:.1f} h over the {args.budget_hours:.0f} h "
                f"budget (probe exponent "
                f"{samples.get('fit_curve', {}).get('exponent', '?')})"), projection
    note = "" if workers == requested else f", throttled to {workers} grid worker(s)"
    logger.info(f"{model_name}: projected {hours:.2f} h, {peak_gb:.2f} GB peak "
                 f"-- within budget{note}")
    return None, projection


def make_model(estimator):
    """The pipeline every model is measured through: features and target scaled."""
    return TransformedTargetRegressor(
        regressor=Pipeline([("scaler", StandardScaler()), ("regressor", estimator)]),
        transformer=StandardScaler(),
    )


def release() -> None:
    """Collect what the caller has just dropped, before it loads the next problem.

    Several of these datasets are hundreds of megabytes and the largest is 4.2M
    rows; holding two at once is what gets the process killed part way through a
    multi-problem run. Freeing them has to happen in the caller -- passing the
    arrays to a function only drops that function's own reference, while the
    caller's names keep them alive -- so the caller does `del` and this only
    forces the collection.
    """
    gc.collect()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--probe-child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--probe-estimator", default=None, help=argparse.SUPPRESS)
    p.add_argument("--probe-init", default="{}", help=argparse.SUPPRESS)
    p.add_argument("--probe-data", default=None, help=argparse.SUPPRESS)
    p.add_argument("--probe-params", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    if not args.probe_child:
        p.error("this module is only run as a probe child; see run_probe")
    # A probe child measures one fit and exits; it must not touch the log files
    # or the protocol manifest of the run that spawned it.
    probe_child(args)
