"""Plots the timing results (logs/timing) to plots/timing/.

    PYTHONPATH=. python plot_funcs/timing_plot.py
"""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd

from src.resultdirs import TIMING

# How these numbers were taken -- idle machine, monotonic clock, median of two
# repeats, every competitor at the parameters the nested CV selected rather than
# the package default -- belongs in the caption, not on the figure itself. So
# does the list of methods the comparison refused: the figure carries bars, the
# caption carries the protocol.
# A bar of zero cannot be drawn on a log axis, and three of these methods fit in
# under a millisecond. They are floored here and the bar is labelled with the
# real reading.
FLOOR = 1e-3


def plot_timing_results():
    log_dir = TIMING
    plot_dir = os.path.join("plots", "timing")
    os.makedirs(plot_dir, exist_ok=True)

    plt.style.use("seaborn-v0_8-darkgrid")

    for filename in os.listdir(log_dir):
        if not filename.endswith(".json"):
            continue

        problem_name = filename.split(".")[0]
        filepath = os.path.join(log_dir, filename)

        with open(filepath) as f:
            data = json.load(f)

        less_av_data = []
        less_gb_data = []
        other_models_data = []

        for model_name, model_results in data.items():
            if model_name == "LESSARegressor":
                for key, time in model_results.items():
                    n_subsets = int(key.split("_")[2])
                    less_av_data.append({"n_subsets": n_subsets, "time": time})
            elif model_name == "LESSBRegressor":
                for key, time in model_results.items():
                    n_subsets = int(key.split("_")[2])
                    less_gb_data.append({"n_subsets": n_subsets, "time": time})
            elif not model_name.startswith("_"):
                other_models_data.append({"model": model_name, "time": model_results})

        less_av_df = pd.DataFrame(less_av_data)
        less_gb_df = pd.DataFrame(less_gb_data)
        other_models_df = pd.DataFrame(other_models_data)

        # Combined plot for LESS-A and LESS-B, fit and predict on one axis.
        # The fit curves are the sweep; the predict curves come from the same
        # models, measured on all n. Without them the figure answers only half of
        # "how expensive is this method".
        less_predict = data.get("_less_predict", {})

        def _curve(model_key):
            d = less_predict.get(model_key, {})
            pts = [(int(k.split("_")[2]), v) for k, v in d.items()]
            return pd.DataFrame(sorted(pts), columns=["n_subsets", "time"])

        if not less_av_df.empty or not less_gb_df.empty:
            fig, ax = plt.subplots(figsize=(9, 6.5))
            series = [
                (less_av_df, "#440154", "-", "o", "LESS-A fit"),
                (less_gb_df, "#c43e3e", "-", "s", "LESS-B fit"),
                (_curve("LESSARegressor"), "#440154", "--", "o", "LESS-A predict"),
                (_curve("LESSBRegressor"), "#c43e3e", "--", "s", "LESS-B predict"),
            ]
            for frame, colour, style, marker, label in series:
                if frame.empty:
                    continue
                f = frame.sort_values("n_subsets")
                ax.plot(f["n_subsets"], f["time"], style, marker=marker,
                        color=colour, label=label, linewidth=2, markersize=6,
                        alpha=0.9 if style == "-" else 0.65)

            ax.set_yscale("log")
            ax.set_xlabel("Number of Subsets", fontsize=16)
            ax.set_ylabel("Time (seconds, log scale)", fontsize=16)
            ax.set_title(f"{problem_name}: LESS Fit and Predict", fontsize=18)
            ax.tick_params(labelsize=14)
            ax.grid(True, which="both", ls="--", linewidth=0.5)
            ax.legend(fontsize=14)
            fig.tight_layout()
            combined_plot_path = os.path.join(plot_dir, f"{problem_name}_less_timing.png")
            fig.savefig(combined_plot_path, dpi=150)
            plt.close(fig)
            print(f"Saved combined LESS timing plot for {problem_name} to {combined_plot_path}")

        # Plot for other models: fit and predict side by side.
        #
        # LocR and KNN keep the training set and do the work at predict time, so a
        # figure of fit times alone ranks them as the cheapest methods in the
        # table -- LocR's fit is 0.00 s on every problem and its predict is 284 s
        # on casp. Both readings are drawn; a method the comparison refused simply
        # has no bar, and the caption says which and why.
        predict = data.get("_predict", {})
        if not other_models_df.empty:
            other_models_df["predict"] = other_models_df["model"].map(
                lambda m: predict.get(m, 0.0))
            order = other_models_df.assign(
                total=lambda d: d["time"] + d["predict"]
            ).sort_values("total", ascending=False)["model"].tolist()
            df = other_models_df.set_index("model").loc[order].reset_index()

            fig, ax = plt.subplots(figsize=(14, 8))
            x = range(len(df))
            w = 0.38
            fit_vals = [max(v, FLOOR) for v in df["time"]]
            pred_vals = [max(v, FLOOR) for v in df["predict"]]
            ax.bar([i - w / 2 for i in x], fit_vals, w, label="fit",
                   color="#21908d")
            ax.bar([i + w / 2 for i in x], pred_vals, w, label="predict",
                   color="#fde725", edgecolor="#8a8a3a")

            for i, (f, p_) in enumerate(zip(df["time"], df["predict"])):
                if f < FLOOR:
                    ax.text(i - w / 2, FLOOR * 1.15, "<1ms", ha="center",
                            fontsize=11, rotation=90)
                if p_ < FLOOR:
                    ax.text(i + w / 2, FLOOR * 1.15, "<1ms", ha="center",
                            fontsize=11, rotation=90)

            # Reference lines: both variants, fit and predict, at the subset
            # count the grid usually picks. Fit lines solid-dashed, predict lines
            # dotted, so a bar can be read against its own kind.
            for frame, key, colour, name in (
                    (less_av_df, "LESSARegressor", "#440154", "LESS-A"),
                    (less_gb_df, "LESSBRegressor", "#c43e3e", "LESS-B")):
                if not frame.empty:
                    row = frame[frame["n_subsets"] == 20]["time"]
                    if not row.empty:
                        ax.axhline(y=row.values[0], color=colour, linestyle="--",
                                   label=f"{name} fit (m=20)", linewidth=2)
                pr20 = less_predict.get(key, {}).get("n_subsets_20")
                if pr20:
                    ax.axhline(y=pr20, color=colour, linestyle=":",
                               label=f"{name} predict (m=20)", linewidth=2)

            ax.set_yscale("log")
            ax.set_xticks(list(x))
            ax.set_xticklabels(df["model"], fontsize=18)
            ax.set_xlim(-0.8, len(df) - 0.2)
            ax.set_ylabel("Time (seconds, log scale)", fontsize=18)
            ax.set_xlabel("")
            ax.set_title(f"{problem_name}: Fit and Predict", fontsize=20)
            ax.tick_params(axis="y", labelsize=16)
            ax.legend(fontsize=12, ncol=2)
            fig.tight_layout()
            other_plot_path = os.path.join(plot_dir, f"{problem_name}_other_models_timing.png")
            fig.savefig(other_plot_path, dpi=150)
            plt.close(fig)
            print(f"Saved other models timing plot for {problem_name} to {other_plot_path}")


if __name__ == "__main__":
    plot_timing_results()
