"""Plots the LESS variant results (logs/variants) to plots/variants/."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from src.resultdirs import VARIANTS

# Set up plotting style
plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 24,
    }
)

# Create output directory
os.makedirs("plots/variants", exist_ok=True)

# Problem names
# Keyed by problem so a problem whose results are missing drops its label with
# it; keeping two positional lists in step is what used to break the plot.


def _complete_problems(result_dir, n_arms=8, n_folds=4):
    """Problems from src.problems whose cells are all there, in suite order.

    Two faults are closed here. The list used to be written out by hand and
    stopped at the seven problems the paper reports, so a figure kept covering
    seven while the tables covered eighteen and nothing said so. And a problem
    with three folds of an arm used to be drawn beside one with four, because the
    loaders average whatever they find -- a partial sweep would quietly shorten a
    bar rather than leave a gap.
    """
    import json as _json
    import os as _os

    from src.problems import ALL_PROBLEMS

    out, partial = [], []
    for name in ALL_PROBLEMS:
        path = _os.path.join(result_dir, f"{name}.json")
        if not _os.path.exists(path):
            continue
        with open(path) as fh:
            data = _json.load(fh)
        cells = {k: len(v) for k, v in data.items()
                 if not k.startswith("_") and isinstance(v, dict)}
        if len(cells) == n_arms and set(cells.values()) == {n_folds}:
            out.append(name)
        else:
            partial.append(f"{name} ({sum(cells.values())}/{n_arms * n_folds})")
    if partial:
        print(f"  atlandi (eksik hucre): {', '.join(partial)}")
    return out


def _label(name):
    """Short axis label: the long names collide even when turned."""
    key = name.replace("_GROUPED", "").lower()
    short = {"cpusmallscale": "cpusmall", "blogfeedback": "blogfeed",
             "airquality": "airqual"}
    return short.get(key, key)


problems = _complete_problems(VARIANTS)
LABELS = {name: _label(name) for name in problems}

# Load and process data
av_data = []
gb_data = []
x_labels = []

for problem in problems:
    result_path = os.path.join(VARIANTS, f"{problem}.json")

    if os.path.exists(result_path):
        with open(result_path) as f:
            results = json.load(f)

        # Calculate means for each variant
        av_row = []
        gb_row = []

        # AV variants: NoV-NoC, NoV-C, V-NoC, V-C
        for variant in ["AV-NoV-NoC", "AV-NoV-C", "AV-V-NoC", "AV-V-C"]:
            if variant in results:
                fold_scores = [
                    results[variant][str(i)] for i in range(1, 6) if str(i) in results[variant]
                ]
                av_row.append(np.mean(fold_scores))
            else:
                av_row.append(np.nan)

        # GB variants: NoV-NoC, NoV-C, V-NoC, V-C
        for variant in ["GB-NoV-NoC", "GB-NoV-C", "GB-V-NoC", "GB-V-C"]:
            if variant in results:
                fold_scores = [
                    results[variant][str(i)] for i in range(1, 6) if str(i) in results[variant]
                ]
                gb_row.append(np.mean(fold_scores))
            else:
                gb_row.append(np.nan)

        av_data.append(av_row)
        gb_data.append(gb_row)
        x_labels.append(LABELS[problem])

# Convert to numpy arrays and normalize
av_data = np.array(av_data)
gb_data = np.array(gb_data)

# Normalize each row by its maximum value
av_data = av_data / np.nanmax(av_data, axis=1).reshape(-1, 1)
gb_data = gb_data / np.nanmax(gb_data, axis=1).reshape(-1, 1)

# Variant labels
variant_labels = ["NoV-NoC", "NoV-C", "V-NoC", "V-C"]

# Color palette
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def create_plot(data, title, filename, method_name):
    """Create a grouped bar plot"""
    # Seventeen names do not fit horizontally; they were overprinting each
    # other, so the axis is wider and the labels are turned.
    _fig, ax = plt.subplots(figsize=(16, 6.5))

    # Set up positions for bars
    n_problems = len(x_labels)
    n_variants = len(variant_labels)
    width = 0.8 / n_variants  # Make bars narrower to fit within the group
    x = np.arange(n_problems) * 1.2  # Decrease space between groups

    # Create bars for each variant
    for i, (variant, color) in enumerate(zip(variant_labels, colors, strict=False)):
        position = x - (n_variants / 2 - i) * width
        ax.bar(
            position,
            data[:, i],
            width * 0.8,  # Add a small gap between bars
            color=color,
            label=variant,
        )

    # Customize plot
    ax.set_ylabel("Scaled MSE")
    ax.set_title(title, pad=34)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlim(x[0] - 0.9, x[-1] + 0.9)
    ax.set_ylim(0, 1.04)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4,
              frameon=False)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f"plots/variants/{filename}", dpi=300, bbox_inches="tight")
    plt.show()


# Create AV plot
if len(av_data) > 0:
    create_plot(av_data, "Results of LESS-A Variants", "less_av_variants.png", "AV")

# Create GB plot
if len(gb_data) > 0:
    create_plot(gb_data, "Results of LESS-B Variants", "less_gb_variants.png", "GB")

print("Plots saved to plots/variants/ directory")
