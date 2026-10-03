"""Heatmap of method ranks per problem, from logs/results.

    PYTHONPATH=. python plot_funcs/plot.py
"""

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.problems import ALL_PROBLEMS
from src.resultdirs import MAIN

# Set academic style with single color
plt.style.use("seaborn-v0_8-whitegrid")
# Use single academic blue color
academic_color = "#2E4A62"
sns.set_palette([academic_color])

HIGH_DIM_PREFIXES = ("FRIEDMAN1_", "REGRESSION_")


def load_all_results(results_dir=MAIN):
    """Load all JSON result files from directory, excluding high-dimensional problems"""
    all_results = {}
    json_files = glob.glob(f"{results_dir}/*.json")

    for file_path in json_files:
        problem_name = Path(file_path).stem
        if problem_name.startswith(HIGH_DIM_PREFIXES):
            continue
        # _protocol.json sits in the results directory beside the problems and
        # is not one; loading it made every reader treat its scoring string as a
        # method's fold scores.
        if problem_name.startswith("_"):
            continue
        # query and road3d were dropped from the suite but their result files
        # stay on disk; globbing the directory put them back into the figure
        # while every table reported eighteen problems.
        if problem_name not in ALL_PROBLEMS:
            continue
        with open(file_path) as f:
            payload = json.load(f)
        # The results files carry bookkeeping beside the scores -- _metrics,
        # _skipped, _failed, _best_params -- and every reader here treats a
        # top-level key as a method name. Without this filter the plots grow a
        # "_metrics" algorithm whose "score" is the mean of a timing record.
        # A cell with fewer than four folds is a run in progress. Averaged in, it
        # ranks a method on a different test set from everyone else and nothing in
        # the figure says so.
        all_results[problem_name] = {
            k: v for k, v in payload.items()
            if not k.startswith("_")
            and not (isinstance(v, dict) and 0 < len(v) < 4)
        }

    return all_results


def calculate_statistics(results_dict):
    """Calculate mean and std for each algorithm across folds"""
    stats = {}
    for algorithm, folds in results_dict.items():
        values = list(folds.values())
        stats[algorithm] = {"mean": np.mean(values), "std": np.std(values), "values": values}
    return stats


def create_ranking_plot_with_averages(all_results):
    """Create ranking visualization with average rankings and sorted algorithms"""
    # Get all unique algorithms across all problems
    all_algorithms = set()
    for results in all_results.values():
        all_algorithms.update(results.keys())

    problems = sorted(all_results.keys())

    # Create ranking matrix and score matrix with NaN for missing algorithms
    ranking_matrix = np.full((len(all_algorithms), len(problems)), np.nan)
    score_matrix = np.full((len(all_algorithms), len(problems)), np.nan)
    algorithm_list = list(all_algorithms)

    for j, problem in enumerate(problems):
        if problem in all_results:
            problem_results = all_results[problem]
            # Get algorithms that exist for this problem
            available_algs = list(problem_results.keys())

            if available_algs:  # If problem has results
                # Calculate means for available algorithms
                means = []
                alg_indices = []
                for alg in available_algs:
                    if alg in problem_results:
                        folds = problem_results[alg]
                        mean_score = np.mean(list(folds.values()))
                        means.append(mean_score)
                        alg_indices.append(algorithm_list.index(alg))

                # Calculate ranks (1 = best, lower MSE is better)
                ranks = np.argsort(np.argsort(means)) + 1

                # Fill matrices
                for idx, (rank, score) in zip(
                    alg_indices, zip(ranks, means, strict=False), strict=False
                ):
                    ranking_matrix[idx, j] = rank
                    score_matrix[idx, j] = score

    # Calculate average rankings for sorting. The coverage count travels with the
    # mean because the two are not comparable without it: a method that ran on
    # four problems and one that ran on sixteen get averages on different bases,
    # and a rank column that hides this reads as if they were the same quantity.
    avg_rankings = []
    coverage = []
    for i in range(len(algorithm_list)):
        valid_ranks = ranking_matrix[i, ~np.isnan(ranking_matrix[i, :])]
        if len(valid_ranks) > 0:
            avg_rankings.append(np.mean(valid_ranks))
            coverage.append(len(valid_ranks))
        else:
            avg_rankings.append(np.inf)  # Put algorithms with no data at the end
            coverage.append(0)

    # Sort algorithms by average ranking. If ranks are equal, prioritize algorithms
    # with 'less' in their name.
    sorted_indices = sorted(
        range(len(algorithm_list)),
        key=lambda i: (avg_rankings[i], 0 if "less" in algorithm_list[i].lower() else 1),
    )
    algorithms = [algorithm_list[i] for i in sorted_indices]
    ranking_matrix = ranking_matrix[sorted_indices, :]
    score_matrix = score_matrix[sorted_indices, :]
    avg_rankings = [avg_rankings[i] for i in sorted_indices]
    coverage = [coverage[i] for i in sorted_indices]

    return algorithms, problems, ranking_matrix, score_matrix, avg_rankings, coverage


def plot_rankings(algorithms, problems, ranking_matrix, avg_rankings,
                  coverage=None, title_suffix="Rankings", source_note=None):
    """Create ranking heatmap with academic color scheme"""
    # Consistent figure size calculation
    fig_width = max(14, len(problems) * 0.7 + 2)  # +2 for avg rank column
    fig_height = max(8, len(algorithms) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Normalize rankings per problem for consistent color mapping
    normalized_ranking_matrix = np.full_like(ranking_matrix, np.nan)

    for j in range(ranking_matrix.shape[1]):  # For each problem (column)
        col_data = ranking_matrix[:, j]
        valid_mask = ~np.isnan(col_data)

        if np.sum(valid_mask) > 1:  # If we have at least 2 valid values
            valid_ranks = col_data[valid_mask]
            min_rank = np.min(valid_ranks)
            max_rank = np.max(valid_ranks)

            if max_rank > min_rank:  # Avoid division by zero
                # Normalize: 0 = best (rank 1), 1 = worst (highest rank)
                normalized_col = (col_data - min_rank) / (max_rank - min_rank)
                normalized_ranking_matrix[:, j] = normalized_col
            else:
                # All ranks are the same, set to best value
                normalized_ranking_matrix[valid_mask, j] = 0.0
        elif np.sum(valid_mask) == 1:
            # Only one valid rank, set to best value
            normalized_ranking_matrix[valid_mask, j] = 0.0

    # Academic blue colormap - dark to light blue (best = dark, worst = light)
    cmap = plt.cm.Blues_r

    # Create heatmap with normalized data
    im = ax.imshow(
        normalized_ranking_matrix, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0, vmax=1
    )

    # Fix x-axis labels - Set ticks and labels properly
    ax.set_xticks(range(len(problems)))
    ax.set_xticklabels(problems, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(range(len(algorithms)))
    ax.set_yticklabels(algorithms, fontsize=12)
    ax.set_xlabel("Problem", fontsize=12)
    ax.set_ylabel("Algorithm", fontsize=12)

    # Add title. A figure drawn from a run that was not repeated for the
    # revision has to name that run (src/resultdirs.py), so source_note goes
    # under the title instead of leaving it indistinguishable from the main one.
    ax.set_title(
        f"Algorithm {title_suffix} Across All Problems",
        fontweight="bold",
        fontsize=16,
        pad=34 if source_note else 20,
    )
    if source_note:
        ax.text(
            0.5,
            1.01,
            source_note,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            color="dimgray",
        )

    # Add text annotations for original rankings
    for i in range(len(algorithms)):
        for j in range(len(problems)):
            original_value = ranking_matrix[i, j]
            normalized_value = normalized_ranking_matrix[i, j]

            if not np.isnan(original_value):
                # Choose text color based on background brightness
                text_color = "white" if normalized_value < 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{int(original_value)}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                    fontsize=11,
                )
            else:
                # Mark missing data
                ax.text(
                    j,
                    i,
                    "—",
                    ha="center",
                    va="center",
                    color="gray",
                    fontweight="bold",
                    fontsize=12,
                )

    # Add average ranking column
    avg_col_x = len(problems)

    # Create average ranking labels
    for i, avg_rank in enumerate(avg_rankings):
        if not np.isinf(avg_rank):
            ax.text(
                avg_col_x + 0.3,
                i,
                f"{avg_rank:.2f}",
                ha="left",
                va="center",
                color="black",
                fontweight="bold",
                fontsize=11,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgray", "alpha": 0.7},
            )
            if coverage is not None:
                ax.text(
                    avg_col_x + 1.45,
                    i,
                    f"({coverage[i]})",
                    ha="left",
                    va="center",
                    color="gray",
                    fontsize=9,
                )
        else:
            ax.text(
                avg_col_x + 0.3,
                i,
                "N/A",
                ha="left",
                va="center",
                color="gray",
                fontweight="bold",
                fontsize=11,
            )

    # Add "Avg Rank" label
    ax.text(
        avg_col_x + 0.3,
        -1,
        "Avg Rank",
        ha="left",
        va="center",
        color="black",
        fontweight="bold",
        fontsize=12,
    )

    # Extend x-axis to accommodate average ranking
    ax.set_xlim(-0.5, len(problems) + 1)

    # Add colorbar with proper rank labels
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Rank", rotation=270, labelpad=15, fontsize=11)
    # Set colorbar ticks to show actual rank values (1 to number of algorithms)
    num_algs_in_problems = []
    for j in range(ranking_matrix.shape[1]):
        valid_ranks = ranking_matrix[~np.isnan(ranking_matrix[:, j]), j]
        if len(valid_ranks) > 0:
            num_algs_in_problems.append(len(valid_ranks))

    if num_algs_in_problems:
        max_algs = max(num_algs_in_problems)
        if max_algs > 1:
            # Create tick positions and labels
            tick_positions = np.linspace(0, 1, max_algs)
            tick_labels = [str(i) for i in range(1, max_algs + 1)]
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_labels)

    # Remove all grid lines completely
    ax.grid(False)
    ax.set_axisbelow(False)

    plt.tight_layout()
    return fig


def plot_scores(algorithms, problems, score_matrix, avg_rankings, title_suffix="Scores"):
    """Create score heatmap with ranking-based coloring"""
    # Consistent figure size calculation (same as rankings)
    fig_width = max(14, len(problems) * 0.8)  # No avg rank column for scores
    fig_height = max(8, len(algorithms) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Convert scores to rankings for color mapping
    ranking_for_colors = np.full_like(score_matrix, np.nan)

    for j in range(score_matrix.shape[1]):  # For each problem (column)
        col_data = score_matrix[:, j]
        valid_mask = ~np.isnan(col_data)

        if np.sum(valid_mask) > 0:  # If we have valid values
            valid_scores = col_data[valid_mask]
            # Convert scores to ranks (lower score = better rank)
            ranks = np.argsort(np.argsort(valid_scores)) + 1

            # Fill ranking matrix for coloring
            valid_indices = np.where(valid_mask)[0]
            for idx, rank in zip(valid_indices, ranks, strict=False):
                ranking_for_colors[idx, j] = rank

    # Normalize rankings per problem for consistent color mapping (same as ranking plot)
    normalized_ranking_matrix = np.full_like(ranking_for_colors, np.nan)

    for j in range(ranking_for_colors.shape[1]):  # For each problem (column)
        col_data = ranking_for_colors[:, j]
        valid_mask = ~np.isnan(col_data)

        if np.sum(valid_mask) > 1:  # If we have at least 2 valid values
            valid_ranks = col_data[valid_mask]
            min_rank = np.min(valid_ranks)
            max_rank = np.max(valid_ranks)

            if max_rank > min_rank:  # Avoid division by zero
                # Normalize: 0 = best (rank 1), 1 = worst (highest rank)
                normalized_col = (col_data - min_rank) / (max_rank - min_rank)
                normalized_ranking_matrix[:, j] = normalized_col
            else:
                # All ranks are the same, set to best value
                normalized_ranking_matrix[valid_mask, j] = 0.0
        elif np.sum(valid_mask) == 1:
            # Only one valid rank, set to best value
            normalized_ranking_matrix[valid_mask, j] = 0.0

    # Academic blue colormap - dark to light blue (best = dark, worst = light)
    cmap = plt.cm.Blues_r

    # Create heatmap with normalized ranking data
    im = ax.imshow(
        normalized_ranking_matrix, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0, vmax=1
    )

    # Customize axes
    ax.set_xticks(range(len(problems)))
    ax.set_xticklabels(problems, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(algorithms)))
    ax.set_yticklabels(algorithms, fontsize=10)
    ax.set_xlabel("Problem", fontsize=10)
    ax.set_ylabel("Algorithm", fontsize=10)

    # Add title
    ax.set_title(
        f"Algorithm {title_suffix} Across All Problems", fontweight="bold", fontsize=14, pad=20
    )

    # Add text annotations for original scores with adaptive text color
    for i in range(len(algorithms)):
        for j in range(len(problems)):
            original_score = score_matrix[i, j]
            normalized_value = normalized_ranking_matrix[i, j]

            if not np.isnan(original_score):
                # Choose text color based on background brightness
                text_color = "white" if normalized_value < 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{original_score:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                    fontsize=8,
                )
            else:
                # Mark missing data
                ax.text(
                    j,
                    i,
                    "—",
                    ha="center",
                    va="center",
                    color="gray",
                    fontweight="bold",
                    fontsize=12,
                )

    # No average ranking column for scores plot
    ax.set_xlim(-0.5, len(problems) - 0.5)

    # Add colorbar with proper rank labels
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Rank", rotation=270, labelpad=15, fontsize=11)
    # Set colorbar ticks to show actual rank values (1 to number of algorithms)
    num_algs_in_problems = []
    for j in range(score_matrix.shape[1]):
        valid_scores = score_matrix[~np.isnan(score_matrix[:, j]), j]
        if len(valid_scores) > 0:
            num_algs_in_problems.append(len(valid_scores))

    if num_algs_in_problems:
        max_algs = max(num_algs_in_problems)
        if max_algs > 1:
            # Create tick positions and labels
            tick_positions = np.linspace(0, 1, max_algs)
            tick_labels = [str(i) for i in range(1, max_algs + 1)]
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_labels)

    # Remove all grid lines completely
    ax.grid(False)
    ax.set_axisbelow(False)

    plt.tight_layout()
    return fig


def plot_problem_specific_results(all_results):
    """Create individual plots for each problem showing scores with error bars"""
    problem_names = sorted(all_results.keys())

    for problem_name in problem_names:
        fig, ax = plt.subplots(figsize=(10, 6))
        problem_results = all_results[problem_name]

        # Calculate statistics for this problem
        stats = calculate_statistics(problem_results)

        # Sort algorithms by mean score (best to worst)
        sorted_algs = sorted(stats.items(), key=lambda x: x[1]["mean"])

        algorithms = [alg for alg, _ in sorted_algs]
        means = [stat["mean"] for _, stat in sorted_algs]
        stds = [stat["std"] for _, stat in sorted_algs]

        # Create color gradient - best (dark blue) to worst (light blue)
        colors = [
            plt.cm.Blues_r(0.2 + 0.6 * i / max(1, len(algorithms) - 1))
            for i in range(len(algorithms))
        ]

        # Create bar plot with error bars - gradient colors
        ax.bar(
            range(len(algorithms)),
            means,
            yerr=stds,
            capsize=5,
            color=colors,
            edgecolor="navy",
            linewidth=0.5,
            alpha=0.9,
        )

        # Customize the plot
        ax.set_title(f"{problem_name}", fontweight="bold", fontsize=14)
        ax.set_xlabel("Algorithm", fontsize=12)
        ax.set_ylabel("Problem Score (MSE)", fontsize=12)
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=45, ha="right", fontsize=10)

        # Add value labels on bars with better positioning and spacing
        for i, (mean, std) in enumerate(zip(means, stds, strict=False)):
            # Position text above error bars with more spacing
            y_pos = mean + std + (max([m + s for m, s in zip(means, stds, strict=False)]) * 0.08)
            # Two-line format: value on top, ±std below with better line spacing
            text_label = f"{mean:.3f}\n±{std:.3f}"
            ax.text(
                i,
                y_pos,
                text_label,
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
                linespacing=1.2,
            )  # Increased line spacing from 0.8 to 1.2

        # Set y-axis with better spacing
        y_min = max(0, min(means) - max(stds) * 0.1)
        y_max = max([m + s for m, s in zip(means, stds, strict=False)]) * 1.3
        ax.set_ylim(y_min, y_max)

        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)

        plt.tight_layout()

        # Save individual plot
        fig.savefig(f"problem_{problem_name}_results.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()


# Main execution
if __name__ == "__main__":
    all_results = load_all_results()

    print(f"Loaded {len(all_results)} problems: {list(all_results.keys())}")

    # Create data matrices with sorted algorithms
    algorithms, problems, ranking_matrix, score_matrix, avg_rankings, coverage = (
        create_ranking_plot_with_averages(all_results)
    )

    # Create ranking visualization
    print("Creating ranking heatmap...")
    fig_rankings = plot_rankings(algorithms, problems, ranking_matrix, avg_rankings,
                             coverage, "Rankings")
    fig_rankings.savefig("plots/main/algorithm_rankings.png", dpi=300, bbox_inches="tight")

    # # Create scores visualization
    # print("Creating scores heatmap...")
    # fig_scores = plot_scores(algorithms, problems, score_matrix, avg_rankings, "Scores")
    # fig_scores.savefig('algorithm_scores.png', dpi=300, bbox_inches='tight')

    # # Create problem-specific plots<
    # print("Creating problem-specific plots...")
    # plot_problem_specific_results(all_results)

    # plt.show()
