"""Plots the incremental results (logs/incremental) to plots/incremental/."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.resultdirs import INCREMENTAL

# Set up plotting style
sns.set_theme(style='whitegrid', color_codes=True)
sns.set_style('ticks')

# Use the vibrant colors from plot_variants.py
bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# Configuration
RESULTS_DIR = INCREMENTAL
PLOTS_DIR = "plots/incremental"

# Problem names


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
    """Short axis label: eighteen names do not fit at full length."""
    key = name.replace("_GROUPED", "").lower()
    short = {"cpusmallscale": "cpusmall", "blogfeedback": "blogfeed",
             "airquality": "airqual"}
    return short.get(key, key)


PROBLEMS = _complete_problems(INCREMENTAL)

# X-axis label mapping for cleaner display. The -LXGB and -TXGB arms take the
# package default global estimator, which is a random forest: xgboost's native RF
# mode, 25 trees at learning_rate 1.0 with subsample 0.8 and colsample_bynode 0.8
# (less.py _build_native_global_xgboost_rf_base_params). The published script
# passed sklearn's RandomForestRegressor, so the arm means the same thing and the
# implementation differs -- RF is the right label.
XLABEL_MAPPING = {
    'AV-D': 'L-L',
    'AV-TL': 'DT-L',
    'AV-LXGB': 'L-RF',
    'AV-TXGB': 'DT-RF',
    'B-D': 'L-L',
    'B-TL': 'DT-L',
    'B-LXGB': 'L-RF',
    'B-TXGB': 'DT-RF'
}

def load_results_from_json(problem_name):
    """Load results from JSON file for a given problem"""
    json_path = os.path.join(RESULTS_DIR, f"{problem_name}.json")
    
    if not os.path.exists(json_path):
        return {}
    
    with open(json_path) as f:
        data = json.load(f)
    
    return data

def calculate_method_averages(results_dict):
    """Calculate average MSE for each method across all folds"""
    averages = {}
    
    for method, fold_results in results_dict.items():
        # Underscore keys are the runner's own bookkeeping, not methods.
        if method.startswith("_") or not isinstance(fold_results, dict):
            continue
        if fold_results:
            fold_values = [float(mse) for mse in fold_results.values()]
            averages[method] = np.mean(fold_values)
        else:
            averages[method] = np.nan
    
    return averages

def collect_all_results():
    """Collect results from all JSON files"""
    all_results = {}
    
    for problem in PROBLEMS:
        results = load_results_from_json(problem)
        if results:
            averages = calculate_method_averages(results)
            all_results[problem] = averages
    
    return all_results

def create_comparison_plot(all_results, methods_to_plot=None, scaling='row_max', 
                          save_path=None, plot_title=None):
    """Create comparison plot"""
    
    if not all_results:
        return
    
    # Determine which methods to plot
    if methods_to_plot is None:
        first_problem = next(iter(all_results.values()))
        available_methods = list(first_problem.keys())
        methods_to_plot = available_methods[:4]
    
    # Create data matrix
    problems_with_data = []
    data_matrix = []
    
    for problem, method_results in sorted(all_results.items()):
        row = []
        has_all_methods = True
        
        for method in methods_to_plot:
            if method in method_results and not np.isnan(method_results[method]):
                row.append(method_results[method])
            else:
                has_all_methods = False
                break
        
        if has_all_methods:
            problems_with_data.append(problem)
            data_matrix.append(row)
    
    if not data_matrix:
        return
    
    data_matrix = np.array(data_matrix)
    
    # Apply scaling
    if scaling == 'row_max':
        data_matrix = data_matrix / np.max(data_matrix, axis=1).reshape(-1, 1)
    elif scaling == 'global_max':
        data_matrix = data_matrix / np.max(data_matrix)
    
    # Create plot
    # eighteen groups do not fit the published 12-inch canvas
    fig, ax = plt.subplots(figsize=(max(10.0, 0.75 * len(problems_with_data) + 3), 5.5))
    
    n_methods = len(methods_to_plot)
    width = 0.8 / n_methods # Make bars narrower to fit within the group
    x = np.arange(len(problems_with_data)) * 1.2 # Decrease space between groups

    # Create bars with mapped labels
    for i, method in enumerate(methods_to_plot):
        xlabel = XLABEL_MAPPING.get(method, method)
        position = x - (n_methods / 2 - i) * width
        ax.bar(position, data_matrix[:, i], width * 0.8, # Add a small gap between bars
               color=bar_colors[i], label=xlabel)
    
    # Customize plot
    ax.set_ylabel('Scaled MSE' if scaling != 'none' else 'MSE')
    ax.set_xticks(x)
    ax.set_xticklabels([_label(p) for p in problems_with_data],
                       rotation=45, ha='right')
    
    # Set title
    if plot_title:
        ax.set_title(plot_title)
    else:
        ax.set_title('Effects of Changing Local and Global Estimators')
    
    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.42),
              fancybox=True, shadow=True, ncol=len(methods_to_plot))
    
    # Add grid
    plt.grid(True, 'major', 'y', ls='--', lw=0.5, c='k', alpha=0.3)
    
    fig.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig, ax

def main():
    """Main function to run the analysis"""
    all_results = collect_all_results()
    
    if not all_results:
        return
    
    # Create output directory for plots
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Plot 1: AV methods only
    av_methods = [method for method in all_results[next(iter(all_results.keys()))] 
                  if method.startswith('AV-')]
    if len(av_methods) > 1:
        av_save_path = os.path.join(PLOTS_DIR, "av_methods_comparison.png")
        create_comparison_plot(
            all_results, 
            av_methods, 
            scaling='row_max',
            save_path=av_save_path,
            plot_title='LESS-A Methods Comparison'
        )
    
    # Plot 2: B methods only  
    b_methods = [method for method in all_results[next(iter(all_results.keys()))] 
                 if method.startswith('B-')]
    if len(b_methods) > 1:
        b_save_path = os.path.join(PLOTS_DIR, "b_methods_comparison.png")
        create_comparison_plot(
            all_results, 
            b_methods, 
            scaling='row_max',
            save_path=b_save_path,
            plot_title='LESS-B Methods Comparison'
        )

if __name__ == "__main__":
    main()