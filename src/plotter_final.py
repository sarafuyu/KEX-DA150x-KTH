#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Plotting

This script is run separately from the main script to generate plots.

:Date: 2024-05-03
:Authors: Sara Rydell, Noah Hopkins

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Imports

# Standard library imports
import warnings
import pickle
from numbers import Real
from pathlib import Path
from collections.abc import Sequence
from copy import deepcopy, copy

# External imports
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For heatmap or advanced plotting
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator, SymmetricalLogLocator, FuncFormatter
from joypy import joyplot
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score

# Local imports
import utils  # The utils module need to be imported first to set verbosity level and random seed
import features


# %% Setup

SEED = utils.RANDOM_SEED  # get random seed
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_SETS = {
    'final-results': {
        'results_directory': 'final-results',
        'cv_results_':       '2024-06-08-014208__GridSearchCV.pkl',
    }
}


# %% Configuration

FEATURE_SELECTION_METRIC = 'final-results'  # 'accuracy', 'roc_auc', 'f1'
GRIDSEARCH_METRIC = 'roc_auc'  # 'accuracy', 'roc_auc', 'f1'  # noqa
PLOT_METRIC: str = 'roc_auc'

CV_RESULTS_DIR = PROJECT_ROOT/'data'/'results'/RESULTS_SETS[FEATURE_SELECTION_METRIC]['results_directory']  # noqa
GRID_SEARCH_RESULTS_PATH: Path = CV_RESULTS_DIR/RESULTS_SETS[FEATURE_SELECTION_METRIC]['cv_results_']

# Specify parameters
PARAMS_TO_GET_BEST = ['C', 'degree', 'coef0', 'kernel', 'gamma', 'class_weight', 'tol']
PARAMS_OF_INTEREST = ['C', 'degree', 'coef0']
FIXED_KERNEL = 'poly'  # 'kernel'
PARAM_PREFIX = 'param_'  # Prefix for the parameter columns in the cv_results_ DataFrame
SVC_DEFAULT_PARAMS = {'param_C': 1.0, 'param_degree': 3, 'param_coef0': 0.0, 'param_gamma': 'scale', 'param_class_weight': 'None', 'param_tol': 0.001}
SVC_DEFAULT_KERNEL = 'rbf'
GAMMA_PARAMS = ['scale', 0.01, 0.1, 1.0, 10.0]

# # Load the cross-validation results
gridsearch_cv = joblib.load(GRID_SEARCH_RESULTS_PATH)
cv_results = pd.DataFrame(gridsearch_cv.cv_results_)

# Load the data
X_training = pd.read_csv(CV_RESULTS_DIR / '2024-06-08-014208__GridSearch_X_training.csv').drop(columns='Unnamed: 0')
X_testing = pd.read_csv(CV_RESULTS_DIR / '2024-06-08-014208__GridSearch_X_testing.csv').drop(columns='Unnamed: 0')
y_training = pd.read_csv(CV_RESULTS_DIR / '2024-06-08-014208__GridSearch_y_training.csv').drop(columns='Unnamed: 0')
y_testing = pd.read_csv(CV_RESULTS_DIR / '2024-06-08-014208__GridSearch_y_testing.csv').drop(columns='Unnamed: 0')

# Make all param prefixes the same
cv_results = utils.replace_column_prefix(cv_results, ['param_classifier__'],  'param_')

# Add prefix to the parameter names
PARAMS_OF_INTEREST = [f'{PARAM_PREFIX}{param}' for param in PARAMS_OF_INTEREST]
PARAMS_TO_GET_BEST = [f'{PARAM_PREFIX}{param}' for param in PARAMS_TO_GET_BEST]
metrics_label = f'FS-{FEATURE_SELECTION_METRIC}_GS-{GRIDSEARCH_METRIC}_PLOT-{PLOT_METRIC}'

# Sanitize the normalizer repr
cv_results['param_normalizer'] = cv_results['param_normalizer'].astype(str).str.split('(').str[0]

# Set dtype for the parameters of interest
cv_results['param_kernel'] = cv_results['param_kernel'].astype(str)
cv_results['param_normalizer'] = cv_results['param_normalizer'].astype(str)
cv_results['param_C'] = cv_results['param_C'].astype(np.float64)
cv_results['param_degree'] = cv_results['param_degree'].astype(int)
cv_results['param_coef0'] = cv_results['param_coef0'].astype(np.float64)
cv_results['param_tol'] = cv_results['param_tol'].astype(np.float64)
cv_results['param_class_weight'] = cv_results['param_class_weight'].astype(str)

# Permanently remove all rows with tol, class weights and normalizer not equal to CHOSEN_TOL, CHOSEN_CLASS_WEIGHT, CHOSEN_NORMALIZER
CHOSEN_TOL = 0.001
CHOSEN_CLASS_WEIGHT = 'balanced'
CHOSEN_NORMALIZER = 'StandardScaler'
CHOSEN_KERNEL = 'poly'
cv_results = cv_results[
    (cv_results['param_tol'] == CHOSEN_TOL) &
    (cv_results['param_class_weight'] == CHOSEN_CLASS_WEIGHT) &
    (cv_results['param_normalizer'] == CHOSEN_NORMALIZER) &
    (cv_results['param_kernel'] == CHOSEN_KERNEL) &
    (cv_results['param_gamma'] != 0.0)
].copy()

# Find the best parameters and break ties
alpha = 0  # 0.1
beta = 0   # 0
gamma = 0  # 1

# Calculate the gamma value for the rbf and sigmoid kernels
# gamma = 1 / (n_features * X.var()) if gamma == 'scale'
# gamma = 1 / n_features if gamma == 'auto'
cv_results['gamma_float'] = np.float64(np.nan)
cv_results['gamma_float'].astype(np.float64)
X = deepcopy(X_training)
y = deepcopy(y_training['FT5'])
best_estimator = deepcopy(gridsearch_cv.best_estimator_)
X, y = best_estimator.named_steps.classifier._validate_data(  # noqa
    X,
    y,
    dtype=np.float64,
    order="C",
    accept_sparse="csr",
    accept_large_sparse=False,
)
sparse = False
for i, row in cv_results.iterrows():
    if isinstance(row['param_gamma'], str):
        if row['param_gamma'] == "scale":
            # var = E[X^2] - E[X]^2 if sparse
            X_var = (X.multiply(X)).mean() - (X.mean()) ** 2 if sparse else X.var()
            cv_results.loc[i, 'gamma_float'] = 1.0 / (X.shape[1] * X_var) if X_var != 0 else 1.0
        elif row['param_gamma'] == "auto":
            cv_results.loc[i, 'gamma_float'] = 1.0 / X.shape[1]
    elif isinstance(row['param_gamma'], Real):
        cv_results.loc[i, 'gamma_float'] = row['param_gamma']
print(f"Number of NaN values in gamma_float: {cv_results['gamma_float'].isna().sum()}")
print(
    f"Number of unique values in gamma_float (should be 4 if 'scale'='auto'): {cv_results['gamma_float'].nunique()}"
    )

all_best_params = utils.get_best_params(cv_results, GRIDSEARCH_METRIC, PARAMS_TO_GET_BEST, return_all=True, alpha=alpha, beta=beta, default_params=False)
if all_best_params.shape[0] > 1:
    # Find and print difference in the rows
    print("Multiple best parameters found...")
    print("Difference among the best cv_results rows:")
    utils.print_differences(all_best_params)

    # Break ties by using the default parameters
    print("Breaking ties with default parameters...")
    all_best_params = utils.get_best_params(
        cv_results, GRIDSEARCH_METRIC, PARAMS_TO_GET_BEST, return_all=True, alpha=alpha, beta=beta,
        default_params=SVC_DEFAULT_PARAMS
        )
    if all_best_params.shape[0] > 1:
        Warning(
            "Multiple best parameters found after breaking ties with defaults... Please check the results manually."
            )
        breakpoint()
        all_best_params = all_best_params.iloc[0, :]
best_params_series = all_best_params if type(all_best_params) is pd.Series else all_best_params.iloc[0]
best_params = best_params_series.to_dict() if type(best_params_series) is pd.Series else best_params_series

# Penalize (modify) scores in cv_results_copy according to:
#   Adjusted Score = Score − α×ln(degree) − β×ln(std(Score)+1) − γ×ln(gamma+1)
#                      (α only if kernel is poly)      (γ only if kernel is rbf or sigmoid)
if alpha > 0 or beta > 0 or gamma > 0:
    # Create a mask for the polynomial kernel
    poly_kernel_mask = cv_results['param_kernel'] == 'poly'
    rbf_sigmoid_kernel_mask = ~poly_kernel_mask
    # Adjust the score based on the conditions
    cv_results[f'mean_test_{GRIDSEARCH_METRIC}'] = (
            cv_results[f'mean_test_{GRIDSEARCH_METRIC}']
            - alpha * (np.log(cv_results['param_degree']) * poly_kernel_mask)
            - beta * (np.log(cv_results[f'mean_test_{GRIDSEARCH_METRIC}'] + 1))
            - gamma * (np.log(cv_results['gamma_float'] + 1) * rbf_sigmoid_kernel_mask)
    )

best_row = utils.get_row_from_best_params(cv_results, best_params, CHOSEN_TOL, CHOSEN_CLASS_WEIGHT, GRIDSEARCH_METRIC)
best_row = utils.replace_column_prefix(best_row, ['classifier__'], 'param_')


# %% Plot functions

def plot_parameter_effects_stacked(cv_results_, parameters, metrics, fixing_mode='optimal', default_params=None):
    """
    Plot the effects of varying parameters on model performance.

    :param cv_results_: (pd.DataFrame) The DataFrame containing the cross-validation results.
    :param parameters: (list[str]) List of parameter names to plot.
    :param metrics: (list[str]) List of metrics to plot.
    :param fixing_mode: (str) Fixing mode: 'default', 'optimal', or 'both'.
    :param default_params: (dict) The default parameters for the model. If None, comparison is made only
        against optimal parameters.
    """
    from copy import deepcopy
    from matplotlib.ticker import MaxNLocator, SymmetricalLogLocator

    global GRIDSEARCH_METRIC, alpha, beta

    # Determine the fixing modes
    if fixing_mode not in ['default', 'optimal', 'both']:
        raise ValueError("fixing_mode must be 'default', 'optimal', or 'both'")

    last_used_color = None

    for param in parameters:
        # if param != 'param_degree':
        #     continue
        fixed_params_ = [p for p in parameters if p != param]

        fixing_modes = [fixing_mode] if fixing_mode != 'both' else ['default', 'optimal']

        fig, ax = plt.subplots(figsize=(8, 5))

        best_params_ = None
        for idx, (metric, mode) in enumerate([(metric, mode) for metric in metrics for mode in fixing_modes]):
            best_params_ = utils.get_best_params(cv_results_, GRIDSEARCH_METRIC, parameters, alpha=alpha, beta=beta) if mode == 'optimal' else pd.Series(default_params)
            mask = pd.DataFrame(cv_results_[fixed_params_] == best_params_.drop(index=param)).all(axis=1)
            varying_param_data = deepcopy(cv_results_[mask])

            x = varying_param_data[param]
            y = varying_param_data[f'mean_test_{metric}']
            std = varying_param_data[f'std_test_{metric}']
            se = std / np.sqrt(len(cv_results_) * 0.8)  # Assuming 80% train split

            if mode == 'default':
                marker = 'o'
                linestyle = '--'
            else:
                marker = 'D'
                linestyle = '-'
            if last_used_color is not None and fixing_mode == 'both' and mode == 'optimal' and len(metrics)*len(fixing_modes) > 2:
                color = last_used_color
                # if param == 'param_degree':
                #     p = ax.errorbar(
                #         x, y, yerr=se, fmt=marker, color=color, capsize=5, elinewidth=1, markeredgewidth=1,
                #         label=f'{metric} ({mode})'
                #     )
                #     ax.set_xticks(x)
                #     ax.set_xticklabels(x, rotation=0)
                # else:
                p = ax.plot(
                    x, y, linestyle=linestyle, marker=marker, color=color, label=f'{metric} ({mode})'
                    )
                ax.fill_between(
                    x, y - se, y + se, alpha=0.2,
                    label=f'{f'SE ({metric}, {mode})' if len(metrics) * len(fixing_modes) < 3 else ''}'
                )
            else:
                # if param == 'param_degree':
                #     p = ax.errorbar(
                #         x, y, yerr=se, fmt=marker, capsize=5, elinewidth=1, markeredgewidth=1,
                #         label=f'{metric} ({mode})'
                #     )
                #     ax.set_xticks(x)
                #     ax.set_xticklabels(x, rotation=0)
                # else:
                p = ax.plot(
                    x, y, linestyle=linestyle, marker=marker, label=f'{metric} ({mode})'
                    )
                ax.fill_between(
                    x, y - se, y + se, alpha=0.2,
                    label=f'{f'SE ({metric}, {mode})' if len(metrics) * len(fixing_modes) < 1 else ''}'
                )
            try:
                last_used_color = p[-1].get_color()
            except AttributeError:
                last_used_color = p[-1][-1].get_color()

        param_label = param.split('_')[1]
        test_metric_label = ', '.join([metric.replace('_', ' ').title() for metric in metrics])
        fixed_param_label = ', '.join([f'{prm.split("_")[1]}={utils.format_number(best_params_[prm] if prm != 'param_degree' else int(best_params_[prm]), latex=True)}' for prm in fixed_params_])
        defauls_param_label = ', '.join([f'{prm.split("_")[1]}={utils.format_number(default_params[prm] if prm != 'param_degree' else int(default_params[prm]), latex=True)}' for prm in fixed_params_])
        title = (
            f'Effect of {param.split("_")[1]} on Model Performance ({test_metric_label})\n'
            f' {f"for Default ({defauls_param_label}) and Optimal Parameters ({fixed_param_label})" if fixing_mode == "both" else f"Using Default Parameters ({fixed_param_label})" if mode == "default" else "Using Optimal Parameters ({fixed_param_label})"}'
        )

        ax.set_xlabel(param_label)
        ax.set_ylabel(f'Mean CV Validation {test_metric_label}')
        ax.set_title(title)

        if param != 'param_degree':
            if param == 'param_coef0':
                linthresh = 1e-2  # np.abs(x[x > 0].min())
                ax.set_xscale('symlog', linthresh=linthresh)
                ax.xaxis.set_major_locator(SymmetricalLogLocator(base=10, linthresh=linthresh))
            else:
                ax.set_xscale('log')
        else:
            ax.set_xscale('linear')
            ax.set_xticks(np.arange(1, 13))


        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.legend()

        plt.tight_layout()
        plt.show()
        fixed_params_text = '_'.join(
                [f'{prm.split("_")[1]}-{utils.format_number(best_params_[prm], latex=False)}' for prm in fixed_params_]
            )
        fig.savefig(
            CV_RESULTS_DIR / f'stacked-effect-param-{param_label}_{test_metric_label.replace(', ', '-').replace(' ', '-')}_{fixing_mode}_fixed-{fixed_params_text}.png',
            dpi=300, transparent=True, bbox_inches='tight'
            )


# TODO: DONE flyttat adjust_color till utils

def plot_parameter_effects_stacked_bars(cv_results_, parameters, metrics, fixing_mode='optimal', default_params=None):
    """
    Plot the effects of varying parameters on model performance.

    :param cv_results_: (pd.DataFrame) The DataFrame containing the cross-validation results.
    :param parameters: (list[str]) List of parameter names to plot.
    :param metrics: (list[str]) List of metrics to plot.
    :param fixing_mode: (str) Fixing mode: 'default', 'optimal', or 'both'.
    :param default_params: (dict) The default parameters for the model. If None, comparison is made only
        against optimal parameters.
    """
    from copy import deepcopy
    from matplotlib.ticker import MaxNLocator

    global GRIDSEARCH_METRIC, alpha, beta

    # Determine the fixing modes
    if fixing_mode not in ['default', 'optimal', 'both']:
        raise ValueError("fixing_mode must be 'default', 'optimal', or 'both'")

    param = 'param_degree'
    fixed_params_ = [p for p in parameters if p != param]
    fixing_modes = [fixing_mode] if fixing_mode != 'both' else ['default', 'optimal']

    fig, ax = plt.subplots(figsize=(8, 5))

    bar_width = 0.95
    bar_charts = []
    all_stacks = []

    base_colors = plt.cm.get_cmap('tab10', len(metrics))  # noqa
    custom_colors = {
        'f1': '#5D3FD3',  # Purple
        'precision': '#ff7f0e',  # Orange
        'recall': '#1f77b4',  # Blue
    }
    color_mapping = {}
    for i, metric in enumerate(metrics):
        diff = 0.11
        lightness = 0.05
        if fixing_mode != 'both':
            diff = 0.1
            lightness = 0.0
        base_color = utils.adjust_color(base_colors(i), diff+lightness)
        if metric in custom_colors:
            base_color = utils.adjust_color(custom_colors[metric], diff)
        color_mapping[(metric, 'default')] = utils.adjust_color(base_color, -diff)  # Slightly darker
        color_mapping[(metric, 'optimal')] = utils.adjust_color(base_color, diff)  # Slightly lighter

    if fixing_mode == 'both':
        hatch_mapping = {'default': '//', 'optimal': '\\\\'}  # Different hatch patterns
    else:
        hatch_mapping = {'default': '//', 'optimal': '\\\\'}

    best_params_ = None
    for idx, (metric, mode) in enumerate([(metric, mode) for metric in metrics for mode in fixing_modes]):
        best_params_ = utils.get_best_params(cv_results_, GRIDSEARCH_METRIC, parameters, alpha=alpha, beta=beta) if mode == 'optimal' else pd.Series(
            default_params
            )
        mask = pd.DataFrame(cv_results_[fixed_params_] == copy(best_params_).drop(index=param)).all(axis=1)
        varying_param_data = deepcopy(cv_results_[mask])

        x = varying_param_data[param]
        y = varying_param_data[f'mean_test_{metric}']
        std = varying_param_data[f'std_test_{metric}']
        se = std / np.sqrt(len(cv_results_) * 0.8)  # Assuming 80% train split

        # Collect the stacks
        for xi, yi, sei in zip(x, y, se):
            all_stacks.append((xi, yi, sei, metric, mode))

    # Sort the stacks for each x value
    sorted_stacks = {}
    for xi, yi, sei, metric, mode in all_stacks:
        if xi not in sorted_stacks:
            sorted_stacks[xi] = []
        sorted_stacks[xi].append((yi, sei, metric, mode))

    # Sort each list in sorted_stacks
    for xi in sorted_stacks:
        sorted_stacks[xi].sort(reverse=True, key=lambda item: item[0])

    # Plot the sorted stacks
    max_y = 0
    labels_added = set()
    for xi in sorted(sorted_stacks.keys()):
        y_offset = 0
        for yi, sei, metric, mode in sorted_stacks[xi]:
            plt.rcParams['hatch.color'] = utils.adjust_color(color_mapping[(metric, mode)], -0.2)
            label = f'{metric} ({mode})' if (metric, mode) not in labels_added else ""
            b = ax.bar(
                xi, yi, bar_width, bottom=y_offset, yerr=sei, capsize=10,
                label=label, color=color_mapping[(metric, mode)], hatch=hatch_mapping[mode]
            )
            y_offset += yi
            max_y = max(max_y, y_offset)
            bar_charts.append(b)
            labels_added.add((metric, mode))

    param_label = param.split('_')[1]
    test_metric_label = ', '.join([metric.replace('_', ' ').title() for metric in metrics])
    fixed_param_label = ', '.join(
        [
            f'{prm.split("_")[1]}={utils.format_number(best_params_[prm] if prm != "param_degree" else int(best_params_[prm]), latex=True)}'
            for prm in fixed_params_]
        )
    title = (
        f'Effect of {param.split("_")[1]} on Model Performance ({test_metric_label})\n'
        f' {"for Default and Optimal" if fixing_mode == "both" else "Using Default" if mode == "default" else "Using Optimal"} Parameters ({fixed_param_label})'
    )

    # Create a sorted legend
    handles, labels = ax.get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: (x[1].split()[0], x[1].split()[-1]))
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)
    legend = ax.legend(sorted_handles, sorted_labels, loc='upper left')
    if not legend:
        ax.legend(sorted_handles, sorted_labels, loc='upper right')

    # Adjust the y-axis limit to prevent collision with the legend
    ax.set_ylim(0, max_y * 1.2)

    ax.set_xlabel(param_label)
    ax.set_ylabel(f'Mean CV Validation {test_metric_label}')
    ax.set_title(title)
    ax.set_xscale('linear')

    ax.set_xticks(np.arange(1, 13))
    ax.set_yticks(0, 0.2, 0.4, 0.6, 0.8, 0.1)
    # ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


    plt.tight_layout()
    plt.show()
    fixed_params_text = '_'.join(
        [f'{prm.split("_")[1]}-{utils.format_number(best_params_[prm], latex=False)}' for prm in fixed_params_]
    )
    fig.savefig(
        CV_RESULTS_DIR / f'stacked-effect-param-{param_label}_{test_metric_label.replace(", ", "-").replace(" ", "-")}_{fixing_mode}_fixed-{fixed_params_text}.png',
        dpi=300, transparent=True, bbox_inches='tight'
    )


def plot_parameter_effects_opt(cv_results_, parameters, test_metric=None, default_params=None):
    """
    Plot the effects of varying parameters on model performance.

    :param cv_results_: (pd.DataFrame) The DataFrame containing the cross-validation results.
    :param parameters: (list[str]) List of parameter names to plot.
    :param test_metric: (str) The name of the test score metric.
        Either 'mean_test_score' or 'final_accuracy'.
    :param default_params: (dict) The default parameters for the model. If None, comparison is made
        against optimal parameters.
    """
    global best_row, metrics_label
    if test_metric is None:
        raise ValueError("test_metric must be specified")
    if default_params is None:
        # Find the best parameters
        # best_index = cv_results_['rank_test_score'].idxmin()
        # best_param_vals = cv_results_.loc[best_index, parameters]
        best_param_vals = pd.Series({'param_C': best_row['param_C'], 'param_degree': best_row['param_degree'], 'param_coef0': best_row['param_coef0']})
    else:
        best_param_vals = pd.Series(default_params)

    # Loop over all unique kernels, create plots for each
    for fixed_kernel_ in cv_results_['param_kernel'].unique():

        # Loop over all unique normalizers, create plots for each
        for fixed_normalizer in cv_results_['param_normalizer'].unique():

            # Loop over each parameter and generate a plot
            for param in parameters:
                # if param == 'param_C' or param == 'param_coef0' or not default_params:
                #     continue  # TODO: for debugging, remove later
                cv_results_filtered = deepcopy(cv_results_)
                if param == 'param_degree' and fixed_kernel_ != 'poly':
                    # Skip the 'degree' parameter if the kernel is not 'poly'
                    continue
                if param == 'param_coef0' and fixed_kernel_ == 'rbf':
                    # Skip the 'coef0' parameter if the kernel is 'rbf'
                    continue
                if param == 'param_tol':
                    # Skip the 'tol' parameter as it is not a hyperparameter anymore
                    continue
                if default_params and param not in default_params:
                    # If we are plotting with default parameters, skip the parameter if it is not in the default parameters
                    continue
                if fixed_kernel_ != 'poly':
                    # # If param is poly, drop duplicate rows that differ only by 'degree'
                    # cv_results_filtered = cv_results_filtered.drop_duplicates(subset=['param_C', 'param_tol', 'param_coef0'])
                    # # Remove degree column
                    # # cv_results_filtered = cv_results_filtered.drop(columns=['param_degree'])
                    #
                    # # # Get list of all the other parameters
                    # # other_params = [p for p in parameters if p != param]
                    pass 


                # Get list of all the other parameters
                fixed_params_ = [p for p in parameters if p != param]

                # Create mask to extract rows where all other parameters are at their best values
                mask = pd.DataFrame(cv_results_[fixed_params_] == best_param_vals.drop(index=param)).all(axis=1)

                # Apply mask and drop duplicates based on the current parameter
                varying_param_data = deepcopy(cv_results_[mask])
                # varying_param_data = varying_param_data.drop_duplicates(subset=param)

                # Extract the values to plot
                x = varying_param_data[param]
                y = varying_param_data[f'mean_test_{test_metric}']
                n_train = np.floor(0.8 * 301)
                std = varying_param_data[f'std_test_{test_metric}']  # see: https://github.com/scikit-learn/scikit-learn/discussions/20680
                se = std / np.sqrt(n_train)

                # Create the plot
                param_label = param.split('_')[1]
                fig, ax = plt.subplots(figsize=(7, 5))

                if param == 'param_degree':
                    # Point plot for ordinal categorical variable 'degree'
                    ax.errorbar(x, y, yerr=se, fmt='o', capsize=5, elinewidth=1, markeredgewidth=1)  # , ecolor='blue')
                    ax.set_xticks(x)
                    ax.set_xticklabels(x, rotation=0)
                else:
                    # Line plot for other parameters
                    ax.plot(x, y, linestyle='--', marker='o', label=param_label)
                    ax.fill_between(x, y - se, y + se, alpha=0.2, label=f'SE')  # , color='red')

                # Set the labels other_params  best_param_vals
                test_metric_labels = {
                    'accuracy': 'Accuracy',
                    'roc_auc': 'ROC AUC',
                    'f1': 'F1 Score',
                    'recall': 'Recall',
                    'precision': 'Precision',
                }

                test_metric_label = test_metric_labels[test_metric]
                fixed_param_label = ''
                for fixed_param in fixed_params_:
                    best_param_val = best_param_vals[fixed_param]
                    fixed_param_label += f'{fixed_param.replace("param_", "")}={f'{utils.round_to_1(best_param_val) if (utils.is_power_of_10(best_param_val) or best_param_val == 0 or best_param_val == 3) else f'{best_param_val:4f}'}' if type(best_param_val) is not str else best_param_val}, '
                fixed_param_label = fixed_param_label[:-2]
                param_label = param_label.replace('param_', '')
                title = (
                    f'Effect of {param_label} on Model Performance ({test_metric_label})\n'
                    f'using {f"Default" if default_params else "Optimal"} Parameters ({fixed_param_label})'
                )

                if param == 'param_degree':
                    ax.set_xlabel('Polynomial Kernel Degree')
                else:
                    ax.set_xlabel(param_label)
                ax.set_ylabel(f'Mean CV Validation {test_metric_label}')
                ax.set_title(title)

                # Set log scale for specific parameters
                if param == 'param_degree':
                    ax.set_xscale('linear')
                else:
                    if param == 'param_coef0':
                        # set linthresh to the positive value of coef0 value closest to zero
                        linthresh = np.abs(x[x > 0].min())
                        ax.set_xscale('symlog', linthresh=linthresh)
                        ax.xaxis.set_major_locator(SymmetricalLogLocator(base=10, linthresh=linthresh))
                    else:
                        ax.set_xscale('log')

                # Dynamic y-axis limits with padding
                y_min, y_max = ax.get_ylim()[0], ax.get_ylim()[1]
                x_min, x_max = ax.get_xlim()[0], ax.get_xlim()[1]

                # Find the degree value with the highest accuracy using optimal parameters
                max_accuracy_value = best_row[param]
                max_accuracy_idx = x[x == max_accuracy_value].index[-1]
                max_accuracy_score = y[max_accuracy_idx]
                max_accuracy_std = se[max_accuracy_idx]

                arrow_pos = (max_accuracy_value, max_accuracy_score + max_accuracy_std + 0.05 * max_accuracy_std)
                arrow_length = 0.02
                padding = (y_max - y_min)
                if param == 'param_degree':
                    ax.set_xlim([x_min - 1, x_max + 1])
                    if default_params:
                        ax.set_ylim([y_min - padding * 0.4, y_max + padding * 0.4])
                        arrow_pos = (max_accuracy_value, max_accuracy_score + max_accuracy_std + 0.02 * max_accuracy_std)
                        arrow_length = 0.0045
                    else:
                        ax.set_ylim([y_min - padding * 0.1, y_max + padding * 0.4])
                elif param == 'param_coef0':
                    ax.set_ylim([y_min - padding * 0.1, y_max + padding * 0.37])
                elif param == 'param_C':
                    if default_params:
                        ax.set_ylim([y_min - padding * 0.25, y_max + padding * 0.95])
                        arrow_pos = (max_accuracy_value, max_accuracy_score + max_accuracy_std + 0.02 * max_accuracy_std)
                        arrow_length = 0.007
                    else:
                        ax.set_ylim([y_min - padding * 0.2, y_max + padding * 0.8])
                        arrow_pos = (max_accuracy_value, max_accuracy_score + max_accuracy_std + 0.02 * max_accuracy_std)
                        arrow_length = 0.011

                # Dynamically set y-axis major ticks
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

                if param == 'param_degree':
                    # Annotate the highest accuracy degree  # TODO: fix fallback option
                    ax.annotate(
                        f'{param_label}={utils.round_to_n(max_accuracy_value, 2)}\n{utils.round_to_n(max_accuracy_score, 2)}±{utils.round_to_n(max_accuracy_std, 2)}',
                        xy=arrow_pos,
                        xytext=(arrow_pos[0], arrow_pos[1] + arrow_length),
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        ha='center', va='bottom', fontsize=10, color='black'
                    )
                else:
                    # Annotate the highest accuracy parameter value  # TODO: fix fallback option
                    ax.annotate(
                        f'{param_label}={utils.round_to_n(max_accuracy_value, 2)}\n{utils.round_to_n(max_accuracy_score, 2)}±{utils.round_to_n(max_accuracy_std, 2)}',
                        xy=arrow_pos,
                        xytext=(arrow_pos[0], arrow_pos[1] + arrow_length),
                        arrowprops=dict(facecolor='black'),
                        ha='center', va='bottom', fontsize=10, color='black'
                    )

                plt.legend()

                plt.show()
                if default_params:
                    fig.savefig(CV_RESULTS_DIR / f'effect-param-{param_label}_fixed-{fixed_kernel_}__fixed-{fixed_kernel_}_{metrics_label}-defaults.png', dpi=300, transparent=True, bbox_inches='tight')
                else:
                    fig.savefig(CV_RESULTS_DIR / f'effect-param-{param_label}_fixed-{fixed_normalizer}_fixed-{fixed_kernel_}_{metrics_label}-opt.png', dpi=300, transparent=True, bbox_inches='tight')

                plt.pause(0.5)
                plt.close(fig)
                # Wait for 0.5 seconds to avoid rate limiting in pycharm
                plt.pause(0.5)


# TODo: add_label_band flyttad till utils

def create_heatmap(cv_results_, x_param, y_param, fixed_params_, test_metric, default_params=None, use_default_params=False):
    global metrics_label, gamma_param
    if default_params is None:
        raise ValueError("Default parameters must be specified")
    test_metric_labels = {
        'accuracy':  'Accuracy',
        'roc_auc':   'ROC AUC',
        'f1':        'F1 Score',
        'recall':    'Recall',
        'precision': 'Precision',
    }

    # Filter results based on fixed_params
    mask = np.ones(len(cv_results_), dtype=bool)
    for param, value in fixed_params_.items():
        if type(value) is pd.Series:
            value = value.iloc[0]
        mask = mask & (cv_results_[param] == value)
    filtered_results = cv_results_[mask]

    # Drop duplicate rows based on each unique (x_param, y_param) pair
    aggregated_results = filtered_results.drop_duplicates(subset=[x_param, y_param])

    # Define custom order for y_param or x_param if needed
    max_degree = 15
    if y_param in ['param_degree'] or x_param in ['param_degree']:
        max_degree = filtered_results['param_degree'].max()
    if y_param == 'param_degree':
        custom_order = list(range(1, max_degree+1))
    else:
        custom_order = None

    if x_param == 'param_degree':
        custom_order_x = list(range(1, 16))
    else:
        custom_order_x = None

    # Pivot table to get the heatmap data
    heatmap_data = aggregated_results.pivot(index=y_param, columns=x_param, values=f'mean_test_{test_metric}')
    std_error_data = aggregated_results.pivot(index=y_param, columns=x_param, values=f'std_test_{test_metric}')

    if heatmap_data.empty:
        print(f"No data available for {x_param} vs {y_param} with fixed params {fixed_params_}")
        return

    # Reorder index and columns based on the custom order
    if custom_order is not None:
        heatmap_data = heatmap_data.reindex(custom_order)
        std_error_data = std_error_data.reindex(custom_order)

    if custom_order_x is not None:
        heatmap_data = heatmap_data.reindex(custom_order_x, axis=1)
        std_error_data = std_error_data.reindex(custom_order_x, axis=1)

    # Remove rows or columns that are completely NaN (i.e., empty)
    heatmap_data = heatmap_data.dropna(how='all')
    std_error_data = std_error_data.dropna(how='all')
    heatmap_data = heatmap_data.dropna(axis=1, how='all')
    std_error_data = std_error_data.dropna(axis=1, how='all')

    # Set fig size and proportion based on the x and y data size
    if x_param == 'param_deg':
        fig, ax = plt.subplots(figsize=(min(10, 2 * heatmap_data.shape[1]), min(10, 2 * heatmap_data.shape[0])))
    else:
        fig, ax = plt.subplots(figsize=(min(10, 2 * heatmap_data.shape[1]), min(10, 1 * heatmap_data.shape[0])))
    # Ensure the cells are square-shaped
    ax.set_aspect('equal', adjustable='box')

    # Format annotations with standard error in parentheses
    labels = (
        np.asarray([f"{x:.2f}\n({y:.2f})" for x, y in zip(heatmap_data.values.flatten(), std_error_data.values.flatten())])
    ).reshape(heatmap_data.shape)

    # Add back column and index names
    labels = pd.DataFrame(labels, index=heatmap_data.index, columns=heatmap_data.columns)

    # Don't show annotations if the number of columns is too large
    max_labeled_columns = 16
    if heatmap_data.shape[1] > max_labeled_columns:
        labels = None

    # Filter ticks to only show powers of 10
    shrink_val = 1  # Set length of colorbar
    xticks = None
    yticks = None
    if x_param != 'param_degree' and x_param != 'param_kernel':
        xticks = [i for i in heatmap_data.columns if type(i) is not str and utils.is_power_of_10(i)]
    if y_param != 'param_degree' and y_param != 'param_kernel':
        yticks = [i for i in heatmap_data.index if type(i) is not str and utils.is_power_of_10(i)]
    else:
        shrink_val = 0.4

    # Make sure 0 is included in the ticks for coef0
    if x_param == 'param_coef0':
        xticks = [0] + xticks
    if y_param == 'param_coef0':
        yticks = [0] + yticks

    test_metric_label = test_metric_labels[test_metric]

    # Create the heatmap
    sns.heatmap(
        heatmap_data, annot=labels, fmt="", cmap="coolwarm",
        annot_kws={"size": 10}, cbar_kws={'label': f'Mean CV Validation {test_metric_label} {'' if labels is None else '(±SE)'}', 'shrink': shrink_val},
        linecolor='lightgrey', linewidths=0.5  # Lighter cell borders
    )

    # Set the ticks and labels
    if xticks and x_param != 'param_degree' and x_param != 'param_kernel':
        ax.set_xticks([heatmap_data.columns.get_loc(i) + 0.5 for i in xticks])
        ax.set_xticklabels(list(utils.scientific_notation_formatter(xtick, pos=None) for xtick in xticks))
    if yticks and y_param != 'param_degree' and y_param != 'param_kernel':
        ax.set_yticks([heatmap_data.index.get_loc(i) + 0.5 for i in yticks])
        ax.set_yticklabels(list(utils.scientific_notation_formatter(ytick, pos=None) for ytick in yticks))

    # Set the labels
    fixed_param_label = ''
    fixed_param_text = ''
    for fixed_param, best_param_val in fixed_params_.items():
        fixed_param_label += f'{fixed_param.replace("param_", "")}={f'{utils.format_number(best_param_val, latex=True) if (fixed_param != 'param_degree') else f'{int(best_param_val)}'}' if type(best_param_val) is not str else best_param_val}, '
        fixed_param_text += f'{fixed_param.replace("param_", "")}={f'{utils.format_number(best_param_val, latex=False) if (fixed_param != 'param_degree') else f'{int(best_param_val)}'}' if type(best_param_val) is not str else best_param_val}, '
    fixed_param_label = fixed_param_label + f'gamma={gamma_param}'
    fixed_param_text = fixed_param_label + f'gamma={gamma_param}'
    x_param_label = x_param.replace('param_', '')
    y_param_label = y_param.replace('param_', '')
    title = (
        f'Interaction of {x_param_label} and {y_param_label}\n'
        f'using {f"Default" if use_default_params else "Optimal"} Parameters ({fixed_param_label})\n'
    )
    plt.title(title)
    plt.xlabel(x_param_label)
    plt.ylabel(y_param_label)

    plt.show()

    fixed_param_text = fixed_param_text.replace(', ', '_')
    fig.savefig(
        CV_RESULTS_DIR / f"heatmap-{x_param_label}-{y_param_label}_fixed-{'defaults' if use_default_params else 'opt'}-{fixed_param_text}.png",
        dpi=300, transparent=True, bbox_inches='tight'
    )
    plt.pause(0.5)
    plt.close(fig)


def get_best_parameter_set_for_kernel(kernel, cv_results_, gridsearch_metric, params_to_get_best, alpha_, beta_, svc_default_params):
    # TODO: lägga till gamma som parameter
    if kernel:
        best_params_ = utils.get_best_params(
            cv_results_[cv_results_['param_kernel'] == kernel], gridsearch_metric, params_to_get_best, alpha=alpha_, beta=beta_, default_params=svc_default_params
        )
    else:
        best_params_ = utils.get_best_params(cv_results_, gridsearch_metric, params_to_get_best, alpha=alpha_, beta=beta_, default_params=svc_default_params)
    if type(best_params_) is pd.DataFrame and best_params_.shape[0] != 1:
        warnings.warn(f"Multiple best parameter sets found for kernel '{kernel}'! Take the first one?")
        breakpoint()
        best_params_ = best_params_.iloc[0, :]
    best_params_ = best_params_.to_dict() if type(best_params_) is pd.Series else best_params_.iloc[0].to_dict()
    return best_params_


for gamma_param in GAMMA_PARAMS:
    # if gamma in [0.01, 0.1]:
    #     continue

    # Extract only the rows with the current gamma value
    gamma_mask = cv_results['param_gamma'] == gamma_param
    cv_results_gamma = cv_results[gamma_mask].copy()

    all_best_params_gamma = get_best_parameter_set_for_kernel(kernel=None, cv_results_=cv_results_gamma, gridsearch_metric=GRIDSEARCH_METRIC, params_to_get_best=PARAMS_TO_GET_BEST, alpha_=alpha, beta_=beta, svc_default_params=SVC_DEFAULT_PARAMS)
    # all_best_params_gamma = utils.get_best_params(
    #     cv_results_gamma, GRIDSEARCH_METRIC, PARAMS_TO_GET_BEST, return_all=True, alpha=alpha, beta=beta, default_params=SVC_DEFAULT_PARAMS
    #     )
    # best_params_gamma_series = all_best_params_gamma if type(all_best_params_gamma) is pd.Series else all_best_params_gamma.iloc[0]
    # best_params_gamma = best_params_gamma_series.to_dict() if type(best_params_gamma_series) is pd.Series else best_params_gamma_series


# %% Generate parameter effect line plots
    barplots = False
    if barplots:
        plot_parameter_effects_stacked_bars(cv_results_=cv_results_gamma, parameters=PARAMS_OF_INTEREST, metrics=['roc_auc'], fixing_mode='both', default_params=SVC_DEFAULT_PARAMS)
        plot_parameter_effects_stacked_bars(cv_results_=cv_results_gamma, parameters=PARAMS_OF_INTEREST, metrics=['accuracy'], fixing_mode='both', default_params=SVC_DEFAULT_PARAMS)
        plot_parameter_effects_stacked_bars(cv_results_=cv_results_gamma, parameters=PARAMS_OF_INTEREST, metrics=['f1', 'recall', 'precision'], fixing_mode='both', default_params=SVC_DEFAULT_PARAMS)
        # plot_parameter_effects_stacked_bars(cv_results_=cv_results_gamma, parameters=PARAMS_OF_INTEREST, metrics=['f1', 'recall', 'precision'], fixing_mode='default', default_params=SVC_DEFAULT_PARAMS)
        # plot_parameter_effects_stacked_bars(cv_results_=cv_results_gamma, parameters=PARAMS_OF_INTEREST, metrics=['f1', 'recall', 'precision'], fixing_mode='optimal', default_params=SVC_DEFAULT_PARAMS)

    stacked_lines = False
    if stacked_lines:
        plot_parameter_effects_stacked(cv_results_=cv_results_gamma, parameters=PARAMS_OF_INTEREST, metrics=['roc_auc'], fixing_mode='both', default_params=SVC_DEFAULT_PARAMS)
        plot_parameter_effects_stacked(cv_results_=cv_results_gamma, parameters=PARAMS_OF_INTEREST, metrics=['accuracy'], fixing_mode='both', default_params=SVC_DEFAULT_PARAMS)
        plot_parameter_effects_stacked(cv_results_=cv_results_gamma, parameters=PARAMS_OF_INTEREST, metrics=['f1', 'recall', 'precision'], fixing_mode='both', default_params=SVC_DEFAULT_PARAMS)
        # plot_parameter_effects_stacked(cv_results_=cv_results_gamma, parameters=PARAMS_OF_INTEREST, metrics=['f1', 'recall', 'precision'], fixing_mode='default', default_params=SVC_DEFAULT_PARAMS)
        # plot_parameter_effects_stacked(cv_results_=cv_results_gamma, parameters=PARAMS_OF_INTEREST, metrics=['f1', 'recall', 'precision'], fixing_mode='optimal', default_params=SVC_DEFAULT_PARAMS)

    param_effects = False
    if param_effects:
        # Call plotting functions
        plot_parameter_effects_opt(cv_results_gamma, PARAMS_OF_INTEREST, test_metric='accuracy', default_params=None)  # SVC_DEFAULT_PARAMS)
        plot_parameter_effects_opt(cv_results_gamma, PARAMS_OF_INTEREST, test_metric='accuracy', default_params=SVC_DEFAULT_PARAMS)

    heatmaps = True
    if not heatmaps:
        exit(0)

    # %% Plot the effect of 'tol' on the mean test score (ridgeline plot)

    # Only plot the effect of 'tol' if it is a hyperparameter
    # with plt.rc_context({'font.size': 12}):
    #     plot_tol_ridge(cv_results_=cv_results_gamma, save_path=CV_RESULTS_DIR, test_metric=PLOT_METRIC)


    # %% Create heatmaps
    SVC_DEFAULT_PARAMS['param_kernel'] = SVC_DEFAULT_KERNEL

    plot_deg = False
    for default in [True, False]:

        # %% 1. param_C vs param_coef0 with one of the non-rbf kernels fixed

        if not default:
            # Set fixed params for the first heatmap
            fixed_parameters = {'param_kernel': 'poly', 'param_degree': all_best_params_gamma['param_degree']}

            create_heatmap(deepcopy(cv_results_gamma[cv_results_gamma['param_kernel'] != 'rbf']), 'param_C', 'param_coef0', fixed_parameters, PLOT_METRIC, default_params=SVC_DEFAULT_PARAMS, use_default_params=default)


        # %% 2. param_C vs param_degree (kernel 'poly') with param_coef0 fixed

        # Find the param_coef0 value to fix
        if default:
            fixed_coef0 = SVC_DEFAULT_PARAMS['param_coef0']
        else:
            fixed_coef0 = all_best_params_gamma['param_coef0']

        # Set fixed params for the second heatmap
        fixed_parameters = {'param_kernel': 'poly', 'param_coef0': fixed_coef0}

        create_heatmap(deepcopy(cv_results_gamma[cv_results_gamma['param_kernel'] == 'poly']), 'param_C', 'param_degree', fixed_parameters, PLOT_METRIC, default_params=SVC_DEFAULT_PARAMS, use_default_params=default)


        # %% 5. param_degree vs coef0 (only for kernel = 'poly') with param_C fixed

        # Find the param_C value to fix
        if default:
            fixed_C = SVC_DEFAULT_PARAMS['param_C']
        else:
            fixed_C = all_best_params_gamma['param_C']

        # Set fixed params for the fifth heatmap
        fixed_parameters = {'param_C': fixed_C}

        create_heatmap(deepcopy(cv_results_gamma[cv_results_gamma['param_kernel'] == 'poly']), 'param_coef0', 'param_degree', fixed_parameters, PLOT_METRIC, default_params=SVC_DEFAULT_PARAMS, use_default_params=default)

    breakpoint_ = 1

breakpoint()
