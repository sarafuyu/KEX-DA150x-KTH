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
from pathlib import Path
from collections.abc import Sequence
from copy import deepcopy

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
    'final-prestudy': {
        'results_directory': 'final-prestudy',
        'cv_results_':       '2024-06-02-223758__GridSearchCV.pkl',
    },
    'final-prestudy-tol-normalizers': {
        'results_directory': 'final-prestudy-tol-normalizers',
        'cv_results_':       '2024-06-05-000027__GridSearchCV.pkl',
    },
}


# %% Configuration

FEATURE_SELECTION_METRIC = 'final-prestudy-tol-normalizers'  # 'accuracy', 'roc_auc', 'f1'
GRIDSEARCH_METRIC = 'roc_auc'  # 'accuracy', 'roc_auc', 'f1'  # noqa
PLOT_METRIC: str = 'roc_auc'

FIXED_META_PARAMS = {
    # 'normalizer': 'StandardScaler', # ['StandardScaler', 'MinMaxScaler']
    # 'kernel': 'poly',  # ['poly', 'rbf', 'sigmoid']
    'param_tol': np.float64(0.001),
    'param_gamma': 'scale',  # ['auto', 'scale', 0, 0.1, 1.0]
    'param_class_weight': 'None',  # [None, 'balanced'],
}

CV_RESULTS_DIR = PROJECT_ROOT/'data'/'results'/RESULTS_SETS[FEATURE_SELECTION_METRIC]['results_directory']  # noqa
# Path to the cv_results_ csv file:
GRID_SEARCH_RESULTS_PATH: Path = CV_RESULTS_DIR/RESULTS_SETS[FEATURE_SELECTION_METRIC]['cv_results_']

# Verbosity level:
VERBOSE: int = 2  # Unused at the moment

# Specify parameters of interest
PARAMS_OF_INTEREST = ['C', 'degree', 'coef0', 'kernel', 'gamma', 'class_weight', 'tol']
FIXED_PARAM = 'kernel'  # 'kernel'

VARYING_PARAMS = ['coef0', 'tol']  # ['C', 'degree']

PARAM_PREFIX = 'param_'  # Prefix for the parameter columns in the cv_results_ DataFrame

# Plot x-axis scale
SCALE_X = 'log'  # 'linear' or 'log'

SVC_DEFAULT_PARAMS = {'param_C': 1.0, 'param_degree': 3, 'param_coef0': 0.0, 'param_gamma': 'scale', 'param_class_weight': 'None', 'param_tol': 0.001, 'param_kernel': 'rbf'}
SVC_DEFAULT_KERNEL = 'rbf'

GAMMA_PARAMS = ['scale']  # [0.01, 0.1, 1.0]

# # Load the cross-validation results
gridsearch_cv = joblib.load(GRID_SEARCH_RESULTS_PATH)
cv_results = pd.DataFrame(gridsearch_cv.cv_results_)

# Make all param prefixes the same
cv_results = utils.replace_column_prefix(cv_results, ['param_classifier__'],  'param_')

# Add prefix to the parameter names
PARAMS_OF_INTEREST = [f'{PARAM_PREFIX}{param}' for param in PARAMS_OF_INTEREST]
FIXED_PARAM = f'{PARAM_PREFIX}{FIXED_PARAM}'
VARYING_PARAMS = [f'{PARAM_PREFIX}{param}' for param in VARYING_PARAMS]
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
cv_results['param_gamma'] = cv_results['param_gamma'].astype(str)
cv_results['param_class_weight'] = cv_results['param_class_weight'].astype(str)

# Permanently remove all rows with tol, class weights and normalizer not equal to CHOSEN_TOL, CHOSEN_CLASS_WEIGHT, CHOSEN_NORMALIZER
CHOSEN_TOL = 0.001
CHOSEN_CLASS_WEIGHT = 'balanced'
CHOSEN_NORMALIZER = 'StandardScaler'
cv_results = cv_results[cv_results['param_tol'] == CHOSEN_TOL]
cv_results = cv_results[cv_results['param_class_weight'] == CHOSEN_CLASS_WEIGHT]
cv_results = cv_results[cv_results['param_normalizer'] == CHOSEN_NORMALIZER]

# Find the best parameters and break ties
alpha = 0  # 0.7
beta = 0   # 74
gamma = 0  # 0.1

best_cv_results_rows = utils.get_best_params(cv_results, GRIDSEARCH_METRIC, PARAMS_OF_INTEREST, return_all=True, alpha=alpha, beta=beta, default_params=False)
best_cv_results_rows_defaults = utils.get_best_params(cv_results, GRIDSEARCH_METRIC, PARAMS_OF_INTEREST, return_all=True, alpha=alpha, beta=beta, default_params=SVC_DEFAULT_PARAMS)
best_cv_results_row = best_cv_results_rows if type(best_cv_results_rows) is pd.Series else best_cv_results_rows.iloc[0]
best_params = best_cv_results_row.to_dict() if type(best_cv_results_row) is pd.Series else best_cv_results_row

# Penalize (modify) scores in cv_results according to:
#   Adjusted Score = Score − α×ln(degree) − β×ln(std(Score)+1) − γ×ln(gamma+1)
#                      (α only if kernel is poly)      (γ only if kernel is rbf or sigmoid)
if alpha > 0 or beta > 0 or gamma > 0:
    # Create a mask for the polynomial kernel
    poly_kernel_mask = cv_results['param_kernel'] == 'poly'
    rbf_sigmoid_kernel_mask = cv_results['param_kernel'].isin(['rbf', 'sigmoid']) & (cv_results['param_gamma'].isin(['scale', 'auto']) == False)
    # Adjust the score based on the conditions
    cv_results[f'mean_test_{GRIDSEARCH_METRIC}'] = (
        cv_results[f'mean_test_{GRIDSEARCH_METRIC}']
        - alpha * (np.log(cv_results['param_degree']) * poly_kernel_mask)
        - beta * (np.log(cv_results[f'mean_test_{GRIDSEARCH_METRIC}'] + 1))
        - gamma * (np.log(cv_results['param_gamma'] + 1) * rbf_sigmoid_kernel_mask)
    )

# Find row of the best parameters in cv_results
mask = pd.DataFrame(cv_results['param_C'] == best_params['param_C']).all(axis=1)
mask &= cv_results['param_degree'] == best_params['param_degree']
mask &= cv_results['param_coef0'] == best_params['param_coef0']
mask &= cv_results['param_kernel'] == best_params['param_kernel']
mask &= cv_results['param_gamma'] == best_params['param_gamma']
mask &= cv_results['param_tol'] == CHOSEN_TOL
mask &= cv_results['param_class_weight'] == CHOSEN_CLASS_WEIGHT

# Pick the row with the best parameters and lowest rank
best_rows = cv_results[mask].sort_values(f'rank_test_{GRIDSEARCH_METRIC}')
lowest_rank = best_rows[f'rank_test_{GRIDSEARCH_METRIC}'].min()
best_params = best_rows[best_rows[f'rank_test_{GRIDSEARCH_METRIC}'] == lowest_rank]

# If there is only one row with the best value, return it
if best_params.shape[0] != 1:
    warnings.warn(f"Multiple rows with the best value found!")
    breakpoint()

best_params = utils.replace_column_prefix(best_params, ['classifier__'],  'param_')

calculate_metrics = False
if calculate_metrics:
    # Load imputed data and split it into training and testing sets
    imputed_X = pd.read_csv(CV_RESULTS_DIR / '2024-06-05-000027__IterativeImputer_X_imputed.csv')
    imputed_X = imputed_X.drop(columns='Unnamed: 0')  # Drop the "Unnamed: 0" column
    dataset_dict = joblib.load(CV_RESULTS_DIR / '2024-06-05-000027__FeatureSelect꞉XGB-RFE-CV_dataset_dict.pkl')
    y_training = dataset_dict['y_training']
    y_testing = dataset_dict['y_testing']
    dataset = dataset_dict['dataset']
    y = dataset_dict['dataset']['FT5']
    TEST_PROPORTION: float = 0.2
    X_training, X_testing, y_training_, y_testing_ = train_test_split(
        imputed_X, y,
        test_size=TEST_PROPORTION,
        random_state=SEED
    )

    # Make sure the labels after split match the original labels (i.e. that we have the same exact split as before)
    if not (y_training['FT5'].equals(y_training_)) and not (y_testing['FT5'].equals(y_testing_)):
        warnings.warn(Warning("The training and testing labels do not match the original labels!"))
        breakpoint()

    # Set parameters for the best estimator using the row with the best parameters making sure all parameters ('param_') are included
    best_estimator = deepcopy(gridsearch_cv.best_estimator_)
    for param in best_params.iloc[0, :].index:
        if param.startswith(PARAM_PREFIX) and param not in ['param_normalizer', 'normalizer']:
            best_estimator.set_params(**{param.replace('param_', 'classifier__'): best_params[param].values[0]})
        elif param in ['param_normalizer', 'normalizer']:
            if best_params[param].values[0] == 'StandardScaler':
                best_estimator.set_params(**{'normalizer': StandardScaler(copy=False)})
            elif best_params[param].values[0] == 'MinMaxScaler':
                best_estimator.set_params(**{'normalizer': MinMaxScaler(copy=False)})
            else:
                warnings.warn(f"Unknown normalizer: {best_params[param].values[0]}")
                breakpoint()
    print("Winning parameters/model based on roc_auc:\n", best_estimator)

    # Fit the model with full training set and evaluate using test set
    best_estimator.fit(X_training, y_training)
    final_roc_auc = roc_auc_score(y_testing, best_estimator.predict(X_testing))
    final_M = confusion_matrix(y_testing, best_estimator.predict(X_testing))

    # Print the calculated final test metrics
    print("Confusion Matrix (final, separate test set):\n", final_M)
    print("ROC AUC (final, separate test set): ", final_roc_auc)

    # Print the CV classification report (mean validation metrics)
    for label in best_params:
        score = best_params[label].values[0]
        if 'train' in label or 'param' in label:
            continue
        print(f"{label}={score}")

    breakpoint_ = 1


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
            bbox_inches='tight'
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
        mask = pd.DataFrame(cv_results_[fixed_params_] == best_params_.drop(index=param)).all(axis=1)
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
        bbox_inches='tight'
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
    global best_params, metrics_label
    if test_metric is None:
        raise ValueError("test_metric must be specified")
    if default_params is None:
        # Find the best parameters
        # best_index = cv_results_['rank_test_score'].idxmin()
        # best_param_vals = cv_results_.loc[best_index, parameters]
        best_param_vals = pd.Series({'param_C': best_params['param_C'], 'param_degree': best_params['param_degree'], 'param_coef0': best_params['param_coef0']})
    else:
        best_param_vals = pd.Series(default_params)

    # Loop over all unique kernels, create plots for each
    for fixed_kernel in cv_results_['param_kernel'].unique():

        # Loop over all unique normalizers, create plots for each
        for fixed_normalizer in cv_results_['param_normalizer'].unique():

            # Loop over each parameter and generate a plot
            for param in parameters:
                # if param == 'param_C' or param == 'param_coef0' or not default_params:
                #     continue  # TODO: for debugging, remove later
                cv_results_filtered = deepcopy(cv_results_)
                if param == 'param_degree' and fixed_kernel != 'poly':
                    # Skip the 'degree' parameter if the kernel is not 'poly'
                    continue
                if param == 'param_coef0' and fixed_kernel == 'rbf':
                    # Skip the 'coef0' parameter if the kernel is 'rbf'
                    continue
                if param == 'param_tol':
                    # Skip the 'tol' parameter as it is not a hyperparameter anymore
                    continue
                if default_params and param not in default_params:
                    # If we are plotting with default parameters, skip the parameter if it is not in the default parameters
                    continue
                if fixed_kernel != 'poly':
                    # # If param is poly, drop duplicate rows that differ only by 'degree'
                    # cv_results_filtered = cv_results_filtered.drop_duplicates(subset=['param_C', 'param_tol', 'param_coef0'])
                    # # Remove degree column
                    # # cv_results_filtered = cv_results_filtered.drop(columns=['param_degree'])
                    #
                    # # # Get list of all the other parameters
                    # # other_params = [p for p in parameters if p != param]
                    continue  # Skip the 'degree' parameter if the kernel is not 'poly' since we have limited our search to poly kernels for now


                # Get list of all the other parameters
                fixed_params_ = [p for p in parameters if p != param]

                # Create mask to extract rows where all other parameters are at their best values
                mask = pd.DataFrame(cv_results_[fixed_params_] == best_param_vals.drop(index=param)).all(axis=1)

                # Since we have limited ourselves to only StandardScaler and poly kernels, we can skip this step for now
                # # Additional filtering to include only rows where the normalizer column is equal to fixed_normalizer
                # mask &= (cv_results_['param_normalizer'] == fixed_normalizer)
                #
                # # Additional filtering to include only rows where the kernel column is equal to fixed_kernel
                # mask &= (cv_results_['param_kernel'] == fixed_kernel)

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
                max_accuracy_value = best_params[param]
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
                    fig.savefig(CV_RESULTS_DIR / f'effect-param-{param_label}_fixed-{fixed_kernel}__fixed-{fixed_kernel}_{metrics_label}-defaults.png', bbox_inches='tight')
                else:
                    fig.savefig(CV_RESULTS_DIR / f'effect-param-{param_label}_fixed-{fixed_normalizer}_fixed-{fixed_kernel}_{metrics_label}-opt.png', bbox_inches='tight')

                plt.pause(0.5)
                plt.close(fig)
                # Wait for 0.5 seconds to avoid rate limiting in pycharm
                plt.pause(0.5)


# TODo: add_label_band flyttad till utils

def create_heatmap(cv_results_, x_param, y_param, fixed_params, test_metric, default_params=False, use_default_params=None):
    global metrics_label
    if default is None:
        raise ValueError("Default parameter must be specified")
    test_metric_labels = {
        'accuracy':  'Accuracy',
        'roc_auc':   'ROC AUC',
        'f1':        'F1 Score',
        'recall':    'Recall',
        'precision': 'Precision',
    }

    # Filter results based on fixed_params
    mask = np.ones(len(cv_results_), dtype=bool)
    for param, value in fixed_params.items():
        mask = mask & (cv_results_[param] == value)
    filtered_results = cv_results_[mask]

    # Aggregate to avoid duplicate entries
    aggregated_results = filtered_results.groupby([x_param, y_param]).agg(
        mean_score=(f'mean_test_{test_metric}', 'mean'),
        std_error=(f'std_test_{test_metric}', lambda x: x.mean() / 2)
    ).reset_index()

    # Define custom order for y_param or x_param if needed
    if y_param == 'param_degree':
        custom_order = list(range(1, 16))
    elif y_param == 'param_kernel':
        custom_order = ['sigmoid', 'rbf'] + [f'poly deg{d}' for d in list(range(1, 16))]
    else:
        custom_order = None

    if x_param == 'param_degree':
        custom_order_x = list(range(1, 16))
    elif x_param == 'param_kernel':
        custom_order_x = ['sigmoid', 'rbf'] + [f'poly deg{d}' for d in list(range(1, 16))]
    else:
        custom_order_x = None

    # Pivot table to get the heatmap data
    heatmap_data = aggregated_results.pivot(index=y_param, columns=x_param, values='mean_score')
    std_error_data = aggregated_results.pivot(index=y_param, columns=x_param, values='std_error')

    if heatmap_data.empty:
        print(f"No data available for {x_param} vs {y_param} with fixed params {fixed_params}")
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
    if x_param == 'param_degree':
        fig, ax = plt.subplots(figsize=(min(10, 2 * heatmap_data.shape[1]), min(10, 2 * heatmap_data.shape[0])))
    else:
        fig, ax = plt.subplots(figsize=(min(10, 1 * heatmap_data.shape[1]), min(10, 1 * heatmap_data.shape[0])))
    # Ensure the cells are square-shaped
    ax.set_aspect('equal', adjustable='box')

    # Format annotations with standard error in parentheses
    labels = (
        np.asarray([f"{x:.2f}\n({y:.2f})" for x, y in zip(heatmap_data.values.flatten(), std_error_data.values.flatten())])
    ).reshape(heatmap_data.shape)

    # Add back column and index names
    labels = pd.DataFrame(labels, index=heatmap_data.index, columns=heatmap_data.columns)

    # Don't show annotations if the number of columns is too large
    if heatmap_data.shape[1] > 15:
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

    test_metric_label = test_metric_labels[test_metric]

    # Create the heatmap
    sns.heatmap(
        heatmap_data, annot=labels, fmt="", cmap="coolwarm",
        annot_kws={"size": 10}, cbar_kws={'label': f'Mean CV Validation {test_metric_label} {'' if labels is None else '(±SE)'}', 'shrink': shrink_val},
        linecolor='lightgrey', linewidths=0.5  # Lighter cell borders
    )

    # #  Set axis labels to scientific notation unless the parameter is 'degree' which is ordinal
    # if x_param != 'param_degree':
    #     ax.xaxis.set_major_formatter(FuncFormatter(utils.scientific_notation_formatter))
    # if y_param != 'param_degree':
    #     ax.yaxis.set_major_formatter(FuncFormatter(utils.scientific_notation_formatter))

    # Set the ticks and labels
    if xticks and x_param != 'param_degree' and x_param != 'param_kernel':
        ax.set_xticks([heatmap_data.columns.get_loc(i) + 0.5 for i in xticks])
        ax.set_xticklabels(list(utils.scientific_notation_formatter(xtick, pos=None) for xtick in xticks))
    if yticks and y_param != 'param_degree' and y_param != 'param_kernel':
        ax.set_yticks([heatmap_data.index.get_loc(i) + 0.5 for i in yticks])
        ax.set_yticklabels(list(utils.scientific_notation_formatter(ytick, pos=None) for ytick in yticks))

    # Adjust the tick parameters to make labels horizontal
    # ax.tick_params(axis='x', labelrotation=0)
    # ax.tick_params(axis='y', labelrotation=0)

    # Set the labels
    fixed_param_label = ''
    fixed_param_text = ''
    for fixed_param, best_param_val in fixed_params.items():
        if fixed_param == 'param_kernel':
            continue
        fixed_param_label += f'{fixed_param.replace("param_", "")}={f'{utils.format_number(best_param_val, latex=True) if (fixed_param != 'param_degree') else f'{int(best_param_val)}'}' if type(best_param_val) is not str else best_param_val}, '
        fixed_param_text += f'{fixed_param.replace("param_", "")}={f'{utils.format_number(best_param_val, latex=False) if (fixed_param != 'param_degree') else f'{int(best_param_val)}'}' if type(best_param_val) is not str else best_param_val}, '
    fixed_param_label = fixed_param_label[:-2]
    fixed_param_text = fixed_param_label[:-2]
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
        bbox_inches='tight'
    )
    plt.pause(0.5)
    plt.close(fig)


for gamma in GAMMA_PARAMS:
    if gamma in [0.01, 0.1]:
        continue
    # Extract only the rows with the current gamma value
    gamma_mask = cv_results['param_gamma'] == gamma
    gamma_results = cv_results[gamma_mask]

    best_cv_results_rows = utils.get_best_params(
        gamma_results, GRIDSEARCH_METRIC, PARAMS_OF_INTEREST, return_all=True, alpha=alpha, beta=beta
        )
    best_cv_results_row = best_cv_results_rows if type(best_cv_results_rows) is pd.Series else best_cv_results_rows.iloc[0]

    # pa = PROJECT_ROOT / 'data' / 'results' / 'IterativeImputer-RFR-tol-00175-iter-98-cutoff-17-Age-GridSearch-tol-0001-FINAL-poly-detailed-accuracy-2'
    # dataset_dict = joblib.load(pa / f'2024-05-27-172747__GridSearch_dataset_dict.pkl')
    # del dataset_dict['X_training']
    # del dataset_dict['y_training']
    # pred_X = gridsearch_cv.best_estimator_.predict(dataset_dict['X_testing'])
    best_params = best_cv_results_row.to_dict() if type(best_cv_results_row) is pd.Series else best_cv_results_row
    best_params['param_kernel'] = 'poly'


# %% Generate parameter effect line plots
    barplots = False
    if barplots:
        plot_parameter_effects_stacked_bars(cv_results_=gamma_results, parameters=PARAMS_OF_INTEREST, metrics=['roc_auc'], fixing_mode='both', default_params=SVC_DEFAULT_PARAMS)
        plot_parameter_effects_stacked_bars(cv_results_=gamma_results, parameters=PARAMS_OF_INTEREST, metrics=['accuracy'], fixing_mode='both', default_params=SVC_DEFAULT_PARAMS)
        plot_parameter_effects_stacked_bars(cv_results_=gamma_results, parameters=PARAMS_OF_INTEREST, metrics=['f1', 'recall', 'precision'], fixing_mode='both', default_params=SVC_DEFAULT_PARAMS)
        # plot_parameter_effects_stacked_bars(cv_results_=gamma_results, parameters=PARAMS_OF_INTEREST, metrics=['f1', 'recall', 'precision'], fixing_mode='default', default_params=SVC_DEFAULT_PARAMS)
        # plot_parameter_effects_stacked_bars(cv_results_=gamma_results, parameters=PARAMS_OF_INTEREST, metrics=['f1', 'recall', 'precision'], fixing_mode='optimal', default_params=SVC_DEFAULT_PARAMS)

    stacked_lines = False
    if stacked_lines:
        plot_parameter_effects_stacked(cv_results_=gamma_results, parameters=PARAMS_OF_INTEREST, metrics=['roc_auc'], fixing_mode='both', default_params=SVC_DEFAULT_PARAMS)
        plot_parameter_effects_stacked(cv_results_=gamma_results, parameters=PARAMS_OF_INTEREST, metrics=['accuracy'], fixing_mode='both', default_params=SVC_DEFAULT_PARAMS)
        plot_parameter_effects_stacked(cv_results_=gamma_results, parameters=PARAMS_OF_INTEREST, metrics=['f1', 'recall', 'precision'], fixing_mode='both', default_params=SVC_DEFAULT_PARAMS)
        # plot_parameter_effects_stacked(cv_results_=gamma_results, parameters=PARAMS_OF_INTEREST, metrics=['f1', 'recall', 'precision'], fixing_mode='default', default_params=SVC_DEFAULT_PARAMS)
        # plot_parameter_effects_stacked(cv_results_=gamma_results, parameters=PARAMS_OF_INTEREST, metrics=['f1', 'recall', 'precision'], fixing_mode='optimal', default_params=SVC_DEFAULT_PARAMS)

    param_effects = False
    if param_effects:
        # Call plotting functions
        plot_parameter_effects_opt(gamma_results, PARAMS_OF_INTEREST, test_metric='accuracy', default_params=None)  # SVC_DEFAULT_PARAMS)
        plot_parameter_effects_opt(gamma_results, PARAMS_OF_INTEREST, test_metric='accuracy', default_params=SVC_DEFAULT_PARAMS)

    heatmaps = True
    if not heatmaps:
        exit(0)

    # %% Plot the effect of 'tol' on the mean test score (ridgeline plot)

    # Only plot the effect of 'tol' if it is a hyperparameter
    # with plt.rc_context({'font.size': 12}):
    #     plot_tol_ridge(cv_results_=gamma_results, save_path=CV_RESULTS_DIR, test_metric=PLOT_METRIC)


    # %% Create heatmaps
    SVC_DEFAULT_PARAMS['param_kernel'] = SVC_DEFAULT_KERNEL


    # Filter the cv_results_ DataFrame to only include data fixed by the FIXED_META_PARAMS
    mask = pd.DataFrame(cv_results[FIXED_META_PARAMS.keys()] == pd.Series(FIXED_META_PARAMS)).all(axis=1)
    cv_results = cv_results[mask]


    default = None
    for fixed_params in [best_params]:  #, SVC_DEFAULT_PARAMS]:
        if fixed_params == SVC_DEFAULT_PARAMS:
            default = True
        else:
            default = False

        # %% 1. param_C vs param_coef0 with the best non-rbf kernel fixed

        # Find the best kernel (excluding 'rbf')
        best_kernel_row = gamma_results[gamma_results['param_kernel'] != 'rbf'].sort_values(by=f'mean_test_{PLOT_METRIC}', ascending=False).iloc[0]
        best_kernel = fixed_params['param_kernel'] if fixed_params['param_kernel'] != 'rbf' else best_kernel_row['param_kernel']
        best_degree = best_params['param_degree']  # fixed_params['param_degree'] if best_kernel == 'poly' else None

        # Set fixed params for the first heatmap
        fixed_parameters = {'param_kernel': best_kernel}
        if best_kernel == 'poly':
            fixed_parameters['param_degree'] = best_degree

        # Skip the first heatmap if we are plotting with default parameters since default kernel is 'rbf' which is not compatible with 'coef0'
        if best_kernel == 'rbf' or fixed_params == SVC_DEFAULT_PARAMS:
            pass
        else:
            create_heatmap(deepcopy(gamma_results), 'param_C', 'param_coef0', fixed_parameters, PLOT_METRIC, use_default_params=default)


        # %% 2. param_C vs param_degree (kernel 'poly') with param_coef0 fixed

        # Find the best param_coef0 value to fix
        best_coef0 = best_params['param_coef0']  # fixed_params['param_coef0']

        # Set fixed params for the second heatmap
        fixed_parameters = {'param_kernel': 'poly', 'param_coef0': best_coef0}

        create_heatmap(deepcopy(cv_results), 'param_C', 'param_degree', fixed_parameters, PLOT_METRIC, use_default_params=default)


        # %% 3. param_C vs kernel (rbf, sigmoid, poly deg1, poly deg2, ... poly deg10) with param_coef0 fixed

        if fixed_params['param_kernel']:
            pass  # Skip the third heatmap if the fixed param is kernel since we have limited ourselves to poly and already plotted the effect of 'degree' on 'C'
        else:
            # Modify param_kernel to include poly degrees
            cv_results['param_kernel'] = cv_results.apply(
                lambda row: f"{row['param_kernel']} deg{row['param_degree']}" if row['param_kernel'] == 'poly' else row[
                    'param_kernel'], axis=1
            )

            create_heatmap(deepcopy(cv_results), 'param_C', 'param_kernel', {'param_coef0': best_coef0}, PLOT_METRIC, use_default_params=default)


        # %% 4. param_coef0 vs kernel ('rbf' excluded; so sigmoid, poly deg1, poly deg2, ... poly deg10) with param_C fixed

        if fixed_params['param_kernel']:
            pass   # Skip the fourth heatmap if the fixed param is kernel since we have limited ourselves to poly and already plotted the effect of 'degree' on 'C'
        else:
            # Find the best param_C value to fix
            best_C = best_params['param_C']   # fixed_params['param_C']

            create_heatmap(
                deepcopy(cv_results[cv_results['param_kernel'] != 'rbf']), 'param_coef0', 'param_kernel', {'param_C': best_C},
                PLOT_METRIC
            )


        # %% 5. param_degree vs coef0 (only for kernel = 'poly') with param_C fixed

        # Find the best param_C value to fix
        best_C = best_params['param_C']   # fixed_params['param_C']

        # Set fixed params for the fifth heatmap
        fixed_parameters = {'param_C': best_C}

        create_heatmap(
            deepcopy(cv_results[cv_results['param_kernel'].str.startswith('poly')]), 'param_coef0', 'param_degree', fixed_parameters,
            PLOT_METRIC, use_default_params=default
        )

    x = 1
