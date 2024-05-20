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
from pathlib import Path
from collections.abc import Sequence
from copy import deepcopy

# External imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For heatmap or advanced plotting
from matplotlib.ticker import MaxNLocator
from joypy import joyplot

# Local imports
import utils


# %% Setup

SEED = utils.RANDOM_SEED  # get random seed
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CV_RESULTS_DIR = PROJECT_ROOT/'data'/'results'/'IterativeImputer-RFR-tol-00175-iter-98-cutoff-17-Age-GridSearch-tol001-Std-FINAL'


# %% Configuration

# Path to the cv_results_ csv file:
CV_RESULTS_PATH: Path = CV_RESULTS_DIR/'2024-05-19-170030__GridSearch_cv_results.csv'

# Verbosity level:
VERBOSE: int = 2  # Unused at the moment

# Specify the column name for the test score
TEST_SCORE_COLUMN_NAME: str = 'final_accuracy'

# Specify parameters of interest
PARAMS_OF_INTEREST = ['C', 'degree', 'coef0']
FIXED_PARAM = 'kernel'  # 'kernel'

VARYING_PARAMS = ['coef0', 'tol']  # ['C', 'degree']

PARAM_PREFIX = 'param_'  # Prefix for the parameter columns in the cv_results_ DataFrame

# Plot x-axis scale
SCALE_X = 'log'  # 'linear' or 'log'

SVC_DEFAULT_PARAMS = {'param_C': 1.0, 'param_degree': 3, 'param_coef0': 0.0}
SVC_DEFAULT_KERNEL = 'rbf'


# %% Plot functions

def plot_parameter_effects_opt(cv_results_, parameters, data_filename, scale, test_score_column_name='mean_test_score', default_params=None, default_kernel='rbf'):
    """
    Plot the effects of varying parameters on model performance.

    :param cv_results_: (pd.DataFrame) The DataFrame containing the cross-validation results.
    :param parameters: (list[str]) List of parameter names to plot.
    :param data_filename: (str) The filename of the data source.
    :param scale: (str) The scale of the x-axis. Either 'linear' or 'log'.
    :param test_score_column_name: (str) The name of the column containing the test score.
        Either 'mean_test_score' or 'final_accuracy'.
    :param default_params: (dict) The default parameters for the model. If None, comparison is made
        against optimal parameters.
    :param default_kernel: (str) The default kernel for the SVC model.
    """
    if default_params is None:
        # Find the best parameters
        best_index = cv_results_['rank_test_score'].idxmin()
        best_param_vals = cv_results_.loc[best_index, parameters]
    else:
        best_param_vals = pd.Series(default_params)

    # Loop over all unique kernels, create plots for each
    for fixed_kernel in cv_results_['param_kernel'].unique():

        # Loop over all unique normalizers, create plots for each
        for fixed_normalizer in cv_results_['param_normalizer'].unique():

            # Loop over each parameter and generate a plot
            for param in parameters:
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
                    # If param is poly, drop duplicate rows that differ only by 'degree'
                    cv_results_filtered = cv_results_filtered.drop_duplicates(subset=['param_C', 'param_tol', 'param_coef0'])
                    # Remove degree column
                    # cv_results_filtered = cv_results_filtered.drop(columns=['param_degree'])

                    # # Get list of all the other parameters
                    # other_params = [p for p in parameters if p != param]

                # Get list of all the other parameters
                other_params = [p for p in parameters if p != param]

                # Create mask to extract rows where all other parameters are at their best values
                mask = pd.DataFrame(cv_results_[other_params] == best_param_vals.drop(index=param)).all(axis=1)

                # Additional filtering to include only rows where the normalizer column is equal to fixed_normalizer
                mask &= (cv_results_['param_normalizer'] == fixed_normalizer)

                # Additional filtering to include only rows where the kernel column is equal to fixed_kernel
                mask &= (cv_results_['param_kernel'] == fixed_kernel)

                # Apply mask and drop duplicates based on the current parameter
                varying_param_data = cv_results_[mask]
                varying_param_data = varying_param_data.drop_duplicates(subset=param)

                # Extract the values to plot
                x = varying_param_data[param]
                y = varying_param_data[test_score_column_name]
                n_train = np.floor(0.8 * 301)
                std = varying_param_data['std_test_score']  # see: https://github.com/scikit-learn/scikit-learn/discussions/20680
                se = std / np.sqrt(n_train)

                # Create the plot
                param_label = param.split('_')[1]
                fig, ax = plt.subplots(figsize=(10, 5))

                if param == 'param_degree':
                    # Point plot for ordinal categorical variable 'degree'
                    ax.errorbar(x, y, yerr=se, fmt='o', capsize=5, elinewidth=1, markeredgewidth=1)
                    ax.set_xticks(x)
                    ax.set_xticklabels(x, rotation=0)

                    # Find the degree with the highest accuracy
                    max_accuracy_idx = y.idxmax()
                    max_accuracy_degree = x[max_accuracy_idx]
                    max_accuracy_value = y[max_accuracy_idx]
                    max_accuracy_std = se[max_accuracy_idx]

                    # Annotate the highest accuracy degree
                    ax.annotate(f'{param_label}={max_accuracy_value:.2f}\n{max_accuracy_value:.2f}±{max_accuracy_std:.2f}', xy=(max_accuracy_degree, max_accuracy_value + max_accuracy_std + 0.05*max_accuracy_std),
                                xytext=(max_accuracy_degree, max_accuracy_value + max_accuracy_std + 0.05*max_accuracy_std + 0.02),
                                arrowprops=dict(facecolor='black', shrink=0.05),
                                ha='center', va='bottom', fontsize=10, color='black')

                else:
                    # Line plot for other parameters
                    ax.plot(x, y, linestyle='--', marker='o', label=param_label)
                    ax.fill_between(x, y - se, y + se, alpha=0.2, label=f'±1σ (SE)')

                    # Find the parameter value with the highest accuracy
                    max_accuracy_idx = y.idxmax()
                    max_accuracy_value = x[max_accuracy_idx]
                    max_accuracy_score = y[max_accuracy_idx]
                    max_accuracy_std = se[max_accuracy_idx]

                    # Annotate the highest accuracy parameter value
                    ax.annotate(f'{param_label}={max_accuracy_value:.2f}\n{max_accuracy_score:.2f}±{max_accuracy_std:.2f}', xy=(max_accuracy_value, max_accuracy_score + max_accuracy_std + 0.05*max_accuracy_std),
                                xytext=(max_accuracy_value, max_accuracy_score + max_accuracy_std + 0.05*max_accuracy_std + 0.02),
                                arrowprops=dict(facecolor='black', shrink=0.05),
                                ha='center', va='bottom', fontsize=10, color='black')

                title = (
                    f'Effect of {param_label} on Model Performance\n'
                    f'using {fixed_normalizer} Normalizer with SVC {fixed_kernel} Kernel'
                )
                if default_params:
                    title += f' and Default Parameters'
                else:
                    title += f' and Optimal Parameters'
                if param == 'param_degree':
                    ax.set_xlabel('Polynomial Kernel Degree')
                else:
                    ax.set_xlabel(param_label)
                ax.set_ylabel('Mean CV Validation Accuracy')
                ax.set_title(title)

                # Set log scale for specific parameters
                if param == 'param_degree':
                    ax.set_xscale('linear')
                elif scale == 'log':
                    ax.set_xscale('log')

                # Dynamic y-axis limits with padding
                y_min, y_max = min(y) - 0.05, max(y) + 0.05
                padding = (y_max - y_min) * 0.1
                ax.set_ylim([y_min - padding, y_max + padding])

                # Dynamically set y-axis major ticks
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

                plt.legend()

                plt.show()
                if default_params:
                    fig.savefig(CV_RESULTS_DIR / f'effect-param-{param_label}_fixed-{fixed_kernel}__fixed-{fixed_kernel}-defaults.png', bbox_inches='tight')
                else:
                    fig.savefig(CV_RESULTS_DIR / f'effect-param-{param_label}_fixed-{fixed_normalizer}_fixed-{fixed_kernel}-opt.png', bbox_inches='tight')

                plt.pause(0.5)
                plt.close(fig)
                # Wait for 0.5 seconds to avoid rate limiting in pycharm
                plt.pause(0.5)


import matplotlib.path as mpath
import matplotlib.patches as mpatches


def add_label_band(ax, top, bottom, label, *, spine_pos=-0.05, tip_pos=-0.02):
    """
    Helper function to add bracket around y-tick labels.

    Author: @tacaswell
    Source: https://stackoverflow.com/questions/67235301/vertical-grouping-of-labels-with-brackets-on-matplotlib

    Parameters
    ----------
    ax : matplotlib.Axes
        The axes to add the bracket to

    top, bottom : floats
        The positions in *data* space to bracket on the y-axis

    label : str
        The label to add to the bracket

    spine_pos, tip_pos : float, optional
        The position in *axes fraction* of the spine and tips of the bracket.
        These will typically be negative

    Returns
    -------
    bracket : matplotlib.patches.PathPatch
        The "bracket" Aritst.  Modify this Artist to change the color etc of
        the bracket from the defaults.

    txt : matplotlib.text.Text
        The label Artist.  Modify this to change the color etc of the label
        from the defaults.

    """
    # grab the yaxis blended transform
    transform = ax.get_yaxis_transform()

    # add the bracket
    bracket = mpatches.PathPatch(
        mpath.Path(
            [
                [tip_pos, top],
                [spine_pos, top],
                [spine_pos, bottom],
                [tip_pos, bottom],
            ]
        ),
        transform=transform,
        clip_on=False,
        facecolor="none",
        edgecolor="k",
        linewidth=2,
    )
    ax.add_artist(bracket)

    # add the label
    txt = ax.text(
        spine_pos - 0.01,
        (top + bottom) / 2,
        label,
        ha="right",
        va="center",
        rotation="vertical",
        clip_on=False,
        transform=transform,
    )

    return bracket, txt


def get_fig_filename(param: str, source_data_filename: Path, suffix: str = '') -> Path:
    """
    Generate filename for figure based on main (e.g. fixed) parameter and source data filename.

    :param param: Parameter to include in the filename.
    :param source_data_filename: Path to the source data file.
    :param suffix: Suffix to add to the filename.
    :return: Path to the figure file.
    """
    return Path(source_data_filename.stem + 'PLT꞉' + param + suffix + '.png')


def replace_column_prefix(df: pd.DataFrame, prefixes: Sequence[str], repl: str = '') -> pd.DataFrame:
    """
    Strip the 'param_' prefix from the column names in a DataFrame.

    :param df: The DataFrame to process.
    :param prefixes: The prefixes to remove from the column names.
    :param repl: The replacement string.
    :return: The DataFrame with the 'param_' prefix removed from the column names.
    """
    for prefix in prefixes:
        df.columns = df.columns.str.replace(prefix, repl)
    return df


def plot_tol_ridge(cv_results_, save_path, accuracy_score_column_name='mean_test_score'):
    # Filter out tolerance parameter values greater or equal to 1e-3
    cv_results_ = cv_results_[cv_results_['param_tol'] >= 1e-3]

    # Ensure the save path is a Path object
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    normalizers = ['MinMaxScaler', 'StandardScaler']
    kernels = ['poly', 'sigmoid', 'rbf']

    # Iterate over each combination of normalizer and kernel
    for normalizer in normalizers:
        for kernel in kernels:
            # Filter the dataframe for the current normalizer and kernel
            df_filtered = cv_results_[(cv_results_['param_normalizer'] == normalizer) &
                                      (cv_results_['param_kernel'] == kernel)]
            if kernel != 'poly':
                # Drop duplicate rows that differ only by 'degree'
                df_filtered = df_filtered.drop_duplicates(subset=['param_C', 'param_tol', 'param_coef0'])
                plot_ridge(df_filtered, save_path / f"ridgeplot-param-tol_fixed-{normalizer}_fixed-{kernel}.png", accuracy_score_column_name, kernel, normalizer)
            else:
                for degree in df_filtered['param_degree'].unique():
                    # Filter the dataframe for the current degree
                    df_degree = df_filtered[df_filtered['param_degree'] == degree]
                    plot_ridge(df_degree, save_path / f"ridgeplot-param-tol_fixed-{normalizer}_fixed-{kernel}-deg{degree}.png", accuracy_score_column_name, kernel, normalizer)

def plot_ridge(df, save_file, accuracy_score_column_name, kernel, normalizer):

    # Group by the combination of hyperparameters excluding 'tol'
    hyperparams = ['param_C', 'param_coef0']
    degree = df['param_degree'].iloc[0] if kernel == 'poly' else None
    df = df.copy()  # Ensure we are working on a copy
    df['hyperparams'] = df[hyperparams].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    # Sort the dataframe by 'tol'
    df_sorted = df.sort_values(by='param_tol')

    # Create a new dataframe for ridgeline plot
    df_ridge = df_sorted.pivot(index='param_tol', columns='hyperparams', values=accuracy_score_column_name)

    # Create the ridgeline plot manually using Matplotlib
    fig, ax = plt.subplots(figsize=(3, 12))  # Adjusted size for better visualization

    # Reduce the vertical offset to allow for more overlap
    vertical_offset = 0.2

    # Colors for lines
    colors = sns.color_palette(palette='husl', n_colors=len(df_ridge.columns))

    # Plot each line
    for i, col in enumerate(df_ridge.columns):
        y = df_ridge[col].values
        x = df_ridge.index.values
        y_offset = y + i * vertical_offset
        ax.plot(x, y_offset, color=colors[i], alpha=1, linewidth=1)

    # Set axis labels and title
    yaxis = ax.yaxis
    yaxis.set_label_text('Mean CV Validation Accuracy')
    yaxis.set_label_position('left')
    yaxis.set_label('Mean CV Validation Accuracy')
    yaxis_sec = ax.secondary_yaxis('right')
    yaxis_sec.set_ylabel('Hyperparameter Combinations', rotation=-90, labelpad=20)
    yaxis_sec.set_yticks([])

    ax.set_xlabel('Tolerance')
    ax.set_title('Effect of Tolerance on CV Validation Accuracy\n'
                 'for Different Hyperparameter Combinations\n'
                 f'using {f'a Deg{degree}-' if degree else ''}{kernel} Kernel and {normalizer} Normalizer')

    # Set x-axis to log scale
    ax.set_xscale('log')

    # Remove y-axis ticks and labels
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_formatter(plt.NullFormatter())

    # Create a small scale bar with a bracket to indicate the y-axis values
    yaxis.set_ticks([0, 1])
    yaxis.set_ticklabels([0, 1])
    x_axis_start, x_axis_end = ax.get_xlim()
    bracket_x = -0.12 + x_axis_start  # x position for the indicator bracket
    bracket = add_label_band(ax, 0, 1, "Accuracy", spine_pos=bracket_x, tip_pos=bracket_x + 0.02)

    yaxis.set_label_coords(x=-0.04, y=0.5)

    plt.show()

    # Save the plot
    fig.savefig(save_file, bbox_inches='tight')
    plt.pause(0.5)
    plt.close(fig)


# %% Generate plots

# Load the cross-validation results
cv_results = pd.read_csv(CV_RESULTS_PATH)

# Make all param prefixes the same
cv_results = replace_column_prefix(cv_results, ['param_classifier__'],  'param_')

# Add prefix to the parameter names
PARAMS_OF_INTEREST = [f'{PARAM_PREFIX}{param}' for param in PARAMS_OF_INTEREST]
FIXED_PARAM = f'{PARAM_PREFIX}{FIXED_PARAM}'
VARYING_PARAMS = [f'{PARAM_PREFIX}{param}' for param in VARYING_PARAMS]

# Sanitize the normalization column
cv_results['param_normalizer'] = cv_results['param_normalizer'].str.split('(').str[0]

# Call plotting functions
# plot_parameter_effects_opt(cv_results, PARAMS_OF_INTEREST, data_filename=CV_RESULTS_PATH, scale=SCALE_X, test_score_column_name=TEST_SCORE_COLUMN_NAME, default_params=None)  # SVC_DEFAULT_PARAMS)
# with plt.rc_context({'font.size': 12}):
#     plot_tol_ridge(cv_results_=cv_results, save_path=CV_RESULTS_DIR, accuracy_score_column_name='mean_test_score')

# for param in PARAMS_OF_INTEREST:
#     varying_params = [p for p in PARAMS_OF_INTEREST if p != param]
#     plot_interactive_effects(cv_results, param, varying_params, CV_RESULTS_PATH, default_params=None)  # SVC_DEFAULT_PARAMS)
#     plot_interactive_effects(cv_results, param, varying_params, CV_RESULTS_PATH, default_params=SVC_DEFAULT_PARAMS)


def create_heatmap(cv_results_, x_param, y_param, fixed_params, accuracy_metric='mean_test_score', default_params=False):
    # Filter the results based on fixed_params
    mask = np.ones(len(cv_results_), dtype=bool)
    for param, value in fixed_params.items():
        mask = mask & (cv_results_[param] == value)
    filtered_results = cv_results_[mask]

    # Aggregate to avoid duplicate entries
    aggregated_results = filtered_results.groupby([x_param, y_param]).agg(
        mean_accuracy=(accuracy_metric, 'mean'),
        std_error=('std_test_score', lambda x: x.mean() / 2)
    ).reset_index()

    # Define custom order for y_param or x_param if needed
    if y_param == 'param_degree':
        custom_order = list(range(1, 11))
    elif y_param == 'param_kernel':
        custom_order = ['sigmoid', 'rbf'] + [f'poly deg{d}' for d in range(1, 11)]
    else:
        custom_order = None

    if x_param == 'param_degree':
        custom_order_x = list(range(1, 11))
    elif x_param == 'param_kernel':
        custom_order_x = ['sigmoid', 'rbf'] + [f'poly deg{d}' for d in range(1, 11)]
    else:
        custom_order_x = None

    # Pivot table to get the heatmap data
    heatmap_data = aggregated_results.pivot(index=y_param, columns=x_param, values='mean_accuracy')
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

    # Format annotations with standard error in parentheses
    labels = (
        np.asarray([f"{x:.2f}\n({y:.2f})" for x, y in zip(heatmap_data.values.flatten(), std_error_data.values.flatten())])
    ).reshape(heatmap_data.shape)

    # Add back column and index names
    labels = pd.DataFrame(labels, index=heatmap_data.index, columns=heatmap_data.columns)

    # Create the heatmap
    fig = plt.figure(figsize=(14, 10))  # Adjusted size for more space
    sns.heatmap(
        heatmap_data, annot=labels, fmt="", cmap="coolwarm",
        annot_kws={"size": 10}, cbar_kws={'label': 'Mean CV Validation Accuracy (±SE)'},
        linecolor='lightgrey', linewidths=0.5  # Lighter cell borders
    )

    fixed_param_label = ''
    for fixed_param, best_param_val in fixed_params.items():
        fixed_param_label += f'{fixed_param.replace("param_", "")}={best_param_val}, '
    fixed_param_label = fixed_param_label[:-2]
    x_param_label = x_param.replace('param_', '')
    y_param_label = y_param.replace('param_', '')
    title = (
        f'Interaction of {x_param_label} and {y_param_label}\n'
        f'with Other Hyperparameters Set to {"Defaults" if default_params else "Optimal"} ({fixed_param_label})\n'
    )
    plt.title(title)
    plt.xlabel(x_param_label)
    plt.ylabel(y_param_label)

    plt.show()

    fixed_param_label = fixed_param_label.replace(', ', '_')
    fig.savefig(
        CV_RESULTS_DIR / f"heatmap-{x_param_label}-{y_param_label}_fixed-{'defaults' if default_params else 'opt'}-{fixed_param_label}.png",
        bbox_inches='tight'
    )
    plt.pause(0.5)
    plt.close(fig)


# %% 1. param_C vs param_coef0 with the best non-rbf kernel fixed

# Find the best kernel (excluding 'rbf')
best_kernel_row = cv_results[cv_results['param_kernel'] != 'rbf'].sort_values(by='mean_test_score', ascending=False).iloc[0]
best_kernel = best_kernel_row['param_kernel']
best_degree = best_kernel_row['param_degree'] if best_kernel == 'poly' else None

# Set fixed params for the first heatmap
fixed_parameters = {'param_kernel': best_kernel}
if best_kernel == 'poly':
    fixed_parameters['param_degree'] = best_degree

create_heatmap(deepcopy(cv_results), 'param_C', 'param_coef0', fixed_parameters, 'mean_test_score')


# %% 2. param_C vs param_degree (kernel 'poly') with param_coef0 fixed

# Find the best param_coef0 value to fix
best_coef0 = cv_results[cv_results['param_kernel'] == 'poly'].sort_values(by='mean_test_score', ascending=False).iloc[0]['param_coef0']

# Set fixed params for the second heatmap
fixed_parameters = {'param_kernel': 'poly', 'param_coef0': best_coef0}

create_heatmap(deepcopy(cv_results), 'param_C', 'param_degree', fixed_parameters, 'mean_test_score')


# %% 3. param_C vs kernel (rbf, sigmoid, poly deg1, poly deg2, ... poly deg10) with param_coef0 fixed

# Modify param_kernel to include poly degrees
cv_results['param_kernel'] = cv_results.apply(
    lambda row: f"{row['param_kernel']} deg{row['param_degree']}" if row['param_kernel'] == 'poly' else row[
        'param_kernel'], axis=1
)

create_heatmap(deepcopy(cv_results), 'param_C', 'param_kernel', {'param_coef0': best_coef0}, 'mean_test_score')


# %% 4. param_coef0 vs kernel ('rbf' excluded; so sigmoid, poly deg1, poly deg2, ... poly deg10) with param_C fixed

# Find the best param_C value to fix
best_C = cv_results[cv_results['param_kernel'] != 'rbf'].sort_values(by='mean_test_score', ascending=False).iloc[0]['param_C']

create_heatmap(
    deepcopy(cv_results[cv_results['param_kernel'] != 'rbf']), 'param_coef0', 'param_kernel', {'param_C': best_C},
    'mean_test_score'
)


# %% 5. param_degree vs coef0 (only for kernel = 'poly') with param_C fixed

# Find the best param_C value to fix
best_C = cv_results[cv_results['param_kernel'].str.startswith('poly')].sort_values(by='mean_test_score', ascending=False).iloc[0]['param_C']

# Set fixed params for the fifth heatmap
fixed_parameters = {'param_C': best_C}

create_heatmap(
    deepcopy(cv_results[cv_results['param_kernel'].str.startswith('poly')]), 'param_coef0', 'param_degree', fixed_parameters,
    'mean_test_score'
)
