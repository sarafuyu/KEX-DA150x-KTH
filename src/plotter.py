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

# External imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For heatmap or advanced plotting

# Local imports
import utils


# %% Setup

SEED = utils.RANDOM_SEED  # get random seed
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CV_RESULTS_DIR = PROJECT_ROOT/'data'/'results'/'GridSearch-full-final-StdScaler-MinMaxScaler-2024-05-16-112335'


# %% Configuration

# Path to the cv_results_ csv file:
CV_RESULTS_PATH: Path = CV_RESULTS_DIR/'2024-05-16-112335__cv_results.csv'

# Verbosity level:
VERBOSE: int = 2  # Unused at the moment

# Specify the column name for the test score
TEST_SCORE_COLUMN_NAME: str = 'final_accuracy'

# Specify parameters of interest
PARAMS_OF_INTEREST = ['C', 'degree', 'coef0', 'tol']
FIXED_PARAM = 'kernel'  # 'kernel'

VARYING_PARAMS = ['coef0', 'tol']  # ['C', 'degree']

PARAM_PREFIX = 'param_'  # Prefix for the parameter columns in the cv_results_ DataFrame

# Plot x-axis scale
SCALE_X = 'log'  # 'linear' or 'log'


# %% Plot functions

def plot_parameter_effects(cv_results_, parameters, data_filename, scale, test_score_column_name='mean_test_score'):
    """
    Plot the effects of varying parameters on model performance.

    :param cv_results_: (pd.DataFrame) The DataFrame containing the cross-validation results.
    :param parameters: (list[str]) List of parameter names to plot.
    :param data_filename: (str) The filename of the data source.
    :param scale: (str) The scale of the x-axis. Either 'linear' or 'log'.
    :param test_score_column_name: (str) The name of the column containing the test score.
        Either 'mean_test_score' or 'final_accuracy'.
    """
    # Find the best parameters
    best_index = cv_results_['rank_test_score'].idxmin()
    best_param_vals = cv_results_.loc[best_index, parameters]

    # Loop over all unique kernels, create plots for each
    for fixed_kernel in cv_results_['param_kernel'].unique():

        # Loop over all unique normalizers, create plots for each
        for fixed_normalizer in cv_results_['param_normalizer'].unique():

            # Loop over each parameter and generate a plot
            for param in parameters:

                # Get list of all the other parameters
                other_params = [p for p in parameters if p != param]

                # Create mask to extract rows where all other parameters are at their best values
                mask = pd.DataFrame(cv_results_[other_params] == best_param_vals.drop(index=param)).all(axis=1)

                # Additional filtering to include only rows where the normalizer column is equal to fixed_normalizer
                mask &= (cv_results_['param_normalizer'] == fixed_normalizer)

                # Apply mask and drop duplicates based on the current parameter
                varying_param_data = cv_results_[mask]
                varying_param_data = varying_param_data.drop_duplicates(subset=param)

                # Extract the values to plot
                x = varying_param_data[param]
                y = varying_param_data[test_score_column_name]
                e = varying_param_data['std_test_score'] / 2  # error bars now represent the standard error (±1 standard deviation)

                # Create the plot
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.errorbar(x, y, e, linestyle='--', marker='o', label=param)
                ax.set_xlabel(param)
                ax.set_ylabel('Test Accuracy Score')
                ax.set_title(f'Effect of {param} on Model Performance')

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(x, y, linestyle='--', marker='o', label=param)
                ax.fill_between(x, y - e, y + e, alpha=0.2, label=f'{param} ±1σ (SE)')
                ax.set_xlabel(param)
                ax.set_ylabel('Test Accuracy Score')
                ax.set_title(f'Effect of {param} on Model Performance')

                # Set log scale for specific parameters
                if scale == 'log':
                    ax.set_xscale('log')

                # Set y-axis limits and labels if needed
                ax.set_ylim([min(y) - 0.05, max(y) + 0.05])  # Adjust based on data range
                ax.yaxis.set_major_locator(plt.MaxNLocator(10))  # Set the number of ticks on y-axis

                plt.legend()
                plt.show()

                fig.savefig(CV_RESULTS_DIR / get_fig_filename(param, data_filename, suffix=f'_{fixed_kernel}_{fixed_normalizer}_{scale}_test'))


def plot_interactive_effects(cv_results_, fixed_param, varying_params, data_filename: Path):
    """
    Plot the effects of varying two parameters on model performance while fixing another parameter at its best value.

    :param cv_results_: (pd.DataFrame) The DataFrame containing the cross-validation results.
    :param fixed_param: (str) The parameter to fix at its best value.
    :param varying_params: (list[str]) The two parameters to vary and visualize.
    :param data_filename: (str) The filename of the data source.
    """
    # Ensure only two varying parameters are specified
    if len(varying_params) != 2:
        raise ValueError("Exactly two varying parameters must be specified.")

    # Loop over all unique normalizers, create a plot for each
    for fixed_normalizer in cv_results_['param_normalizer'].unique():

        # Find the best parameter value to fix
        best_index = cv_results_['final_accuracy'].idxmax()
        best_value = cv_results_.loc[best_index, f'{fixed_param}']

        # Filter the DataFrame to include only rows where the fixed parameter is at its best value
        filtered_data = cv_results_[cv_results_[f'{fixed_param}'] == best_value]

        # Filter the DataFrame to include only rows where the fixed normalizer is the specified type
        filtered_data = filtered_data[filtered_data['param_normalizer'] == fixed_normalizer]

        # Pivot table for the heatmap
        pivot_table = filtered_data.pivot_table(
            values='final_accuracy',
            index=f'{varying_params[0]}',
            columns=f'{varying_params[1]}'
        )

        # Plotting
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="coolwarm", cbar_kws={'label': 'Test Accuracy Score'})
        plt.title(f'Interaction of {varying_params[0]} and {varying_params[1]} \nwith {fixed_param} fixed at {best_value}')
        plt.xlabel(varying_params[1])
        plt.ylabel(varying_params[0])
        plt.show()

        fig.savefig(CV_RESULTS_DIR/get_fig_filename(fixed_param, data_filename, suffix='heatmap'))


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

# Call the plotting function
plot_parameter_effects(cv_results, PARAMS_OF_INTEREST, data_filename=CV_RESULTS_PATH, scale=SCALE_X)

# Call the function with example parameters
plot_interactive_effects(cv_results, FIXED_PARAM, VARYING_PARAMS, data_filename=CV_RESULTS_PATH)
