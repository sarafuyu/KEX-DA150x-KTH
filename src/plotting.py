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
CV_RESULTS_DIR = PROJECT_ROOT/'data'/'results_test'


# %% Configuration

# Path to the cv_results_ csv file:
CV_RESULTS_PATH: Path = CV_RESULTS_DIR/'TP꞉NAN_ELIMINATION_CL꞉17_KF꞉30_PR꞉0.2_RS꞉42_CF꞉SVC_DT꞉2024-05-04-110044.csv'

# Verbosity level:
VERBOSE: int = 2  # Unused at the moment

# Specify the column name for the test score
TEST_SCORE_COLUMN_NAME: str = 'final_accuracy'

# Specify parameters of interest
PARAMS_OF_INTEREST = ['C', 'kernel', 'degree', 'coef0', 'tol']
FIXED_PARAM = 'degree' # 'kernel'
VARYING_PARAMS = ['coef0','tol'] # ['C', 'degree']


# %% Plot functions

def plot_parameter_effects(cv_results_, parameters, data_filename, test_score_column_name='mean_test_score'):
    """
    Plot the effects of varying parameters on model performance.

    :param cv_results_: (pd.DataFrame) The DataFrame containing the cross-validation results.
    :param parameters: (list of str) List of parameter names to plot.
    :param test_score_column_name: (str) The name of the column containing the test score.
        Either 'mean_test_score' or 'final_accuracy'.
    :param data_filename: (str) The filename of the data source.
    """
    # Find the best parameters
    best_index = cv_results_['rank_test_score'].idxmin()
    best_params = cv_results_.loc[best_index, [f'param_{param}' for param in parameters]]

    # Loop over each parameter and generate a plot
    for param_name in parameters:
        full_param_name = f'param_{param_name}'

        # Filter data for plots
        mask = (cv_results_.loc[:, [f'param_{p}' for p in parameters if p != param_name]] == best_params.drop(
            index=full_param_name
        )).all(axis=1)
        varying_param_data = cv_results_[mask]
        varying_param_data = varying_param_data.drop_duplicates(subset=full_param_name)

        # Extract the values to plot
        x = varying_param_data[full_param_name]
        y = varying_param_data[test_score_column_name]
        e = varying_param_data['std_test_score']

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(x, y, e, linestyle='--', marker='o', label=param_name)
        ax.set_xlabel(param_name)
        ax.set_ylabel('Test Accuracy Score')
        ax.set_title(f'Effect of {param_name} on Model Performance')

        # Set log scale for specific parameters
        if param_name in ['C', 'coef0', 'tol']:
            ax.set_xscale('log')

        # Set y-axis limits and labels if needed
        ax.set_ylim([min(y) - 0.05, max(y) + 0.05])  # Adjust based on data range
        ax.yaxis.set_major_locator(plt.MaxNLocator(10))  # Set the number of ticks on y-axis

        plt.legend()
        plt.show()

        fig.savefig(PROJECT_ROOT/'out'/get_fig_filename(param_name, data_filename))


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

    # Find the best parameter value to fix
    # TODO(Sara): Need rewrite to work with the new final_accuracy column
    # TODO: Check quick change
    best_index = cv_results_['final_accuracy'].idxmax()
    best_value = cv_results_.loc[best_index, f'param_{fixed_param}']

    # Filter the DataFrame to include only rows where the fixed parameter is at its best value
    filtered_data = cv_results_[cv_results_[f'param_{fixed_param}'] == best_value]

    # Pivot table for the heatmap
    pivot_table = filtered_data.pivot_table(
        values='final_accuracy',
        index=f'param_{varying_params[0]}',
        columns=f'param_{varying_params[1]}'
    )

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="coolwarm", cbar_kws={'label': 'Test Accuracy Score'})
    plt.title(f'Interaction of {varying_params[0]} and {varying_params[1]} \nwith {fixed_param} fixed at {best_value}')
    plt.xlabel(varying_params[1])
    plt.ylabel(varying_params[0])
    plt.show()

    fig.savefig(get_fig_filename(fixed_param, data_filename))


def get_fig_filename(param: str, source_data_filename: Path): # source_data_filename: Path
    """
    Generate filename for figure based on main (e.g. fixed) parameter and source data filename.

    :param param: Parameter to include in the filename.
    :param source_data_filename: Path to the source data file.
    :return: Path to the figure file.
    """
    return Path(source_data_filename.stem + 'PLT꞉' + param + '.png') # Path(source_data_filename.stem + 'PLT꞉' + param + '.png')


# %% Generate plots
cv_results = pd.read_csv(CV_RESULTS_PATH)

# Call the plotting function
#plot_parameter_effects(cv_results, PARAMS_OF_INTEREST, data_filename=CV_RESULTS_PATH)

# Call the function with example parameters
plot_interactive_effects(cv_results, FIXED_PARAM, VARYING_PARAMS, data_filename=CV_RESULTS_PATH)
