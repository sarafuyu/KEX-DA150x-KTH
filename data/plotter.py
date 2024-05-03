#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data plotting module

:Date: 2024-05-03
:Authors: Sara Rydell, Noah Hopkins

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Imports

# External imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For heatmap or advanced plotting

# %% Configuration
path: str = 'TP꞉nan_elimination_CL꞉17_KF꞉60_PR꞉0.2_RS꞉42_CF꞉SVC_DT꞉2024-05-03-044410.csv'
verbose = 2

# %% Load Data

cv_results = pd.read_csv(path)


# %% Plott functions

def plot_parameter_effects(cv_results, parameters):
    """
    Plot the effects of varying parameters on model performance.

    :param cv_results: (pd.DataFrame) The DataFrame containing the cross-validation results.
    :param parameters: (list of str) List of parameter names to plot.
    :return:
    """
    # Find the best parameters
    best_index = cv_results['rank_test_score'].idxmin()
    best_params = cv_results.loc[best_index, [f'param_{param}' for param in parameters]]

    # Loop over each parameter and generate a plot
    for param_name in parameters:
        full_param_name = f'param_{param_name}'

        # Filter data for plots
        mask = (cv_results.loc[:, [f'param_{p}' for p in parameters if p != param_name]] == best_params.drop(
            index=full_param_name
        )).all(axis=1)
        varying_param_data = cv_results[mask]
        varying_param_data = varying_param_data.drop_duplicates(subset=full_param_name)

        # Extract the values to plot
        x = varying_param_data[full_param_name]
        y = varying_param_data['mean_test_score']
        e = varying_param_data['std_test_score']

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(x, y, e, linestyle='--', marker='o', label=param_name)
        ax.set_xlabel(param_name)
        ax.set_ylabel('Mean Test Score')
        ax.set_title(f'Effect of {param_name} on Model Performance')

        # Set log scale for specific parameters
        if param_name in ['C', 'coef0', 'tol']:
            ax.set_xscale('log')

        # Set y-axis limits and labels if needed
        ax.set_ylim([min(y) - 0.05, max(y) + 0.05])  # Adjust based on data range
        ax.yaxis.set_major_locator(plt.MaxNLocator(10))  # Set the number of ticks on y-axis

        plt.legend()
        plt.show()

        fig.savefig(f'plots/test1_{param_name}')


def plot_interactive_effects(cv_results, fixed_param, varying_params, data_filename):
    """
    Plot the effects of varying two parameters on model performance while fixing another parameter at its best value.

    Args:
        cv_results (pd.DataFrame): The DataFrame containing the cross-validation results.
        fixed_param (str): The parameter to fix at its best value.
        varying_params (list of str): The two parameters to vary and visualize.
    """
    # Ensure only two varying parameters are specified
    if len(varying_params) != 2:
        raise ValueError("Exactly two varying parameters must be specified.")

    # Find the best parameter value to fix
    best_index = cv_results['rank_test_score'].idxmin()
    best_value = cv_results.loc[best_index, f'param_{fixed_param}']

    # Filter the DataFrame to include only rows where the fixed parameter is at its best value
    filtered_data = cv_results[cv_results[f'param_{fixed_param}'] == best_value]

    # Pivot table for the heatmap
    pivot_table = filtered_data.pivot_table(
        values='mean_test_score',
        index=f'param_{varying_params[0]}',
        columns=f'param_{varying_params[1]}'
    )

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="coolwarm", cbar_kws={'label': 'Mean Test Score'})
    plt.title(f'Interaction of {varying_params[0]} and {varying_params[1]} \nwith {fixed_param} fixed at {best_value}')
    plt.xlabel(varying_params[1])
    plt.ylabel(varying_params[0])
    plt.show()

    fig.savefig(f'plots/{get_fig_filename(fixed_param=fixed_param, source_data_filename=data_filename)}')

# %% Run plot functions

# Specify parameters of interest
params_of_interest = ['C', 'kernel', 'degree', 'coef0', 'tol']
# Call the plotting function
plot_parameter_effects(cv_results, params_of_interest)

# Call the function with example parameters
plot_interactive_effects(cv_results, 'kernel', ['C', 'degree'], data_filename=path)

def get_fig_filename(fixed_param, source_data_filename):
    return source_data_filename.split('.csv')[0] + 'PLT꞉' + fixed_param + '.png'

