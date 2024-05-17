#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Plotting

This script is run separately from the main script to generate distribution plots.

:Date: 2024-05-17
:Authors: Sara Rydell, Noah Hopkins

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Imports

# Standard library imports
from pathlib import Path
from collections.abc import Sequence

import joblib
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

# Path to cleaned dataset
CLEANED_DATASET_PATH: Path = CV_RESULTS_DIR/'2024-05-16-112335__cleaned_data.csv'

# Path to the pickled dataset dictionary
PICKLED_DATASET_PATH: Path = CV_RESULTS_DIR/'2024-05-16-112335__dataset_dict.pkl'

# Verbosity level:
VERBOSE: int = 2  # Unused at the moment

# Specify the column name for the test score
TEST_SCORE_COLUMN_NAME: str = 'final_accuracy'

# Specify parameters of interest
PARAMS_OF_INTEREST = ['C', 'degree', 'coef0', 'tol']
FIXED_PARAM = 'kernel'  # 'kernel'
KERNEL = 'poly'  # 'rbf', 'poly', or 'sigmoid'  # TODO: Implement filtering based on kernel
FIXED_NORMALIZATION = 'StandardScaler(copy=False)'  # 'param_normalizer'
VARYING_PARAMS = ['coef0', 'tol']  # ['C', 'degree']

PARAM_PREFIX = 'param_'  # Prefix for the parameter columns in the cv_results_ DataFrame

# Plot x-axis scale
SCALE_X = 'log'  # 'linear' or 'log'


# %% Plot functions

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

def plot_distribution(df, column):
    """
    Plots the distribution of a specified column in a DataFrame based on its characteristics.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The name of the column to plot.
    """
    # Identify the column type
    column_data = df[column]
    column_type = column_data.dtype

    # Convert the column to numeric if possible, otherwise leave as is
    if pd.api.types.is_numeric_dtype(column_data):
        column_data = pd.to_numeric(column_data, errors='coerce')
        numeric = True
    else:
        numeric = False

    # Drop NaN values for plotting
    column_data = column_data.dropna()

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 6))

    if numeric:
        unique_values = column_data.nunique()
        if unique_values < 20:
            # If the number of unique numeric values is less than 20, use a count plot
            sns.countplot(x=column_data, palette='viridis')
            plt.title(f'Count Plot of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
        else:
            # Use a histogram and boxplot for numeric columns with many unique values
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            sns.histplot(column_data, kde=True, ax=axs[0], color='skyblue')
            axs[0].set_title(f'Histogram of {column}')
            axs[0].set_xlabel(column)
            axs[0].set_ylabel('Frequency')

            sns.boxplot(x=column_data, ax=axs[1], color='salmon')
            axs[1].set_title(f'Boxplot of {column}')
            axs[1].set_xlabel(column)
            plt.tight_layout()
    else:
        # For categorical columns, use a count plot
        sns.countplot(x=column_data, palette='viridis')
        plt.title(f'Count Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

    # Show the plot
    plt.show()


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

# Load cleaned dataset
cleaned_dataset = pd.read_csv(CLEANED_DATASET_PATH)

# Load pickled dataset dictionary
dataset_dict = joblib.load(PICKLED_DATASET_PATH)

# Plot distributions of the specified columns
plot_distribution(cleaned_dataset, 'Age')
plot_distribution(cleaned_dataset, 'FT5')
plot_distribution(dataset_dict['dataset'],  'FT5')

breakpoint()