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
from matplotlib.ticker import MaxNLocator # new

# Local imports
import utils


# %% Setup

SEED = utils.RANDOM_SEED  # get random seed
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CV_RESULTS_DIR = PROJECT_ROOT/'data'/'results'/'final-prestudy'


# %% Configuration

# Path to the cv_results_ csv file:
CV_RESULTS_PATH: Path = CV_RESULTS_DIR/'2024-06-02-223758__FeatureSelect꞉XGB-RFE-CV꞉cv_results_.csv' #2024-05-16-112335__cv_results.csv'

# Path to cleaned dataset
CLEANED_DATASET_PATH: Path = CV_RESULTS_DIR/'2024-06-02-223758__cleaned_data.csv'

# Path to the pickled dataset dictionary
PICKLED_DATASET_PATH: Path = CV_RESULTS_DIR/'2024-06-02-223758__FeatureSelect꞉XGB-RFE-CV_dataset_dict.pkl'

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

def plot_distribution(df, column, name):
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
            plt.title(f'Count Plot of {name}', fontsize=16)
            plt.xlabel(name, fontsize=13)
            plt.ylabel('Frequency', fontsize=13)
        else:
            # Use a histogram and boxplot for numeric columns with many unique values
            fig, axs = plt.subplots(1, 2, figsize=(15, 6))
            sns.histplot(column_data, kde=True, ax=axs[0], color='skyblue')
            axs[0].set_title(f'Histogram of {name}', fontsize=13)
            axs[0].set_xlabel(name)
            axs[0].set_ylabel('Frequency', fontsize=13)

            sns.boxplot(x=column_data, ax=axs[1], color='salmon')
            axs[1].set_title(f'Boxplot of {column}', fontsize=16)
            axs[1].set_xlabel(column)
            plt.tight_layout()
    else:
        # For categorical columns, use a count plot
        sns.countplot(x=column_data, palette='viridis')
        plt.title(f'Count Plot of {column}', fontsize=16)
        plt.xlabel(column)
        plt.ylabel('Frequency', fontsize=13)

    # Show the plot
    plt.show()

# %% Data distribution before data split
def hist_bar_plot(df, file, hist_col, hist_name, bar_col, bar_name):
    """
    Plots the distribution of specified columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    file (str): The save directory for final plot.
    hist_col (str): Name of df column to plot in histogram.
    hist_name (str): Name used in lables and the plot axis.
    bar_col (str): Name of df column to plot in bar plot.
    bar_name (str): Name used in lables and the plot axis.
    """

    # Extract data
    hist_data = df[hist_col] # floats
    bar_data = df[bar_col] # int: 0-34

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 6))

    # Use a histogram and boxplot for numeric columns with many unique values
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Histogram for 'Age'
    sns.histplot(hist_data, kde=True, bins=np.arange(int(hist_data.min()), int(hist_data.max()+2)), ax=axs[0], color='skyblue')
    axs[0].set_title(f'Distribution of {hist_name}', fontsize=16)
    axs[0].set_xlabel(f'Age (Years)', fontsize=13)
    axs[0].set_ylabel(f'Frequency', fontsize=13)
    axs[0].set_xticks(range(int(hist_data.min()), int(hist_data.max()) + 2))

    # Bar plot for 'NSAA'
    bar_freq = bar_data.value_counts().sort_index()
    sns.barplot(x=bar_freq.index, y=bar_freq.values, ax=axs[1], color='salmon')
    axs[1].set_title(f'Distribution of {bar_name} Scores', fontsize=16)
    axs[1].set_xlabel(f'NSAA Score', fontsize=13)
    axs[1].set_ylabel('')
    axs[1].set_xticks(range(0, 35))
    axs[1].set_yticks(range(0, bar_freq.max() + 2, 2))  # Ensure y-axis has even integer values
    axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure y-axis has integer values

    # Show and save the plot
    plt.tight_layout()
    plt.show()
    fig.savefig(file)

# %% Age distribution based on cut-off
def age_split_plot(df, file, col1, name1, col2, name2, cutoff):
    """
    Plots the distribution of specified columns in a DataFrame based on the binary classes.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    file (str): The save directory for final plot.
    col1 (str): Name of df column to plot in histogram.
    name1 (str): Name used in lables and the plot axis.
    col2 (str): Name of df column to plot in bar plot.
    name2 (str): Name used in lables and the plot axis.
    cutoff (int): Treshold used for binary classification of col2.
    """

    # Split data
    less_than_cutoff = [] # 1
    greater_equal_to_cutoff = [] # 0

    for idx, row in df.iterrows():
        if row[col2] < cutoff:
            less_than_cutoff.append(row[col1])
        else:
            greater_equal_to_cutoff.append(row[col1])

    # print(f"Less than cut-off len: {len(less_than_cutoff)}")
    # print(f"Equal or greater than cut-off len: {len(greater_equal_to_cutoff)}")

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 6))

    # Use a histogram and boxplot for numeric columns with many unique values
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    for (i, list) in enumerate([less_than_cutoff, greater_equal_to_cutoff]):
        # Histogram for 'Age'
        sns.histplot(list, kde=True, bins=np.arange(int(min(list)), int(max(list)+2)), ax=axs[i], color='skyblue')
        axs[i].set_xlabel(f'Age (Years)', fontsize=13)
        if i == 0:
            axs[i].set_ylabel(f'Frequency', fontsize=13)
        else:
            axs[i].set_ylabel('')
        axs[i].set_ylim([0,65])  # Achive same scale for both subplots
        axs[i].set_xticks(range(int(min(list)), int(max(list))+2))

    axs[0].set_title(f'First NSAA Category Age Distribution (NSAA $<$ 17)', fontsize=16)
    axs[1].set_title(f'Second NSAA Category Age Distribution (NSAA $\\geq$ 17)', fontsize=16)

    # Show and save the plot
    plt.tight_layout()
    plt.show()
    fig.savefig(file)

# %% Generate plots
cleaned_dataset = pd.read_csv(CLEANED_DATASET_PATH)
hist_bar_plot(cleaned_dataset, CV_RESULTS_DIR/'age_nsaa_dist.png','Age', 'Age', 'FT5', 'NSAA')
age_split_plot(cleaned_dataset, CV_RESULTS_DIR/'age_split_nsaa.png', 'Age', 'Age', 'FT5', 'NSAA', 17)

# Check the number of unique patients
unique_patients = cleaned_dataset['Participant_ID'].nunique()
print(f"Number of unique participants: {unique_patients}")

# Check the number of unique patients with disease status 0
unique_patients_with_disease_0 = cleaned_dataset[cleaned_dataset['Disease'] == 0]['Participant_ID'].nunique()
print(f"Number of unique Cnt participants with: {unique_patients_with_disease_0}")

# Load the cross-validation results
#cv_results = pd.read_csv(CV_RESULTS_PATH)

# Make all param prefixes the same
#cv_results = replace_column_prefix(cv_results, ['param_classifier__'],  'param_')

# Add prefix to the parameter names
#PARAMS_OF_INTEREST = [f'{PARAM_PREFIX}{param}' for param in PARAMS_OF_INTEREST]
#FIXED_PARAM = f'{PARAM_PREFIX}{FIXED_PARAM}'
#VARYING_PARAMS = [f'{PARAM_PREFIX}{param}' for param in VARYING_PARAMS]

# Sanitize the normalization column
#cv_results['param_normalizer'] = cv_results['param_normalizer'].str.split('(').str[0]

# Load cleaned dataset
#cleaned_dataset = pd.read_csv(CLEANED_DATASET_PATH)

# Load pickled dataset dictionary
#dataset_dict = joblib.load(PICKLED_DATASET_PATH)

# Plot distributions of the specified columns
#plot_distribution(cleaned_dataset, 'Age', 'Age')
#plot_distribution(cleaned_dataset, 'FT5', 'NSAA')
#plot_distribution(dataset_dict['dataset'],  'FT5', 'NSAA')