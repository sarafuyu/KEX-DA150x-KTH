#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive loader for exploring pickled objects

:Date: 2024-05-16
:Authors: Sara Rydell, Noah Hopkins

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Imports

## Standard library imports
import sys
import logging
import joblib
from collections.abc import Sequence, Callable
from copy import deepcopy
from datetime import datetime
from pathlib import Path

## External library imports
# noinspection PyUnresolvedReferences
import numpy as np  # needed for np.linspace/printspace in config
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.impute import MissingIndicator
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from model_selection._search import CustomCvGridSearch  # noqa


# %% Configuration

VERBOSE: int = 2  # Verbosity level
SEED: int = 42    # Random seed


# %% Local imports

# The utils module need to be imported first to set verbosity level and random seed
import utils
utils.VERBOSITY_LEVEL = VERBOSE  # Set verbosity level for all modules
utils.RANDOM_SEED = SEED         # Set random seed for all modules

# Import the other modules (they import utils to get verbosity level and random seed)
import cleaning
import imputation
import normalization
import features
import classifier

def print_missing_values(X=None):
    if X is None:
        print("No DataFrame provided.")
        return

    # Get the total number of missing values
    total_missing = X.isnull().sum().sum()

    # Get the total number of cells in the DataFrame
    total_cells = X.size

    # Calculate the percentage of missing values
    percent_missing = (total_missing / total_cells) * 100

    # Get the variable name of X
    var_name = [k for k, v in globals().items() if v is X][0]

    # Print the results
    print(f"Total number of missing values for {var_name}: {total_missing}")
    print(f"Percentage of missing values for {var_name}: {percent_missing:.2f}%")


# %% Setup


START_TIME = datetime.now()
PROJECT_ROOT = Path(__file__).parents[1]

DATA_DIR = 'final-prestudy'
DATA_FILE = '2024-06-02-223758__FeatureSelect꞉XGB-RFE-CV_dataset_dict.pkl'


# Load the pickled object
data_path = PROJECT_ROOT / 'data' / 'results' / DATA_DIR
dataset_dict = joblib.load(data_path / DATA_FILE)


# Try print missingness in train feature columns after selection
try:
    print("Printing missingness in training feature columns after selection")
    num_missing = dataset_dict['X_training'].isna().sum()
    missingness = num_missing / dataset_dict['X_training'].shape[0]
    mean_vals = dataset_dict['X_training'].mean()
    med_vals = dataset_dict['X_training'].median()
    var_vals = dataset_dict['X_training'].var()
    min_vals = dataset_dict['X_training'].min()
    max_vals = dataset_dict['X_training'].max()
    combined_df = pd.concat([num_missing, missingness, med_vals, mean_vals, var_vals, min_vals, max_vals], axis=1)
    combined_df.index.name = 'Feature_Name'
    combined_df.columns = ['Num_Missing', 'Percentage_Missing', 'Median', 'Mean', 'Variance', 'Min', 'Max']
    combined_df.to_csv(
        data_path / (START_TIME.strftime("%Y-%m-%d-%H%M%S") + '__X_train_missingness_after_select.csv')
    )
    print(f"Mean num missing in feature columns after selection {num_missing.mean()}")
    print(f"Mean missingness in feature columns after selection {missingness.mean()}")
    print(f"Mean Age: {dataset_dict['X_training']['Age'].mean()}")
    print(f"Variance Age: {dataset_dict['X_training']['Age'].var()}")
except Exception as e:
    print(f"Error when trying to print missingness in feature columns after selection: {e}")

# Try print missingness in test feature columns after selection
try:
    print("Printing missingness in testing feature columns after selection")
    num_missing = dataset_dict['X_testing'].isna().sum()
    missingness = num_missing / dataset_dict['X_training'].shape[0]
    mean_vals = dataset_dict['X_testing'].mean()
    med_vals = dataset_dict['X_testing'].median()
    var_vals = dataset_dict['X_testing'].var()
    min_vals = dataset_dict['X_testing'].min()
    max_vals = dataset_dict['X_testing'].max()
    combined_df = pd.concat([num_missing, missingness, med_vals, mean_vals, var_vals, min_vals, max_vals], axis=1)
    combined_df.index.name = 'Feature_Name'
    combined_df.columns = ['Num_Missing', 'Percentage_Missing', 'Median', 'Mean', 'Variance', 'Min', 'Max']
    combined_df.to_csv(
        data_path / (START_TIME.strftime("%Y-%m-%d-%H%M%S") + '__X_test_missingness_after_select.csv')
    )
    print(f"Mean num missing in feature columns after selection {num_missing.mean()}")
    print(f"Mean missingness in feature columns after selection {missingness.mean()}")
    print(f"Mean Age: {dataset_dict['X_testing']['Age'].mean()}")
    print(f"Variance Age: {dataset_dict['X_testing']['Age'].var()}")
except Exception as e:
    print(f"Error when trying to print missingness in feature columns after selection: {e}")

# Concatenate the training and testing data
X = pd.concat([dataset_dict['X_training'], dataset_dict['X_testing']], axis=0)

# Try print missingness in feature columns after selection
try:
    print("Printing missingness in testing feature columns after selection")
    num_missing = X.isna().sum()
    missingness = num_missing / dataset_dict['X_training'].shape[0]
    mean_vals = X.mean()
    med_vals = X.median()
    var_vals = X.var()
    min_vals = X.min()
    max_vals = X.max()
    combined_df = pd.concat([num_missing, missingness, med_vals, mean_vals, var_vals, min_vals, max_vals], axis=1)
    combined_df.index.name = 'Feature_Name'
    combined_df.columns = ['Num_Missing', 'Percentage_Missing', 'Median', 'Mean', 'Variance', 'Min', 'Max']
    combined_df.to_csv(
        data_path / (START_TIME.strftime("%Y-%m-%d-%H%M%S") + '__X_missingness_after_select.csv')
    )
    print(f"Mean num missing in feature columns after selection {num_missing.mean()}")
    print(f"Mean missingness in feature columns after selection {missingness.mean()}")
    print(f"Mean Age: {X['Age'].mean()}")
    print(f"Variance Age: {X['Age'].var()}")
except Exception as e:
    print(f"Error when trying to print missingness in feature columns after selection: {e}")