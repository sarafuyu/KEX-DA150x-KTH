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
import numpy as np  # needed for np.linspace/logspace in config
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

DATA_DIR = 'XGB-RFECV-binlog-feat-select-num-est꞉ALL-Cut꞉17-Age'
DATA_FILE = '2024-05-17-154927__dataset_dict.pkl'

d = joblib.load(PROJECT_ROOT / 'data' / 'results' / DATA_DIR / DATA_FILE)


# Concatenate the training and testing data
X = pd.concat([d['X_training'], d['X_testing']], axis=0)
X_prot = X.iloc[:, 1:]
X_age = X.iloc[:, 0]
# X_test_prot = d['X_testing'].iloc[:, 1:]
# X_train_prot = d['X_training'].iloc[:, 1:]
# X_train_age = d['X_training'].iloc[:, 0]
# X_test_age = d['X_testing'].iloc[:, 0]

# print_missing_values(X=X)
# print_missing_values(X=X_train_prot)
# print_missing_values(X=X_test_prot)
# print_missing_values(X=X_train_age)
# print_missing_values(X=X_test_age)
print_missing_values(X=X_prot)
print_missing_values(X=X_age)

breakpoint()
