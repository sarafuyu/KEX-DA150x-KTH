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


# %% Setup

START_TIME = datetime.now()
PROJECT_ROOT = Path(__file__).parents[1]


DATA_DIR = 'XGB-RFECV-binlog-feat-select-num-est꞉ALL'
DATA_FILE = '2024-05-11-041302__FeatureSelect__RFECV.pkl'

rfecv = joblib.load(PROJECT_ROOT / 'data' / 'results' / DATA_DIR / DATA_FILE)


breakpoint()
