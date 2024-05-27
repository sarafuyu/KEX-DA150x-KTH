#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main file for the project. It will be used to run the project.

:Date: 2024-05-01
:Authors: Sara Rydell, Noah Hopkins

Currently supported: Binary classification using SVM classifier.

TODOs
=====

------ MVP for today ------
TODO(priority 5): Refactor main.py pipeline to make it more DRY (construct grid_param in main and 
                  pass to a function that calls the various grid search functions for the various 
                  models depending on the different
                  dataset dict attributes)
--------------------------------

# Cutoff
Write in thesis: we choose 17 since it represents av average score of 1 on each NSAA item.
o Score of 2 = ‘Normal’ – no obvious modification of activity
o Score of 1 = Modified method but achieves goal with no physical assistance
o Score of 0 = Unable to achieve goal independently
The class [0, 17) would be the class of patients who are unable to on average achieve the goals independently.
The class [17, 34] would be the class of patients who are able to  on average achieve the goals independently.

# pre-study:
# see how iterative vs simple imputer affects the results (compare time and accuracy)
# see how feature selection affects the results (try a few different values, pick and freeze the best one)
# write in the report that we tried with different kernels and that the best one was poly
# write in the report that we tried with different tols and that they were all the same under a certain threshold
# Main study:
# See how C and coeff0 affects the final accuracy (detailed heatmap etc)

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

# Local imports
# noinspection PyUnresolvedReferences
from model_selection._search import CustomCvGridSearch  # noqa


# %% Setup

START_TIME = datetime.now()
PROJECT_ROOT = Path(__file__).parents[1]


# %% Configuration

# **********----------------------------------------------------------------------------********** #
# |                                        ~~ General ~~                                         | #
# **********----------------------------------------------------------------------------********** #

# Set to True to use the debug configuration from the debug_config.py file.
DEBUG: bool = False

# ----------
# Verbosity
# ----------

# The higher, the more verbose. Can be 0, 1, 2, 3, or 4.
VERBOSE: int = 1
# Level 0: No prints.
# Level 1: Only essential prints. Minimal prints for each individual hyperparameter/data set.
# Level 2: Essential prints and additional prints for each hyperparameter config/data set.
# Level 3: All prints. Maybe some previews of data.
# Level 4: All prints and all previews of data.

# --------
# Logging
# --------

# Set to True to log the output to a file.
LOGGER: bool = True
# Set the name of the log file.
LOG_FILE: Path = PROJECT_ROOT/'out'/('pipline-log-DT꞉'+START_TIME.strftime("%Y-%m-%d-%H%M%S")+'.log')

# -----------
# Randomness
# -----------

# Randomness seed for all methods that take a seed in all files of the project
SEED: int = 42  # 42

# ----------------
# Data Extraction
# ----------------

# Path to dataset
DATA_FILE: Path = PROJECT_ROOT/'data'/'dataset'/'normalised_data_all_w_clinical_kex_20240321.csv'


# **********----------------------------------------------------------------------------********** #
# |                                 ~~ Missing Indicators ~~                                     | #
# **********----------------------------------------------------------------------------********** #

ADD_INDICATORS: bool = False

# Values to consider as missing values
MISSING_VALUES_INDICATOR: float = np.nan

# Whether to create missing indicators for all features or only for features with some missingness
FEATURES_INDICATOR: str = 'missing-only'

# Whether to output a sparse matrix for the missing indicators
SPARSE_INDICATOR: bool = False


# **********----------------------------------------------------------------------------********** #
# |                                   ~~ Train-test split ~~                                     | #
# **********----------------------------------------------------------------------------********** #

# For detailed configuration for each feature selection mode, see features.py
TEST_PROPORTION: float = 0.2  # TODO: full cross validation settings

# Column index to start from. Will split the selected X and y data, into the two data subsets:
#   X_train, y_train,
#   X_test, y_test,
# where X_train and X_test are the input variables and y_train and y_test are the target variables.
X_START_COLUMN_IDX: int = 11  # X columns are from this column index and onwards TODO: check if 11 is the correct index.
Y_COLUMN_LABEL: str = 'FT5'   # y column label


# **********----------------------------------------------------------------------------********** #
# |                        ~~ Re-categorization of Northstar score (y) ~~                        | #
# **********----------------------------------------------------------------------------********** #
# Northstar score (y) is originally a discrete variable on [0,34]
#
# Bin Northstar score into a categorical variable.
# For N cutoffs, the data is divided into N+1 classes.
# For binary classification, use one cutoff, [a].
#
# Example:
# CUTOFFS=[a] will create a binary variable with classes:
#   df.iloc[:, 0:(a - 1)] == 1
#   df.iloc[:, a:       ] == 0
# i.e. the variable is 1 if (x < a) else 0.
#
CUTOFFS: Sequence[int] | bool = [17]  # False  # [17]  # break into classes at 17
COLUMN_TO_CATEGORIZE: str = 'FT5'


# **********----------------------------------------------------------------------------********** #
# |                              ~~ Feature selection (XGB-RFECV) ~~                             | #
# **********----------------------------------------------------------------------------********** #

# Use XGB feature selection
SELECT_XGB: bool = True

# Stop the pipeline after feature selection
STOP_AFTER_FEATURE_SELECTION: bool = False

# Use precomputed XGB selected data dict
# If set, will skip: cleaning, adding imputer objects, adding indicators, categorizing y, splitting data
# and will start with imputing the data using the precomputed imputer objects.
PRECOMPUTED_XGB_SELECTED_DATA: Path | None = None  # PROJECT_ROOT/'data'/'results'/'XGB-RFECV-binlog-feat-select-num-est꞉ALL'/'2024-05-11-041302__FeatureSelect__XGB-RFE-CV_dataset_dict.pkl'  # 'XGB-RFECV-binlog-feat-select-num-est꞉ALL-Cut꞉17-Age'/'2024-05-17-154927__dataset_dict.pkl'

# 37 specific features were found to be the best number of features using XGB feature selection.

# XGB-RFECV Config
N_ESTIMATORS_XGB = -1
VERBOSITY_XGB = 1
USE_LABEL_ENCODER_XGB = False
VALIDATE_PARAMETERS_XGB = True
MISSING_XGB = np.nan
OBJECTIVE_XGB = 'binary:logistic'
N_JOBS_XGB = 6
N_JOBS_RFECV = 2
SCORING_RFECV = 'accuracy'  # 'f1', 'roc_auc', 'accuracy', 'precision', 'recall'
CV_RFE = 5
MIN_FEATURES_TO_SELECT_XGB = 1
STEP_XGB = 1


# **********----------------------------------------------------------------------------********** #
# |                                     ~~ Data imputation ~~                                    | #
# **********----------------------------------------------------------------------------********** #

# Pick imputation modes to use:
SIMPLE_IMPUTER: bool = False
ITERATIVE_IMPUTER: bool = True
KNN_IMPUTER: bool = False
NAN_ELIMINATION: bool = False
NO_IMPUTATION: bool = False
SPARSE_NO_IMPUTATION: bool = False  # Note: if `True`, `NO_IMPUTATION` must be set to `True`.

# -----------------------------
# Simple imputer configuration
# -----------------------------

# Strategy for imputing missing values
STRATEGY_SIMPLE_IMP: Sequence[str] = ['mean']  # ["mean", "median", "most_frequent"]
# Add indicator for missing values
ADD_INDICATOR_SIMPLE_IMP: bool = True

# Should always be True, since the implementation expects a copy of the data
COPY_SIMPLE_IMP: bool = True

# --------------------------------
# Iterative imputer configuration
# --------------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Estimator, e.g. a BayesianRidge() object or an estimator object from scikit-learn.
# Can probably be customized, but leave default for now.
# For future type hints, see: https://stackoverflow.com/a/60542986/6292000
ESTIMATOR_CRITERION_ITER_IMP = 'squared_error'  # 'friedman_mse'  # 'squared_error'
ESTIMATOR_ITER_IMP = [RandomForestRegressor(
    criterion=ESTIMATOR_CRITERION_ITER_IMP, max_depth=None, min_samples_split=2, min_samples_leaf=1,
    min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True,
    oob_score=r2_score, n_jobs=7, random_state=SEED, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None,
    monotonic_cst=None,
)]  # TODO: Try more: , DecisionTreeRegressor(random_state=SEED), RandomForestRegressor(random_state=SEED)
# Maximum number of imputation rounds to perform. The imputer will stop iterating after this many iterations.
MAX_ITER_ITER_IMP: int = 400  # try low number of iterations first, see if converges, then try higher numbers
TOL_ITER_IMP: float = 0.0175  # might need to adjust
# Number of other features to use to estimate the missing values of each feature column.
# None means all features, which might be too many.
N_NEAREST_FEATURES_ITER_IMP: Sequence[int | None] = [None]  # [15]  # [5, 10, 20, 50, None]  # [10, 100, 500, None]
INITIAL_STRATEGY_ITER_IMP: Sequence[str] = ["mean"]  # ["mean", "median", "most_frequent", "constant"]
IMPUTATION_ORDER_ITER_IMP: Sequence[str] = ["ascending"]  # Default, alternatives: ["ascending", "descending" "random"]
ADD_INDICATOR_ITER_IMP: bool = False  # Add indicator for missing values
# ascending: From the features with the fewest missing values to those with the most
MIN_VALUE_ITER_IMP: str | int = 'stat'  # no fea tures have negative values, adjust tighter for prot intensities?
MAX_VALUE_ITER_IMP: str | int = 'stat'
VERBOSE_ITER_IMP: int = 3

# Use precomputed imputed dataset
# Leave as None to use the imputed dataset generated in the pipeline
# Note that PRECOMPUTED_ITERATIVE_IMPUTED_DF is the full un-imputed dataset that was used to generate the imputed dataset
PRECOMPUTED_ITERATIVE_IMPUTED_X_DATA: Path | None = None  # PROJECT_ROOT / 'data' / 'results' / 'IterativeImputer-RFR-tol-0009-iter-131' / '2024-05-13-231235__IterativeImputer_X_imputed.csv'
PRECOMPUTED_ITERATIVE_IMPUTED_DF: Path | None = None

# Stop the pipeline after imputation
STOP_AFTER_IMPUTATION: bool = False


# --------------------------
# KNN imputer configuration
# --------------------------

# Missing values to impute
MISSING_VALUES_KNN_IMP = np.nan
# Initial span of neighbours considering dataset size
N_NEIGHBOURS_KNN_IMP: Sequence[int] = [5, 10, 20, 50]
# default='uniform', callable has potential for later fitting
WEIGHTS_KNN_IMP: Sequence[str] = ['distance']  # 'uniform'
METRIC_KNN_IMP: Sequence[str] = ['nan_euclidean']
ADD_INDICATOR_KNN_IMP = False
KEEP_EMPTY_FEATURES_KNN_IMP = False

# Should always be True, since the implementation expects a copy of the data
COPY_KNN_IMP = True

# ------------------------------
# NaN elimination configuration
# ------------------------------

# If True, drop all columns with NaN values. If False, drop rows with NaN values.
DROP_COLS_NAN_ELIM = True


# **********----------------------------------------------------------------------------********** #
# |                                   ~~ Data normalization ~~                                   | #
# **********----------------------------------------------------------------------------********** #

def identity(x): return x


# Normalization modes to try in Grid search. Can be 'None', 'StandardScaler', 'MinMaxScaler'.
NORMALIZATION_MODES_PARAMS = [
    StandardScaler(copy=False, with_mean=True, with_std=True),
    # MinMaxScaler(feature_range=(0, 1), copy=False, clip=False),
    # FunctionTransformer(identity, validate=True), # No normalization
]

# First column to normalize. Will normalize all columns from this index and onwards.
FIRST_COLUMN_TO_NORMALIZE: int = 11
# We only normalize the antibody/protein intensity columns (cols 11 and up). Age, disease, FTs not normalized.

# TODO: check if this is the correct index. Should it be 10 instead?
# TODO: instead iterate over the column labels until we reach the first protein intensity column
#       then normalize all columns from there and onwards instead of hardcoding the index.


# **********----------------------------------------------------------------------------********** #
# |                              ~~ Model training & fitting ~~                                  | #
# **********----------------------------------------------------------------------------********** #

# ------------
# Grid Search
# ------------

# Set the verbosity level for the grid search printouts that are not logged.
GRID_SEARCH_VERBOSITY: int = 0
# Number of cross-validation folds
K_CV_FOLDS: int = 5
# Calculate final accuracy for all models
CALC_FINAL_SCORES: bool = True
# Number of jobs to run in parallel, -1 means using all processors, 1 is the default
N_JOBS_GRID_SEARCH: int = 7


# ---------------
# SVM Classifier
# ---------------

# Enable SVC
SVC = True

# Hyperparameters:            # np.logspace(start, stop, num=50)
C_PARAMS_SVC: Sequence[float] = sorted(np.unique(np.concatenate([
    np.logspace(start=-7, stop=5, num=7+5+1, base=10),  # 10^-7 to 10^5
    np.logspace(start=-3, stop=-2, num=5, base=10)[1:-1],
    np.logspace(start=-2, stop=-1, num=5, base=10)[1:-1],
    np.logspace(start=-1, stop=0, num=5, base=10)[1:-1],
    np.logspace(start=0, stop=1, num=5, base=10)[1:-1],
    np.logspace(start=1, stop=2, num=5, base=10)[1:-1],
    np.logspace(start=2, stop=3, num=5, base=10)[1:-1],
    np.logspace(start=3, stop=4, num=5, base=10)[1:-1],
    np.logspace(start=4, stop=5, num=5, base=10)[1:-1],
])))
KERNEL_PARAMS_SVC: Sequence[str] = ['poly']  # 'linear', 'rbf', 'precomputed'
DEGREE_PARAMS_SVC: Sequence[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
GAMMA_PARAMS_SVC: Sequence[str] = ['auto']  # scale not needed since normalization X_var
COEF0_PARAMS_SVC = sorted(np.unique(np.concatenate([
    -np.logspace(start=2, stop=3, num=5, base=10)[1:-1],
    -np.logspace(start=1, stop=2, num=5, base=10)[1:-1],
    -np.logspace(start=0, stop=1, num=5, base=10)[1:-1],
    -np.logspace(start=-1, stop=0, num=5, base=10)[1:-1],
    -np.logspace(start=-2, stop=-1, num=5, base=10)[1:-1],
    -np.logspace(start=-3, stop=-2, num=5, base=10)[1:-1],
    -np.logspace(start=-3, stop=3, num=3+3+1, base=10),  # -10^-3 to -10^3
    np.linspace(0, 0, 1),              # 0
    np.logspace(start=-3, stop=3, num=3+3+1, base=10),  # 10^-3 to  10^3
    np.logspace(start=-3, stop=-2, num=5, base=10)[1:-1],
    np.logspace(start=-2, stop=-1, num=5, base=10)[1:-1],
    np.logspace(start=-1, stop=0, num=5, base=10)[1:-1],
    np.logspace(start=0, stop=1, num=5, base=10)[1:-1],
    np.logspace(start=1, stop=2, num=5, base=10)[1:-1],
    np.logspace(start=2, stop=3, num=5, base=10)[1:-1],
])))
SHRINKING_PARAMS_SVC: Sequence[bool] = [True]
PROBABILITY_SVC: Sequence[bool] = [False]
TOL_PARAMS_SVC: Sequence[float] = [0.001]  # np.linspace(0.01, 0.0001, 10)  # np.linspace(0.01, 0.0001, 10)
CACHE_SIZE_PARAMS_SVC: Sequence[int] = [500]
CLASS_WEIGHT_SVC: dict | None = None
VERB_SVC: int = VERBOSE
MAX_ITER_PARAMS_SVC: Sequence[int] = [10_000_000]  # [-1]
DECISION_FUNCTION_PARAMS_SVC: Sequence[str] = ['ovr']  # only ovo if multi class
BREAK_TIES_PARAMS_SVC: Sequence[bool] = [False]

#### OLD SVC PARAMS from 2024-05-21
# # Hyperparameters:            # np.logspace(start, stop, num=50)
# C_PARAMS_SVC: Sequence[float] = [0.000_0001, 0.000_001, 0.000_01, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # np.linspace(0.00001, 3, num=10)  # np.linspace(0.001, 100, num=60)
# KERNEL_PARAMS_SVC: Sequence[str] = ['poly', 'sigmoid', 'rbf']  # 'linear', 'rbf', 'precomputed'
# DEGREE_PARAMS_SVC: Sequence[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# GAMMA_PARAMS_SVC: Sequence[str] = ['auto']  # scale not needed since normalization X_var
# COEF0_PARAMS_SVC: Sequence[float] = [-100.0, -10.0, -1.0, -0.1, -0.01, -0.001, 0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  # np.linspace(-2, 4, num=10)  # np.linspace(-10, 10, num=60)
# SHRINKING_PARAMS_SVC: Sequence[bool] = [True]
# PROBABILITY_SVC: Sequence[bool] = [False]
# TOL_PARAMS_SVC: Sequence[float] = [0.01]  # np.linspace(0.01, 0.0001, 10)  # np.linspace(0.01, 0.0001, 10)
# CACHE_SIZE_PARAMS_SVC: Sequence[int] = [500]
# CLASS_WEIGHT_SVC: dict | None = None
# VERB_SVC: int = VERBOSE
# MAX_ITER_PARAMS_SVC: Sequence[int] = [1_000_000,]  # [-1]
# DECISION_FUNCTION_PARAMS_SVC: Sequence[str] = ['ovr']  # only ovo if multi class
# BREAK_TIES_PARAMS_SVC: Sequence[bool] = [False]

#### OLD SVC PARAMS
# # Hyperparameters:            # np.logspace(start, stop, num=50)
# C_PARAMS_SVC: Sequence[float] = [0.0000_0001, 0.000_001, 0.0001, 0.01, 1.0, 10.0]  # np.linspace(0.00001, 3, num=10)  # np.linspace(0.001, 100, num=60)
# KERNEL_PARAMS_SVC: Sequence[str] = ['poly', 'sigmoid', 'rbf']  # 'linear', 'rbf', 'precomputed'
# DEGREE_PARAMS_SVC: Sequence[int] = [1, 2, 3, 4, 5, 6, 7, 8]
# GAMMA_PARAMS_SVC: Sequence[str] = ['auto']  # scale not needed since normalization X_var
# COEF0_PARAMS_SVC: Sequence[float] = [-1000.0, -10.0, -1.0, -0.1, -0.01, 0.0, 0.01, 0.1, 1.0, 10.0, 1000.0]  # np.linspace(-2, 4, num=10)  # np.linspace(-10, 10, num=60)
# SHRINKING_PARAMS_SVC: Sequence[bool] = [True]
# PROBABILITY_SVC: Sequence[bool] = [False]
# TOL_PARAMS_SVC: Sequence[float] = [0.00001, 0.0001, 0.001, 0.01, 0.1,]  # np.linspace(0.01, 0.0001, 10)  # np.linspace(0.01, 0.0001, 10)
# CACHE_SIZE_PARAMS_SVC: Sequence[int] = [500]
# CLASS_WEIGHT_SVC: dict | None = None
# VERB_SVC: int = VERBOSE
# MAX_ITER_PARAMS_SVC: Sequence[int] = [1_000_000,]  # [-1]
# DECISION_FUNCTION_PARAMS_SVC: Sequence[str] = ['ovr']  # only ovo if multi class
# BREAK_TIES_PARAMS_SVC: Sequence[bool] = [False]

# RETURN_TRAIN_SCORE_SVC:
# Include scores in `cv_results_`. Computing training scores is used to get insights on how different parameter settings
# impact the overfitting/underfitting trade-off. However, computing the scores on the training set can be
# computationally expensive and is not strictly required to select the parameters that yield the best generalization
# performance.
RETURN_TRAIN_SCORE_SVC: bool = True
GRID_SEARCH_SCORING_SVC: str | list[str] = ['roc_auc', 'accuracy', 'f1', 'precision', 'recall']


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


# %% Logging

# Set up logging
handlers = []
if LOGGER:
    file_handler = logging.FileHandler(filename=LOG_FILE, encoding="utf-8")  # TODO: generate unique filename
    handlers.append(file_handler)
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers.append(stdout_handler)

# Configure logging level and format
logging.basicConfig(level=logging.DEBUG,
                    format=('[%(asctime)s] '
                            '{%(filename)-13s:%(lineno)-4d} : '
                            '%(levelname)s | '
                            '%(message)s'),
                    handlers=handlers)

# Suppress matplotlib warning logs
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Get the logger
log = logging.getLogger('LOGGER_NAME').info

if VERBOSE:
    log("======================= PIPELINE LOG ========================= |")
    log("Date: " + START_TIME.strftime("%Y-%m-%d %H:%M:%S"))
    log(f"Debug mode: {DEBUG}")
if DEBUG:
    # Import and unpack pipeline debug config from debug_config.py
    from debug_config import pipeline_debug_config

    for varname, value in pipeline_debug_config.items():
        globals()[varname] = value  # Overwrite variables in this script with those from the debug config

pipeline_config = {
    'DEBUG':                         DEBUG,
    'SEED':                          SEED,
    'VERBOSE':                       VERBOSE,
    'DATA_FILE':                     DATA_FILE,
    'SIMPLE_IMPUTER':                SIMPLE_IMPUTER,
    'ITERATIVE_IMPUTER':             ITERATIVE_IMPUTER,
    'KNN_IMPUTER':                   KNN_IMPUTER,
    'NAN_ELIMINATION':               NAN_ELIMINATION,
    'NO_IMPUTATION':                 NO_IMPUTATION,
    'SPARSE_NO_IMPUTATION':          SPARSE_NO_IMPUTATION,
    'ADD_INDICATOR_SIMPLE_IMP':      ADD_INDICATOR_SIMPLE_IMP,
    'COPY_SIMPLE_IMP':               COPY_SIMPLE_IMP,
    'STRATEGY_SIMPLE_IMP':           STRATEGY_SIMPLE_IMP,
    'ESTIMATOR_ITER_IMP':            ESTIMATOR_ITER_IMP,
    'MAX_ITER_ITER_IMP':             MAX_ITER_ITER_IMP,
    'TOL_ITER_IMP':                  TOL_ITER_IMP,
    'ADD_INDICATOR_ITER_IMP':        ADD_INDICATOR_ITER_IMP,
    'INITIAL_STRATEGY_ITER_IMP':     INITIAL_STRATEGY_ITER_IMP,
    'N_NEAREST_FEATURES_ITER_IMP':   N_NEAREST_FEATURES_ITER_IMP,
    'IMPUTATION_ORDER_ITER_IMP':     IMPUTATION_ORDER_ITER_IMP,
    'MIN_VALUE_ITER_IMP':            MIN_VALUE_ITER_IMP,
    'MAX_VALUE_ITER_IMP':            MAX_VALUE_ITER_IMP,
    'N_NEIGHBOURS_KNN_IMP':          N_NEIGHBOURS_KNN_IMP,
    'WEIGHTS_KNN_IMP':               WEIGHTS_KNN_IMP,
    'METRIC_KNN_IMP':                METRIC_KNN_IMP,
    'VERBOSE_ITER_IMP':              VERBOSE_ITER_IMP,
    'DROP_COLS_NAN_ELIM':            DROP_COLS_NAN_ELIM,
    'FIRST_COLUMN_TO_NORMALIZE':     FIRST_COLUMN_TO_NORMALIZE,
    'CUTOFFS':                       CUTOFFS,
    'COLUMN_TO_CATEGORIZE':          COLUMN_TO_CATEGORIZE,
    'TEST_PROPORTION':               TEST_PROPORTION,
    'X_START_COLUMN_IDX':            X_START_COLUMN_IDX,
    'ADD_INDICATORS':                ADD_INDICATORS,
    'Y_COLUMN_LABEL':                Y_COLUMN_LABEL,
    'SELECT_XGB':                    SELECT_XGB,
    'SVC':                           SVC,
    'START_TIME':                    START_TIME,
    'PRECOMPUTED_XGB_SELECTED_DATA': PRECOMPUTED_XGB_SELECTED_DATA,
    'STOP_AFTER_FEATURE_SELECTION':  STOP_AFTER_FEATURE_SELECTION,
    'PRECOMPUTED_IMPUTED_X':         PRECOMPUTED_ITERATIVE_IMPUTED_X_DATA,
    'STOP_AFTER_IMPUTATION':         STOP_AFTER_IMPUTATION,
    'N_JOBS_GRID_SEARCH':            N_JOBS_GRID_SEARCH,
}


# %% Load Data

# Load the data
dataset_dicts = []
dataset = pd.read_csv(DATA_FILE)


# %% Data Cleaning

# Clean and preprocess the data
verbose_cleaning = 0 if (PRECOMPUTED_XGB_SELECTED_DATA or PRECOMPUTED_ITERATIVE_IMPUTED_X_DATA) else VERBOSE
dataset = cleaning.clean_data(dataset, verbose=verbose_cleaning, log=log, date=START_TIME, dataset_path=DATA_FILE)
original_dataset = deepcopy(dataset)


# %% Generate new XGB selected data

if not (PRECOMPUTED_XGB_SELECTED_DATA or PRECOMPUTED_ITERATIVE_IMPUTED_X_DATA):
    if VERBOSE:
        log("Data loaded and cleaned successfully.\n")

    # Add NaN-eliminated and un-imputed datasets
    if NAN_ELIMINATION:
        dataset_dicts = dataset_dicts + imputation.eliminate_nan(dataset, drop_cols=DROP_COLS_NAN_ELIM)
    if NO_IMPUTATION:
        dataset_dicts = dataset_dicts + imputation.no_imputer(dataset, copy=True)


    # %% Sparse Data Handling

    # TODO: might not work with the current implementation, we can probably remove this since we don't use it
    if SPARSE_NO_IMPUTATION:
        dataset_dicts = dataset_dicts + imputation.sparse_no_impute(
            utils.get_dict_from_list_of_dict(
                dataset_dicts, dict_key='type',
                dict_value='NO_IMPUTATION',
            ),
            protein_start_col=X_START_COLUMN_IDX,
        )


    # %% Data Imputation Setup

    # Create imputers
    if SIMPLE_IMPUTER:
        dataset_dicts = dataset_dicts + imputation.create_simple_imputers(
            df=dataset,
            add_indicator=ADD_INDICATOR_SIMPLE_IMP,
            copy=COPY_SIMPLE_IMP,
            strategy=STRATEGY_SIMPLE_IMP
            )
    if ITERATIVE_IMPUTER and not PRECOMPUTED_ITERATIVE_IMPUTED_X_DATA:
        dataset_dicts = dataset_dicts + imputation.create_iterative_imputers(
            df=dataset,
            estimators=ESTIMATOR_ITER_IMP,
            estimator_criterion=ESTIMATOR_CRITERION_ITER_IMP,
            max_iter=MAX_ITER_ITER_IMP,
            tol=TOL_ITER_IMP,
            initial_strategy=INITIAL_STRATEGY_ITER_IMP,
            n_nearest_features=N_NEAREST_FEATURES_ITER_IMP,
            imputation_order=IMPUTATION_ORDER_ITER_IMP,
            add_indicator=ADD_INDICATOR_ITER_IMP,
            min_value=MIN_VALUE_ITER_IMP,
            max_value=MAX_VALUE_ITER_IMP,
            verbose=VERBOSE_ITER_IMP,
        )
    if KNN_IMPUTER:
        dataset_dicts = dataset_dicts + imputation.create_KNN_imputers(
            df=dataset,
            missing_values=MISSING_VALUES_KNN_IMP,
            n_neighbours=N_NEIGHBOURS_KNN_IMP,
            weights=WEIGHTS_KNN_IMP,
            metric=METRIC_KNN_IMP,
            copy=COPY_KNN_IMP,
            add_indicator=ADD_INDICATOR_KNN_IMP,
            keep_empty_features=KEEP_EMPTY_FEATURES_KNN_IMP,
        )


    # %% Add Missing Indicators

    if ADD_INDICATORS:
        for dataset_dict in dataset_dicts:
            indicator = MissingIndicator(missing_values=MISSING_VALUES_INDICATOR, features=FEATURES_INDICATOR, sparse=SPARSE_INDICATOR)
            indicators = indicator.fit_transform(dataset_dict['dataset'].iloc[:, 11:]).astype(float)
            feature_names = indicator.get_feature_names_out()
            indicators = pd.DataFrame(indicators, columns=feature_names)
            indicators.index = dataset_dict['dataset'].index
            dataset_dict['dataset'] = pd.concat((dataset_dict['dataset'], indicators), axis=1)


    # %% Categorization of Northstar score (y)

    dataset_dicts = [
        utils.make_binary(
            data_dict,
            column_label=COLUMN_TO_CATEGORIZE,
            cutoffs=CUTOFFS, copy=False
        )
        for data_dict in dataset_dicts
    ]


    # %% Train-test Split

    # Split data
    dataset_dicts = [
        features.split_data(
            data=dataset_dict,
            test_size=TEST_PROPORTION,
            random_state=SEED,
            start_col=X_START_COLUMN_IDX,
            y_col_label='FT5',
            X_age_col_label='Age',
        )
        for dataset_dict in dataset_dicts
    ]


    # %% Feature selection

    if STOP_AFTER_FEATURE_SELECTION:
        utils.log_results(
            original_dataset=original_dataset, original_protein_start_col=FIRST_COLUMN_TO_NORMALIZE, config=pipeline_config, data_dict=dataset_dicts[0], log=log
        )

    if SPARSE_NO_IMPUTATION or SELECT_XGB and not PRECOMPUTED_XGB_SELECTED_DATA:
        dataset_dicts = [
            features.select_XGB(
                data_dict=data_dict,
                log=log,
                n_estimators=N_ESTIMATORS_XGB,
                verbosity=VERBOSITY_XGB,
                use_label_encoder=USE_LABEL_ENCODER_XGB,
                validate_parameters=VALIDATE_PARAMETERS_XGB,
                missing=MISSING_XGB,
                objective=OBJECTIVE_XGB,
                n_jobs_xgb=N_JOBS_XGB,
                n_jobs_rfecv=N_JOBS_RFECV,
                scoring=SCORING_RFECV,
                cv=CV_RFE,
                min_features_to_select=MIN_FEATURES_TO_SELECT_XGB,
                step=STEP_XGB,
                original_dataset=dataset,
                original_protein_start_col=FIRST_COLUMN_TO_NORMALIZE,
                config=pipeline_config,
                start_time=START_TIME,
                logfile=LOG_FILE,
            )
            for data_dict in dataset_dicts
        ]

    # if STOP_AFTER_FEATURE_SELECTION:
    utils.log_time(start_time=START_TIME, end_time=datetime.now(), log=log, logfile=LOG_FILE)
    joblib.dump(
        dataset_dicts[0], PROJECT_ROOT / 'out' / Path(
            utils.get_file_name(deepcopy(dataset_dicts[0]), pipeline_config) + '__FeatureSelect꞉XGB-RFE-CV_dataset_dict.pkl'
            )
    )
    del dataset_dicts[0]['feature_selection_rfecv']
    del dataset_dicts[0]['feature_selection_xgb']
    # exit(0)


# %% Load Precomputed XGB Selected Data

else:
    dataset_dicts = [joblib.load(PRECOMPUTED_XGB_SELECTED_DATA)]
    original_dataset = deepcopy(dataset_dicts[0]['dataset'])
    dataset = dataset_dicts[0]['dataset']


# %% Data Imputation

imputer_dicts = []
if ITERATIVE_IMPUTER and not PRECOMPUTED_ITERATIVE_IMPUTED_X_DATA:
    imputer_dicts = []
    imputer_dicts = imputer_dicts + imputation.create_iterative_imputers(
        df=dataset,
        estimators=ESTIMATOR_ITER_IMP,
        estimator_criterion=ESTIMATOR_CRITERION_ITER_IMP,
        max_iter=MAX_ITER_ITER_IMP,
        tol=TOL_ITER_IMP,
        initial_strategy=INITIAL_STRATEGY_ITER_IMP,
        n_nearest_features=N_NEAREST_FEATURES_ITER_IMP,
        imputation_order=IMPUTATION_ORDER_ITER_IMP,
        add_indicator=ADD_INDICATOR_ITER_IMP,
        min_value=MIN_VALUE_ITER_IMP,
        max_value=MAX_VALUE_ITER_IMP,
        verbose=VERBOSE_ITER_IMP,
    )

# if STOP_AFTER_IMPUTATION:
utils.log_results(
    original_dataset=original_dataset,
    original_protein_start_col=FIRST_COLUMN_TO_NORMALIZE,
    config=pipeline_config,
    data_dict=dataset_dicts[0],
    log=log
)

# Impute data using generated imputers
if ITERATIVE_IMPUTER and PRECOMPUTED_ITERATIVE_IMPUTED_X_DATA:
    # Unpickle the precomputed imputed data
    try:
        dataset_dicts[0]['X_imputed'] = pd.read_csv(PRECOMPUTED_ITERATIVE_IMPUTED_X_DATA)
        if hasattr(dataset_dicts[0]['X_imputed'], 'Unnamed: 0'):
            dataset_dicts[0]['X_imputed'] = dataset_dicts[0]['X_imputed'].drop(columns='Unnamed: 0')
        if PRECOMPUTED_ITERATIVE_IMPUTED_DF:
            dataset_dicts[0]['dataset'] = pd.read_csv(PRECOMPUTED_ITERATIVE_IMPUTED_DF)
        else:
            X_imputed_temp = deepcopy(dataset_dicts[0]['X_imputed'])
            X_imputed_temp.index = dataset_dicts[0]['dataset'].iloc[:, 11:].index
            dataset_dicts[0]['dataset'].drop(columns=[col for col in dataset_dicts[0]['dataset'].iloc[:, 11:].columns if col not in X_imputed_temp.columns], inplace=True)
            dataset_dicts[0]['dataset'][X_imputed_temp.columns] = X_imputed_temp
    except Exception as e:
        log(f"Error: {e}")
        log("Could not load precomputed imputed data. Exiting.")
        exit(1)
    else:
        log("Precomputed imputed data loaded successfully!\n")
else:
    dataset_dicts = [
        imputation.impute_data(
            data_dict=data_dict,
            df=data_dict['dataset'],
            pipeline_start_time=START_TIME,
            imputer_dict=imp_dict,
            start_col_X=11,
            log=log
        )
        for data_dict, imp_dict in zip(dataset_dicts, imputer_dicts)
    ]


# %% Report Summary Statistics

# if STOP_AFTER_IMPUTATION:
if VERBOSE:
    # Report summary statistics for the imputed datasets
    for data_dict in dataset_dicts:
        utils.print_summary_statistics(data_dict, current_step='(POST-IMPUTATION) ', start_column=11)
utils.log_time(start_time=START_TIME, end_time=datetime.now(), log=log, logfile=LOG_FILE)
# exit(0)


# %% Train-test Split

# Split data
dataset_dicts = [
    features.split_data(
        X=dataset_dict['X_imputed'],
        y=dataset_dict['dataset']['FT5'],
        data=dataset_dict,
        test_size=TEST_PROPORTION,
        random_state=SEED,
        start_col=X_START_COLUMN_IDX,
        y_col_label='FT5',
    )
    for dataset_dict in dataset_dicts
]


# %% Log Preprocessing Results

utils.log_results(
    original_dataset=original_dataset, original_protein_start_col=FIRST_COLUMN_TO_NORMALIZE, config=pipeline_config, data_dict=dataset_dicts[0], log=log
)

print(
    """[LibSVM] From the documentation:
    Q: What is the difference between "." and "*" outputed during training?
    "." means every 1,000 iterations (or every #data iterations is your #data is less than 1,000). 
    "*" means that after iterations of using a smaller shrunk problem, we reset to use the whole set. 
    See: https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f209
    """
)

# %% Data Normalization and SVM Classifier Grid Search

# TODO: pipeline now assumes that we only have one dataset_dict up to this point.
#       Refactor to clarify this fact. Remove the loops over dataset_dicts and use Pipeline with CV for
#       normalization and SVC hyperparameter optimization instead.
dataset_dict = dataset_dicts[0]

# Extract the training and testing data
X = deepcopy(dataset_dict['X_imputed'])
y = deepcopy(dataset_dict['dataset']['FT5'])
X_training = deepcopy(dataset_dict['X_training'])
X_testing = deepcopy(dataset_dict['X_testing'])
y_training = deepcopy(dataset_dict['y_training']['FT5'])
y_testing = deepcopy(dataset_dict['y_testing']['FT5'])

param_grid_SVC = {
    'classifier__C':                       C_PARAMS_SVC,
    'classifier__kernel':                  KERNEL_PARAMS_SVC,
    'classifier__degree':                  DEGREE_PARAMS_SVC,
    'classifier__gamma':                   GAMMA_PARAMS_SVC,
    'classifier__coef0':                   COEF0_PARAMS_SVC,
    'classifier__shrinking':               SHRINKING_PARAMS_SVC,
    'classifier__probability':             PROBABILITY_SVC,
    'classifier__tol':                     TOL_PARAMS_SVC,
    'classifier__cache_size':              CACHE_SIZE_PARAMS_SVC,
    'classifier__verbose':                 [VERB_SVC],
    'classifier__max_iter':                MAX_ITER_PARAMS_SVC,
    'classifier__decision_function_shape': DECISION_FUNCTION_PARAMS_SVC,
    'classifier__break_ties':              BREAK_TIES_PARAMS_SVC,
    'classifier__random_state':            [SEED],
    }
# Construct parameter grid
param_grid = []
for i, normalizer in enumerate(NORMALIZATION_MODES_PARAMS):
    param_grid.append(deepcopy(param_grid_SVC))
    param_grid[i]['normalizer'] = [normalizer]

if CLASS_WEIGHT_SVC is not None:
    for param_set in param_grid:
        param_set['classifier__class_weight'] = [CLASS_WEIGHT_SVC]

pipeline = Pipeline([
    ('normalizer', None),
    ('classifier', svm.SVC())
])

# Perform grid search
clf = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=GRID_SEARCH_SCORING_SVC,
    refit=GRID_SEARCH_SCORING_SVC[0],
    cv=K_CV_FOLDS,
    verbose=GRID_SEARCH_VERBOSITY,
    return_train_score=CALC_FINAL_SCORES,
    n_jobs=N_JOBS_GRID_SEARCH,
)

# Fit the model
# if CALC_FINAL_SCORES:
#     clf = clf.fit_calc_final_scores(X_training, y_training, X_testing, y_testing)
# else:
#     clf = clf.fit(X_training, y_training)

try:
    clf = clf.fit(X_testing, y_testing)
except Exception as e:
    log(e)


# clf = clf.fit(X, y)

# With whole dataset StandardScaler(copy=False) is the best normalizer with 0.8524590163934426 accuracy
# [0.85245902 0.85245902 0.85245902 0.85245902 0.85245902 0.85245902]
# With train-test split StandardScaler(copy=False) is the best normalizer with 0.8524590163934426 accuracy
# [0.85245902 0.85245902 0.85245902 0.85245902 0.85245902 0.85245902]

# # Calculate and log best score (accuracy)
# if hasattr(clf, 'score'):
#     test_accuracy = clf.score(deepcopy(X_testing), deepcopy(y_testing))
# else:
#     Warning("The classifier does not have a 'score' attribute. Was it fitted?")
#     test_accuracy = None
#
# if hasattr(clf, 'dual_coef_'):
#     log('yi * alpha_i: \n', clf.dual_coef_)

# if hasattr(clf, 'cv_results_'):
#     clf_ = deepcopy(clf)  # make a copy of the classifier to use for calculating final scores
#     cv_results_ = clf.cv_results_  # but add final accuracies to cv_results_ of the original classifier
#     if CALC_FINAL_SCORES:
#         final_accuracies = []
#         if cv_results_ is not None:
#             # TODO: parallelize this loop
#             for params in cv_results_['params']:
#                 if hasattr(clf_, 'estimator'):
#                     estimator = clf_.estimator.set_params(**params)  # set the parameters of the copied estimator to the best parameters
#                     estimator.fit(X_training, y_training)            # fit the copied estimator
#                     final_accuracy = accuracy_score(y_testing.copy(), estimator.predict(X_testing.copy()))
#                     final_accuracies.append(final_accuracy)
#
#        # Add final accuracy scores to cv_results of the original classifier
#        # clf.cv_results_['final_accuracy'] = np.array(final_accuracies)
#
# dataset_dict['svm'] = {'clf': deepcopy(clf), 'test_accuracy': test_accuracy} # save original classifier and final test accuracy to the dataset_dict

dataset_dict['svm'] = {'clf': deepcopy(clf)} # save original classifier and final test accuracy to the dataset_dict

if VERBOSE:
    utils.log_grid_search_results(
        pipeline_config, dataset_dict, clf=clf, accuracy=None, log=log,  # log the original classifier and final test accuracy
    )

joblib.dump(deepcopy(clf), PROJECT_ROOT/'out'/f'{START_TIME.strftime("%Y-%m-%d-%H%M%S")}__GridSearchCV.pkl')
joblib.dump(dataset_dicts[0], PROJECT_ROOT/'out'/f'{START_TIME.strftime("%Y-%m-%d-%H%M%S")}__GridSearch_dataset_dict.pkl')


# %% End Time

utils.log_time(start_time=START_TIME, end_time=datetime.now(), log=log, logfile=LOG_FILE)


# %% Breakpoint

# breakpoint()
