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
TODO(priority 1): Add filtering of dataset dicts in each find_best_Xmodel function
                  so that only compatible datasets are used during gridsearch
TODO(priority 1): Add SVR classifier
------ Nice to have for today ------
TODO(priority 2): Implement SVM classifier with NaN
TODO(priority 2): Validate that multiclass SVM still works in the pipeline
------ MVP for tomorrow ------
TODO(priority 3): Add Ridge Regression
TODO(priority 3): Add Lasso Regression
------ Nice to have for tomorrow ------
TODO(priority 4): Add Decision Tree classifier
TODO(priority 4): Add Naive Bayes classifier
TODO(priority 4): Refactor cleaning.py
TODO(priority 5): Refactor main.py pipeline to make it more DRY (construct grid_param in main and 
                  pass to a function that calls the various grid search functions for the various 
                  models depending on the different
                  dataset dict attributes)
--------------------------------

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Imports

## Standard library imports
import sys
import logging
from collections.abc import Sequence, Callable
from datetime import datetime
from pathlib import Path

## External library imports
import pandas as pd
import numpy as np  # needed for np.linspace/logspace in config
from sklearn.feature_selection import f_classif
from sklearn.linear_model import BayesianRidge


# %% Setup

START_TIME = datetime.now()
PROJECT_ROOT = Path(__file__).parents[1]

print(PROJECT_ROOT)

# %% Configuration

# **********----------------------------------------------------------------------------********** #
# |                                        ~~ General ~~                                         | #
# **********----------------------------------------------------------------------------********** #

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
SEED: int = 42

# ----------------
# Data Extraction
# ----------------

# Path to dataset
DATA_FILE: Path = PROJECT_ROOT/'data'/'dataset'/'normalised_data_all_w_clinical_kex_20240321.csv'


# **********----------------------------------------------------------------------------********** #
# |                                     ~~ Data imputation ~~                                    | #
# **********----------------------------------------------------------------------------********** #

# Pick imputation modes to use:
SIMPLE_IMPUTER: bool = False # True
ITERATIVE_IMPUTER: bool = False
KNN_IMPUTER: bool = False
NAN_ELIMINATION: bool = True
NO_IMPUTATION: bool = False
SPARSE_NO_IMPUTATION: bool = False  # Note: if `True`, `NO_IMPUTATION` must be set to `True`.

# -----------------------------
# Simple imputer configuration
# -----------------------------

# Strategy for imputing missing values
STRATEGY_SIMPLE_IMP: Sequence[str] = ['mean']  # ["mean", "median", "most_frequent"]
# Add indicator for missing values
ADD_INDICATOR_SIMPLE_IMP: bool = False

# Should always be True, since the implementation expects a copy of the data
COPY_SIMPLE_IMP: bool = True

# --------------------------------
# Iterative imputer configuration
# --------------------------------

# Estimator, e.g. a BayesianRidge() object or an estimator object from scikit-learn.
# Can probably be customized, but leave default for now.
# For future type hints, see: https://stackoverflow.com/a/60542986/6292000
ESTIMATOR_ITER_IMP = BayesianRidge()
# Maximum number of imputation rounds to perform. The imputer will stop iterating after this many iterations.
MAX_ITER_ITER_IMP: int = 100  # try low number of iterations first, see if converges, then try higher numbers
TOL_ITER_IMP: float = 1e-3  # might need to adjust
INITIAL_STRATEGY_ITER_IMP: Sequence[str] = ["mean"]  # ["mean", "median", "most_frequent", "constant"]
# Number of other features to use to estimate the missing values of each feature column.
# None means all features, which might be too many.
N_NEAREST_FEATURES_ITER_IMP: Sequence[int] = [5, 10, 20, 50, None]  # [10, 100, 500, None]

IMPUTATION_ORDER_ITER_IMP: Sequence[str] = ["ascending"]  # ["ascending", "descending" "random"]
# ascending: From the features with the fewest missing values to those with the most
MIN_VALUE_ITER_IMP: int = 0  # no features have negative values, adjust tighter for prot intensities?
MAX_VALUE_ITER_IMP: str | int = '10% higher than max'

# --------------------------
# KNN imputer configuration
# --------------------------

# Missing values to impute
MISSING_VALUES_KNN_IMP = pd.NA
# Initial span of neighbours considering dataset size
N_NEIGHBOURS_KNN_IMP: Sequence[int] = [5, 10, 20, 50]
# default='uniform', callable has potential for later fitting
WEIGHTS_KNN_IMP: Sequence[str] = ['uniform', 'distance']
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

# First column to normalize. Will normalize all columns from this index and onwards.
FIRST_COLUMN_TO_NORMALIZE: int = 11
# We only normalize the antibody/protein intensity columns (cols 11 and up). Age, disease, FTs not normalized.

# TODO: check if this is the correct index. Should it be 10 instead?
# TODO: instead iterate over the column labels until we reach the first protein intensity column
#       then normalize all columns from there and onwards instead of hardcoding the index.


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
# |                                   ~~ Train-test split ~~                                     | #
# **********----------------------------------------------------------------------------********** #

# For detailed configuration for each feature selection mode, see features.py
TEST_PROPORTION: float = 0.2

# Column index to start from. Will split the selected X and y data, into the two data subsets:
#   X_train, y_train,
#   X_test, y_test,
# where X_train and X_test are the input variables and y_train and y_test are the target variables.
X_START_COLUMN_IDX: int = 11  # X columns are from this column index and onwards TODO: check if 11 is the correct index.
Y_COLUMN_LABEL: str = 'FT5'   # y column label


# **********----------------------------------------------------------------------------********** #
# |                                   ~~ Feature selection ~~                                    | #
# **********----------------------------------------------------------------------------********** #

# SCORE_FUNC_FEATURES is a function taking in X, an array of columns (features), and y, a target column,
# and returning a pair of arrays (scores, p-values).
SCORE_FUNC_FEATURES: Callable[[Sequence, Sequence], tuple[Sequence, Sequence]] = f_classif
K_FEATURES: int = 30  # 216  # 100 # TODO: add different levels: 30, 60, 90, 120, 150, 200 ...


# **********----------------------------------------------------------------------------********** #
# |                              ~~ Model training & fitting ~~                                  | #
# **********----------------------------------------------------------------------------********** #

# ------------
# Grid Search
# ------------

# Set the verbosity level for the grid search printouts that are not logged.
GRID_SEARCH_VERBOSITY: int = -1
# Number of cross-validation folds
K_CV_FOLDS: int = 5
# Calculate final accuracy for all models
CALC_FINAL_SCORES: bool = True

# ---------------
# SVM Classifier
# ---------------

# Enable SVC
SVC = True

# Hyperparameters:            # np.logspace(start, stop, num=50)
C_PARAMS_SVC: Sequence[float] = [1.0] # [0.0000_0001, 0.000_0001, 0.000_001, 0.000_01, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0]  # np.linspace(0.00001, 3, num=10)  # np.linspace(0.001, 100, num=60)
KERNELS_SVC: Sequence[str] = ['poly'] # ['poly', 'sigmoid', 'rbf']  # 'linear', 'rbf', 'precomputed'
DEGREE_PARAMS_SVC: Sequence[int] = [1,2,3,4,5] # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30]
GAMMA_PARAMS_SVC: Sequence[str] = ['auto']  # scale not needed since normalization X_var
COEF0_PARAMS_SVC: Sequence[float] = [0.001,0.01,0.1,0.0,1.0] # np.linspace(0.00001, 3, num=2)  # [-1000_000.0, -100_000.0, -10_000.0, -1000.0, -100.0, -10.0, -1.0, -0.1, -0.01, -0.001, 0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0, 1000_000.0]  # np.linspace(-2, 4, num=10)  # np.linspace(-10, 10, num=60)
SHRINKING_SVC: Sequence[bool] = [True]
PROBABILITY_SVC: Sequence[bool] = [False]
TOL_PARAMS_SVC: Sequence[float] = [0.00001,0.0001,0.001,0.01,0.1] # np.linspace(0.00001, 3, num=2)  # [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # np.linspace(0.01, 0.0001, 10)  # np.linspace(0.01, 0.0001, 10)
CACHE_SIZE_PARAMS_SVC: Sequence[int] = [500]
CLASS_WEIGHT_SVC: dict | None = None
VERB_SVC: int = VERBOSE
MAX_ITER_PARAMS_SVC: Sequence[int] = [1_000_000]  # [-1]
DECISION_FUNCTION_PARAMS_SVC: Sequence[str] = ['ovr']  # onli ovo if multi class
BREAK_TIES_PARAMS_SVC: Sequence[bool] = [False]

# RETURN_TRAIN_SCORE_SVC:
# Include scores in `cv_results_`. Computing training scores is used to get insights on how different parameter settings
# impact the overfitting/underfitting trade-off. However, computing the scores on the training set can be
# computationally expensive and is not strictly required to select the parameters that yield the best generalization
# performance.
RETURN_TRAIN_SCORE_SVC: bool = False
GRID_SEARCH_SCORING_SVC: str = 'accuracy'

# --------------
# SVR Classifier
# --------------

# Enable SVR
SVR = False

KERNELS_SVR: Sequence[str] = ['rbf']
DEGREE_PARAMS_SVR: Sequence[int] = [3]
GAMMA_PARAMS_SVR: Sequence[str] = ['scale']
COEF0_PARAMS_SVR: Sequence[float] = [0.0]
TOL_PARAMS_SVR: Sequence[float] = [0.001]
C_PARAMS_SVR: Sequence[float] = [1.0, 10.0]
EPSILON_PARAMS_SVR: Sequence[float] = [0.1]
SHRINKING_PARAMS_SVR: Sequence[bool] = [True]
CACHE_SIZE_PARAMS_SVR: Sequence[int] = [200]
VERB_SVR: int = VERBOSE
MAX_ITER_PARAMS_SVR: Sequence[int] = [-1]

# RETURN_TRAIN_SCORE_SVR:
# Include scores in `cv_results_`. Computing training scores is used to get insights on how different parameter settings
# impact the overfitting/underfitting trade-off. However, computing the scores on the training set can be
# computationally expensive and is not strictly required to select the parameters that yield the best generalization
# performance.
RETURN_TRAIN_SCORE_SVR: bool = False
GRID_SEARCH_SCORING_SVR: str = 'neg_mean_squared_error'


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

logging.basicConfig(level=logging.DEBUG,
                    format=('[%(asctime)s] '
                            '{%(filename)s:%(lineno)d} : '
                            '%(levelname)s | '
                            '%(message)s'),
                    handlers=handlers)

log = logging.getLogger('LOGGER_NAME').info


# %% Load Data

# Load the data
dataset = pd.read_csv(DATA_FILE)

if VERBOSE:
    log("Data loaded successfully.")
if VERBOSE == 2:
    dataset.head()  # Pre-view first five rows


# %% Data Cleaning

# Clean and preprocess the data
dataset = cleaning.clean_data(dataset, verbose=VERBOSE, log=log, date=START_TIME)


# %% Data Imputation

# Create imputers
dataset_dicts = []
if SIMPLE_IMPUTER:
    dataset_dicts = dataset_dicts + imputation.create_simple_imputers(
        add_indicator=ADD_INDICATOR_SIMPLE_IMP, copy=COPY_SIMPLE_IMP, strategy=STRATEGY_SIMPLE_IMP
        )
if ITERATIVE_IMPUTER:
    dataset_dicts = dataset_dicts + imputation.create_iterative_imputers(
        df=dataset,
        estimator=ESTIMATOR_ITER_IMP,
        max_iter=MAX_ITER_ITER_IMP,
        tol=TOL_ITER_IMP,
        initial_strategy=INITIAL_STRATEGY_ITER_IMP,
        n_nearest_features=N_NEAREST_FEATURES_ITER_IMP,
        imputation_order=IMPUTATION_ORDER_ITER_IMP,
        min_value=MIN_VALUE_ITER_IMP,
        max_value=MAX_VALUE_ITER_IMP,
    )
if KNN_IMPUTER:
    dataset_dicts = dataset_dicts + imputation.create_KNN_imputers(
        missing_values=MISSING_VALUES_KNN_IMP,
        n_neighbours=N_NEIGHBOURS_KNN_IMP,
        weights=WEIGHTS_KNN_IMP,
        metric=METRIC_KNN_IMP,
        copy=COPY_KNN_IMP,
        add_indicator=ADD_INDICATOR_KNN_IMP,
        keep_empty_features=KEEP_EMPTY_FEATURES_KNN_IMP,
    )

# Impute data using generated imputers
dataset_dicts = [imputation.impute_data(imputer_dict, dataset, 11)
                 for imputer_dict in dataset_dicts]

# Add NaN-eliminated and un-imputed datasets
if NAN_ELIMINATION:
    dataset_dicts = dataset_dicts + imputation.eliminate_nan(dataset, drop_cols=DROP_COLS_NAN_ELIM)
if NO_IMPUTATION:
    dataset_dicts = dataset_dicts + imputation.no_imputer(dataset, copy=True)


# %% Report Summary Statistics

# Report summary statistics for the imputed datasets
if VERBOSE:
    for data_dict in dataset_dicts:
        utils.print_summary_statistics(data_dict, start_column=11)


# %% Data Normalization

# Normalize the datasets
dataset_dicts = [
    normalization.std_normalization(
        data=data_dict,
        start_column=FIRST_COLUMN_TO_NORMALIZE,
    )
    for data_dict in dataset_dicts
]


# %% Categorization of Northstar score (y)

if CUTOFFS:
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
        y_col_label=Y_COLUMN_LABEL,
    )
    for dataset_dict in dataset_dicts
]


# %% Sparse Data Handling

if SPARSE_NO_IMPUTATION:
    dataset_dicts = dataset_dicts + imputation.sparse_no_impute(
        utils.get_dict_from_list_of_dict(dataset_dicts, dict_key='type', dict_value='NO_IMPUTATION')
    )


# %% Feature selection

# Feature selection
dataset_dicts = [
    features.select_KBest(
        data_dict=data_dict,
        score_func=SCORE_FUNC_FEATURES,
        k=K_FEATURES
    )
    for data_dict in dataset_dicts
]


# %% Log results

pipeline_config = {
    'SEED':                        SEED,
    'VERBOSE':                     VERBOSE,
    'DATA_FILE':                   DATA_FILE,
    'SIMPLE_IMPUTER':              SIMPLE_IMPUTER,
    'ITERATIVE_IMPUTER':           ITERATIVE_IMPUTER,
    'KNN_IMPUTER':                 KNN_IMPUTER,
    'NAN_ELIMINATION':             NAN_ELIMINATION,
    'NO_IMPUTATION':               NO_IMPUTATION,
    'SPARSE_NO_IMPUTATION':        SPARSE_NO_IMPUTATION,
    'ADD_INDICATOR_SIMPLE_IMP':    ADD_INDICATOR_SIMPLE_IMP,
    'COPY_SIMPLE_IMP':             COPY_SIMPLE_IMP,
    'STRATEGY_SIMPLE_IMP':         STRATEGY_SIMPLE_IMP,
    'ESTIMATOR_ITER_IMP':          ESTIMATOR_ITER_IMP,
    'MAX_ITER_ITER_IMP':           MAX_ITER_ITER_IMP,
    'TOL_ITER_IMP':                TOL_ITER_IMP,
    'INITIAL_STRATEGY_ITER_IMP':   INITIAL_STRATEGY_ITER_IMP,
    'N_NEAREST_FEATURES_ITER_IMP': N_NEAREST_FEATURES_ITER_IMP,
    'IMPUTATION_ORDER_ITER_IMP':   IMPUTATION_ORDER_ITER_IMP,
    'MIN_VALUE_ITER_IMP':          MIN_VALUE_ITER_IMP,
    'MAX_VALUE_ITER_IMP':          MAX_VALUE_ITER_IMP,
    'N_NEIGHBOURS_KNN_IMP':        N_NEIGHBOURS_KNN_IMP,
    'WEIGHTS_KNN_IMP':             WEIGHTS_KNN_IMP,
    'METRIC_KNN_IMP':              METRIC_KNN_IMP,
    'DROP_COLS_NAN_ELIM':          DROP_COLS_NAN_ELIM,
    'FIRST_COLUMN_TO_NORMALIZE':   FIRST_COLUMN_TO_NORMALIZE,
    'CUTOFFS':                     CUTOFFS,
    'COLUMN_TO_CATEGORIZE':        COLUMN_TO_CATEGORIZE,
    'TEST_PROPORTION':             TEST_PROPORTION,
    'X_START_COLUMN_IDX':          X_START_COLUMN_IDX,
    'Y_COLUMN_LABEL':              Y_COLUMN_LABEL,
    'SCORE_FUNC_FEATURES':         SCORE_FUNC_FEATURES,
    'K_FEATURES':                  K_FEATURES,
    'SVC':                         SVC,
    # 'C_PARAMS_SVC':                 C_params_SVC,
    # 'KERNELS_SVC':                  kernels_SVC,
    # 'DEGREE_PARAMS_SVC':            degree_params_SVC,
    # 'GAMMA_PARAMS_SVC':             gamma_params_SVC,
    # 'COEF0_PARAMS_SVC':             coef0_params_SVC,
    # 'TOL_PARAMS_SVC':               tol_params_SVC,
    # 'DECISION_FUNCTION_PARAMS_SVC': decision_function_params_SVC,
    'SVR':                         SVR,
    # 'KERNELS_SVR':                  kernels_SVR,
    # 'DEGREE_PARAMS_SVR':            degree_params_SVR,
    # 'GAMMA_PARAMS_SVR':             gamma_paramS_SVR,
    # 'COEF0_PARAMS_SVR':             coef0_paramS_SVR,
    # 'TOL_PARAMS_SVR':               tol_params_SVR,
    # 'C_PARAMS_SVR':                 C_params_SVR,
    # 'EPSILON_PARAMS_SVR':           epsilon_params_SVR,
    'START_TIME':                  START_TIME,
}

utils.log_results(
    original_dataset=dataset, original_protein_start_col=FIRST_COLUMN_TO_NORMALIZE, config=pipeline_config
    )


# %% Model Training & Fitting

# Create Naive Bayes models
# dataset_dicts = [classifier.add_naive_bayes_models(dataset_dict) for dataset_dict in dataset_dicts]

# Find best SVM model
if SVC:
    dataset_dicts = [
        classifier.find_best_svm_model(
            pipeline_config=pipeline_config,
            dataset_dict=dataset_dict,
            C_params=C_PARAMS_SVC,
            kernels=KERNELS_SVC,
            degree_params=DEGREE_PARAMS_SVC,
            gamma_params=GAMMA_PARAMS_SVC,
            coef0_params=COEF0_PARAMS_SVC,
            shrinking=SHRINKING_SVC,
            probability=PROBABILITY_SVC,
            tol_params=TOL_PARAMS_SVC,
            cache_size_params=CACHE_SIZE_PARAMS_SVC,
            class_weight=CLASS_WEIGHT_SVC,
            verb=VERB_SVC,
            max_iter_params=MAX_ITER_PARAMS_SVC,
            decision_function_params=DECISION_FUNCTION_PARAMS_SVC,
            break_ties=BREAK_TIES_PARAMS_SVC,
            random_state=SEED,
            grid_search_verbosity=GRID_SEARCH_VERBOSITY,
            return_train_score=RETURN_TRAIN_SCORE_SVC,
            grid_search_scoring=GRID_SEARCH_SCORING_SVC,
            k_cv_folds=K_CV_FOLDS,
            calc_final_scores=CALC_FINAL_SCORES
        )
        for dataset_dict in dataset_dicts
    ]

# Find best SVR model
if SVR:
    dataset_dicts = [
        classifier.find_best_svr_model(
            pipeline_config=pipeline_config,
            dataset_dict=dataset_dict,
            kernels=KERNELS_SVR,
            degree_params=DEGREE_PARAMS_SVR,
            gamma_params=GAMMA_PARAMS_SVR,
            coef0_params=COEF0_PARAMS_SVR,
            tol_params=TOL_PARAMS_SVR,
            C_params=C_PARAMS_SVR,
            epsilon_params=EPSILON_PARAMS_SVR,
            shrinking_params=SHRINKING_PARAMS_SVR,
            cache_size_params=CACHE_SIZE_PARAMS_SVR,
            verb=VERB_SVR,
            max_iter_params=MAX_ITER_PARAMS_SVR,
            grid_search_verbosity=GRID_SEARCH_VERBOSITY,
            return_train_score=RETURN_TRAIN_SCORE_SVR,
            grid_search_scoring=GRID_SEARCH_SCORING_SVR,
            k_cv_folds=K_CV_FOLDS,
            calc_final_scores=CALC_FINAL_SCORES
            )
        for dataset_dict in dataset_dicts
    ]

# Find best Ridge Regression model
# TODO

# Find best Lasso Regression model
# TODO

# Find best Decision Tree model
# TODO


# %% End Time

utils.log_time(start_time=START_TIME, end_time=datetime.now(), log=print)


# %% Breakpoint

# breakpoint()