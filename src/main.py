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
from collections.abc import Sequence, Callable
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import joblib
## External library imports
# noinspection PyUnresolvedReferences
import numpy as np  # needed for np.linspace/logspace in config
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.impute import MissingIndicator
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler

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
SEED: int = 42

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
FEATURE_SELECT_ONLY: bool = False

# Use precomputed XGB selected data dict
# If set, will skip: cleaning, adding imputer objects, adding indicators, categorizing y, splitting data
# and will start with imputing the data using the precomputed imputer objects.
PRECOMPUTED_XGB_SELECTED_DATA: Path | None = PROJECT_ROOT/'data'/'results'/'XGB-RFECV-binlog-feat-select-num-est꞉ALL'/'2024-05-11-041302__FeatureSelect__XGB-RFE-CV_dataset_dict.pkl'

# 37 specific features was found to be the best number of features using XGB feature selection.

# XGB-RFECV Config
N_ESTIMATORS_XGB = -1,
VERBOSITY_XGB = 0,
USE_LABEL_ENCODER_XGB = False,
VALIDATE_PARAMETERS_XGB = True,
MISSING_XGB = np.nan,
OBJECTIVE_XGB = 'binary:logistic',
N_JOBS_XGB = 6
N_JOBS_RFECV = 2
SCORING_RFECV = 'accuracy'
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
MAX_ITER_ITER_IMP: int = 600  # try low number of iterations first, see if converges, then try higher numbers
TOL_ITER_IMP: float = 0.009  # might need to adjust
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
PRECOMPUTED_ITERATIVE_IMPUTED_DATA: Path | None = None  # PROJECT_ROOT/'data'/'results'/'IterativeImputed-dataset-dict-minimal-test'/'TP꞉IterativeImputer_IT꞉200_TO꞉0.04_NF꞉None_IS꞉mean_IM꞉ascending_DT꞉2024-05-09-201931_Imputed_Datadict_using_23_features.pkl'

# Stop the pipeline after imputation
STOP_AFTER_IMPUTATION: bool = True


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

# ---------------
# SVM Classifier
# ---------------

# Enable SVC
SVC = True

# Hyperparameters:            # np.logspace(start, stop, num=50)
C_PARAMS_SVC: Sequence[float] = [0.0000_0001, 0.000_0001, 0.000_001, 0.000_01, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0]  # np.linspace(0.00001, 3, num=10)  # np.linspace(0.001, 100, num=60)
KERNEL_PARAMS_SVC: Sequence[str] = ['poly', 'sigmoid', 'rbf']  # 'linear', 'rbf', 'precomputed'
DEGREE_PARAMS_SVC: Sequence[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30]
GAMMA_PARAMS_SVC: Sequence[str] = ['auto']  # scale not needed since normalization X_var
COEF0_PARAMS_SVC: Sequence[float] = [-1000_000.0, -100_000.0, -10_000.0, -1000.0, -100.0, -10.0, -1.0, -0.1, -0.01, -0.001, 0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0, 1000_000.0]  # np.linspace(-2, 4, num=10)  # np.linspace(-10, 10, num=60)
SHRINKING_PARAMS_SVC: Sequence[bool] = [True]
PROBABILITY_SVC: Sequence[bool] = [False]
TOL_PARAMS_SVC: Sequence[float] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # np.linspace(0.01, 0.0001, 10)  # np.linspace(0.01, 0.0001, 10)
CACHE_SIZE_PARAMS_SVC: Sequence[int] = [500]
CLASS_WEIGHT_SVC: dict | None = None
VERB_SVC: int = VERBOSE
MAX_ITER_PARAMS_SVC: Sequence[int] = [1_000_000,]  # [-1]
DECISION_FUNCTION_PARAMS_SVC: Sequence[str] = ['ovr']  # only ovo if multi class
BREAK_TIES_PARAMS_SVC: Sequence[bool] = [False]

# RETURN_TRAIN_SCORE_SVC:
# Include scores in `cv_results_`. Computing training scores is used to get insights on how different parameter settings
# impact the overfitting/underfitting trade-off. However, computing the scores on the training set can be
# computationally expensive and is not strictly required to select the parameters that yield the best generalization
# performance.
RETURN_TRAIN_SCORE_SVC: bool = True
GRID_SEARCH_SCORING_SVC: str = 'accuracy'


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

# Suppress matplotlib logs
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Get the logger
log = logging.getLogger('LOGGER_NAME').info

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
    'STOP_AFTER_IMPUTATION':         STOP_AFTER_IMPUTATION,
}


# %% Generate new XGB selected data

dataset = None
dataset_dicts = []
if not PRECOMPUTED_XGB_SELECTED_DATA:


    # %% Load Data

    # Load the data
    dataset = pd.read_csv(DATA_FILE)

    if VERBOSE:
        log("|--- PIPELINE ---|")
        log("Date: " + START_TIME.strftime("%Y-%m-%d %H:%M:%S"))
        log(f"Debug mode: {DEBUG}")
        log("Data loaded successfully.\n")
    if DEBUG:
        # Import and unpack pipeline debug config from debug_config.py
        from debug_config import pipeline_debug_config
        for varname, value in pipeline_debug_config.items():
            globals()[varname] = value  # Overwrite variables in this script with those from the debug config


    # %% Data Cleaning

    # Clean and preprocess the data
    dataset = cleaning.clean_data(dataset, verbose=VERBOSE, log=log, date=START_TIME, dataset_path=DATA_FILE)

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
    if ITERATIVE_IMPUTER and not PRECOMPUTED_ITERATIVE_IMPUTED_DATA:
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
        )
        for dataset_dict in dataset_dicts
    ]


    # %% Feature selection

    if SPARSE_NO_IMPUTATION or SELECT_XGB:
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

    if FEATURE_SELECT_ONLY:
        utils.log_results(
            original_dataset=dataset, original_protein_start_col=FIRST_COLUMN_TO_NORMALIZE, config=pipeline_config, log=log
        )
        utils.log_time(start_time=START_TIME, end_time=datetime.now(), log=log, logfile=LOG_FILE)

        joblib.dump(
            dataset_dicts[0], PROJECT_ROOT / 'out' / Path(
                utils.get_file_name(deepcopy(dataset_dicts[0]), pipeline_config) + '__FeatureSelect꞉XGB-RFE-CV_dataset_dict.pkl'
                )
            )

        exit(0)


# %% Load Precomputed XGB Selected Data

else:
    dataset_dicts = [joblib.load(PRECOMPUTED_XGB_SELECTED_DATA)]
    dataset = dataset_dicts[0]['dataset']


# %% Data Imputation

imputer_dicts = []
if ITERATIVE_IMPUTER and not PRECOMPUTED_ITERATIVE_IMPUTED_DATA and PRECOMPUTED_XGB_SELECTED_DATA:
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


# Impute data using generated imputers
if ITERATIVE_IMPUTER and PRECOMPUTED_ITERATIVE_IMPUTED_DATA:
    # Unpickle the precomputed imputed data
    try:
        dataset_dicts = dataset_dicts + [joblib.load(PRECOMPUTED_ITERATIVE_IMPUTED_DATA)]
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

if STOP_AFTER_IMPUTATION:
    utils.log_results(
        original_dataset=dataset,
        original_protein_start_col=FIRST_COLUMN_TO_NORMALIZE,
        config=pipeline_config,
        log=log
    )
    utils.log_time(start_time=START_TIME, end_time=datetime.now(), log=log, logfile=LOG_FILE)
    exit(0)


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


# %% Log results

utils.log_results(
    original_dataset=dataset, original_protein_start_col=FIRST_COLUMN_TO_NORMALIZE, config=pipeline_config, log=log
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
            kernels=KERNEL_PARAMS_SVC,
            degree_params=DEGREE_PARAMS_SVC,
            gamma_params=GAMMA_PARAMS_SVC,
            coef0_params=COEF0_PARAMS_SVC,
            shrinking=SHRINKING_PARAMS_SVC,
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
            calc_final_scores=CALC_FINAL_SCORES,
            log=log,
        )
        for dataset_dict in dataset_dicts
    ]


# %% End Time

joblib.dump(dataset_dicts[0], PROJECT_ROOT/'out'/f'{START_TIME.strftime("%Y-%m-%d-%H%M%S")}__dataset_dict.pkl')
log(f"Pickled dataset_dict saved to: ----------------------------- {PROJECT_ROOT/'out'/f'{START_TIME.strftime("%Y-%m-%d-%H%M%S")}__dataset_dict.pkl'}")

utils.log_time(start_time=START_TIME, end_time=datetime.now(), log=log, logfile=LOG_FILE)


# %% Breakpoint

# breakpoint()
