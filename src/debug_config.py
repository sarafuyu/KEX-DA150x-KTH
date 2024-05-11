#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug config used if `DEBUG` is set to `True` in main.py

If `DEBUG` is set to `True` the `pipeline_debug_config` dict below will be
imported and unpacked by main.

The values in this file will override the values in the main script.
Feel free to comment out any parameters you still want to set from main.

:Date: 2024-05-01
:Authors: Sara Rydell, Noah Hopkins
"""
# %% Imports

# Note: Leave unused imports in place to allow for easy switching between.

# Imports

# noinspection PyUnresolvedReferences
import pandas as pd                              # needed for pd.NA
# noinspection PyUnresolvedReferences
import numpy as np                               # needed for np.linspace/logspace and np.nan
# noinspection PyUnresolvedReferences
from sklearn.feature_selection import f_classif  # needed for f_classif
# noinspection PyUnresolvedReferences
from sklearn.linear_model import BayesianRidge   # needed for BayesianRidge
from pathlib import Path


# %% Debug Config

PROJECT_ROOT = Path(__file__).parents[1]

pipeline_debug_config = {
    # --- General ------------------------------------------------------------ #
    # 'VERBOSE':                     1,
    # 'LOGGER':                      True,
    # 'SEED':                        42,
    # 'DATA_FILE':                   PROJECT_ROOT/'data'/'dataset'/'normalised_data_all_w_clinical_kex_20240321.csv',
    # --- Imputation --------------------------------------------------------- #
    'SIMPLE_IMPUTER':               False,
    'ITERATIVE_IMPUTER':            True,
    'KNN_IMPUTER':                  False,
    'NO_IMPUTATION':                False,
    'SPARSE_NO_IMPUTATION':         False,
    # --- SimpleImputer ------------------------------------------------------ #
    # 'STRATEGY_SIMPLE_IMP':          ['mean'],
    # 'ADD_INDICATOR_SIMPLE_IMP':     False,
    # 'COPY_SIMPLE_IMP':              False,
    # --- IterativeImputer --------------------------------------------------- #
    # 'ESTIMATOR_ITER_IMP':          BayesianRidge(),
    # 'MAX_ITER_ITER_IMP':           100,
    # 'TOL_ITER_IMP':                1e-3,
    # 'INITIAL_STRATEGY_ITER_IMP':   ["mean"],
    # 'N_NEAREST_FEATURES_ITER_IMP': [5],
    # 'IMPUTATION_ORDER_ITER_IMP':   ["ascending"],
    # 'MIN_VALUE_ITER_IMP':          0,
    # 'MAX_VALUE_ITER_IMP':          '10% higher than max',
    # --- KNNImputer --------------------------------------------------------- #
    # 'MISSING_VALUES_KNN_IMP':      pd.NA,
    # 'N_NEIGHBOURS_KNN_IMP':        [30],
    # 'WEIGHTS_KNN_IMP':             ['uniform'],
    # 'METRIC_KNN_IMP':              ['nan_euclidean'],
    # 'ADD_INDICATOR_KNN_IMP':       False,
    # 'KEEP_EMPTY_FEATURES_KNN_IMP': False,
    # 'COPY_KNN_IMP':                True,
    # --- NaN Elimination ---------------------------------------------------- #
    # 'DROP_COLS_NAN_ELIM':          True,
    # --- Normalization ------------------------------------------------------ #
    # 'FIRST_COLUMN_TO_NORMALIZE':   11,
    # --- Categorization ----------------------------------------------------- #
    # 'CUTOFFS':                     [17],
    # 'COLUMN_TO_CATEGORIZE':        'FT5',
    # --- Training Split ----------------------------------------------------- #
    # 'TEST_PROPORTION':             0.2,
    # 'X_START_COLUMN_IDX':          11,
    # 'Y_COLUMN_LABEL':              'FT5',
    # --- Feature Selection -------------------------------------------------- #
    # 'SCORE_FUNC_FEATURES':         f_classif,
    # 'K_FEATURES':                  30,
    # --- Classifier/Grid Search---------------------------------------------- #
    # 'GRID_SEARCH_VERBOSITY':       1,
    # 'K_CV_FOLDS':                  5,
    # 'CALC_FINAL_SCORES':           True,
    # --- SVC ---------------------------------------------------------------- #
    'SVC':                          True,
    'C_PARAMS_SVC':                 [0.1, 1.0],
    'KERNEL_PARAMS_SVC':            ['poly'],
    'DEGREE_PARAMS_SVC':            [3],
    'GAMMA_PARAMS_SVC':             ['auto'],
    'COEF0_PARAMS_SVC':             [1],
    'SHRINKING_PARAMS_SVC':         [True],
    'PROBABILITY_SVC':              [False],
    'TOL_PARAMS_SVC':               [0.01],
    'CACHE_SIZE_PARAMS_SVC':        [500],
    'CLASS_WEIGHT_SVC':             None,
    'VERB_SVC':                     1,
    'MAX_ITER_PARAMS_SVC':          [1_000_000],
    'DECISION_FUNCTION_PARAMS_SVC': ['ovr'],
    'BREAK_TIES_PARAMS_SVC':        [False],
    'RETURN_TRAIN_SCORE_SVC':       False,
    'GRID_SEARCH_SCORING_SVC':      'accuracy',
    # --- SVR ---------------------------------------------------------------- #
    # 'SVR':                          False,
    # 'KERNEL_PARAMS_SVR':            ['rbf'],
    # 'DEGREE_PARAMS_SVR':            [3],
    # 'GAMMA_PARAMS_SVR':             ['scale'],
    # 'COEF0_PARAMS_SVR':             [0.0],
    # 'TOL_PARAMS_SVR':               [0.001],
    # 'C_PARAMS_SVR':                 [1.0, 10.0],
    # 'EPSILON_PARAMS_SVR':           [0.1],
    # 'SHRINKING_PARAMS_SVR':         [True],
    # 'CACHE_SIZE_PARAMS_SVR':        [500],
    # 'VERB_SVR':                     1,
    # 'MAX_ITER_PARAMS_SVR':          [1_000_000],
    # 'RETURN_TRAIN_SCORE_SVR':       [False],
    # 'GRID_SEARCH_SCORING_SVR':      'neg_mean_squared_error',
    # ------------------------------------------------------------------------ #
}
