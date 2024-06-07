#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prestudy metrics

This script is run separately from the main script to determine which normalizer
and class weight to use for the final grid search.

The script loads the cross-validation results from the final prestudy and
compares the cv_results for different normalizers and class weights. The script
then selects the best parameters for these normalizers and class weights based
on the roc_auc score and evaluates the final model using the test set to get the
final roc_auc score and confusion matrix.

:Date: 2024-05-03
:Authors: Sara Rydell, Noah Hopkins

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Imports

# Standard library imports
import warnings
import pickle
import itertools
from pathlib import Path
from collections.abc import Sequence
from copy import deepcopy, copy

# External imports
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For heatmap or advanced plotting
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator, SymmetricalLogLocator, FuncFormatter
from joypy import joyplot
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score

# Local imports
import utils  # The utils module need to be imported first to set verbosity level and random seed
import features


# %% Setup

SEED = utils.RANDOM_SEED  # get random seed
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_SETS = {
    'final-prestudy': {  # Bad data (used test data for training)
        'results_directory': 'final-prestudy',
        'cv_results_':       '2024-06-02-223758__GridSearchCV.pkl',
    },
    'final-prestudy-tol-normalizers': {  # Bad data (used test data for training)
        'results_directory': 'final-prestudy-tol-normalizers',
        'cv_results_':       '2024-06-05-000027__GridSearchCV.pkl',
    },
    'final-prestudy-new': {  # Good data
        'results_directory': 'final-prestudy-new',
        'cv_results_':       '2024-06-06-190955__GridSearchCV.pkl',
    },
}


# %% Configuration

FEATURE_SELECTION_METRIC = 'final-prestudy-new'  # 'accuracy', 'roc_auc', 'f1'
GRIDSEARCH_METRIC = 'roc_auc'  # 'accuracy', 'roc_auc', 'f1'  # noqa

CV_RESULTS_DIR = PROJECT_ROOT/'data'/'results'/RESULTS_SETS[FEATURE_SELECTION_METRIC]['results_directory']  # noqa
GRID_SEARCH_RESULTS_PATH: Path = CV_RESULTS_DIR/RESULTS_SETS[FEATURE_SELECTION_METRIC]['cv_results_']

# Specify parameters of interest
PARAMS_TO_GET_BEST = ['C', 'degree', 'coef0', 'kernel', 'gamma', 'class_weight', 'tol']
PARAM_PREFIX = 'param_'  # Prefix for the parameter columns in the cv_results_ DataFrame
SVC_DEFAULT_PARAMS = {'param_C': 1.0, 'param_degree': 3, 'param_coef0': 0.0, 'param_gamma': 'scale', 'param_class_weight': 'None', 'param_tol': 0.001}
SVC_DEFAULT_KERNEL = 'rbf'

# # Load the cross-validation results
gridsearch_cv = joblib.load(GRID_SEARCH_RESULTS_PATH)
cv_results = pd.DataFrame(gridsearch_cv.cv_results_)

# Load the data
X_training = pd.read_csv(CV_RESULTS_DIR / '2024-06-06-190955__GridSearch_X_training.csv').drop(columns='Unnamed: 0')
X_testing = pd.read_csv(CV_RESULTS_DIR / '2024-06-06-190955__GridSearch_X_testing.csv').drop(columns='Unnamed: 0')
y_training = pd.read_csv(CV_RESULTS_DIR / '2024-06-06-190955__GridSearch_y_training.csv').drop(columns='Unnamed: 0')
y_testing = pd.read_csv(CV_RESULTS_DIR / '2024-06-06-190955__GridSearch_y_testing.csv').drop(columns='Unnamed: 0')

# Make all param prefixes the same
cv_results = utils.replace_column_prefix(cv_results, ['param_classifier__'],  'param_')

# Add prefix to the parameter names
PARAMS_TO_GET_BEST = [f'{PARAM_PREFIX}{param}' for param in PARAMS_TO_GET_BEST]
PARAMS_TO_GET_BEST.append(f'mean_test_{GRIDSEARCH_METRIC}')

# Sanitize the normalizer repr
cv_results['param_normalizer'] = cv_results['param_normalizer'].astype(str).str.split('(').str[0]

# Set dtype for the parameters of interest
cv_results['param_kernel'] = cv_results['param_kernel'].astype(str)
cv_results['param_normalizer'] = cv_results['param_normalizer'].astype(str)
cv_results['param_C'] = cv_results['param_C'].astype(np.float64)
cv_results['param_degree'] = cv_results['param_degree'].astype(int)
cv_results['param_coef0'] = cv_results['param_coef0'].astype(np.float64)
cv_results['param_tol'] = cv_results['param_tol'].astype(np.float64)
cv_results['param_class_weight'] = cv_results['param_class_weight'].astype(str)

# Permanently remove all rows with tol, class weights and normalizer not equal to CHOSEN_TOL, CHOSEN_CLASS_WEIGHT, CHOSEN_NORMALIZER
CHOSEN_TOL = 0.001
cv_results = cv_results[cv_results['param_tol'] == CHOSEN_TOL]

# Generate all possible combinations of class weights and normalizers
FIXED_META_PARAMS = {
    'param_class_weight': ['balanced', 'None'],
    'param_normalizer':   ['StandardScaler', 'MinMaxScaler'],
}
cv_results_copy, best_estimator = None, None
for combination in [dict(zip(FIXED_META_PARAMS.keys(), val_combination))
                    for val_combination in list(itertools.product(*FIXED_META_PARAMS.values()))]:
    CHOSEN_CLASS_WEIGHT = combination['param_class_weight']
    CHOSEN_NORMALIZER = combination['param_normalizer']
    print(f"\nPRESTUDY RESULTS for class_weight={repr(CHOSEN_CLASS_WEIGHT)} and normalizer={repr(CHOSEN_NORMALIZER)}")

    del cv_results_copy  # Delete the previous copy of cv_results_copy to free up memory
    cv_results_copy = deepcopy(cv_results)  # TODO: potentially re-load the pickle file instead of copying the DataFrame if memory is an issue
    cv_results_copy = cv_results_copy[cv_results_copy['param_class_weight'] == CHOSEN_CLASS_WEIGHT]
    cv_results_copy = cv_results_copy[cv_results_copy['param_normalizer'] == CHOSEN_NORMALIZER]

    # Find the best parameters and break ties
    alpha = 0
    beta = 0
    gamma = 0

    all_best_params = utils.get_best_params(cv_results_copy, GRIDSEARCH_METRIC, PARAMS_TO_GET_BEST, return_all=True, alpha=alpha, beta=beta, default_params=False)
    if all_best_params.shape[0] > 1:
        # Find and print difference in the rows
        print("Multiple best parameters found...")
        print("Difference among the best cv_results rows:")
        utils.print_differences(all_best_params)

        # Break ties by using the default parameters
        print("Breaking ties with default parameters...")
        all_best_params = utils.get_best_params(cv_results_copy, GRIDSEARCH_METRIC, PARAMS_TO_GET_BEST, return_all=True, alpha=alpha, beta=beta, default_params=SVC_DEFAULT_PARAMS)
        if all_best_params.shape[0] > 1:
            Warning("Multiple best parameters found after breaking ties with defaults... Please check the results manually.")
            breakpoint()
            all_best_params = all_best_params.iloc[0, :]
    best_params_series = all_best_params if type(all_best_params) is pd.Series else all_best_params.iloc[0]
    best_params = best_params_series.to_dict() if type(best_params_series) is pd.Series else best_params_series


    # Penalize (modify) scores in cv_results_copy according to:
    #   Adjusted Score = Score − α×ln(degree) − β×ln(std(Score)+1) − γ×ln(gamma+1)
    #                      (α only if kernel is poly)      (γ only if kernel is rbf or sigmoid)
    if alpha > 0 or beta > 0 or gamma > 0:
        # Create a mask for the polynomial kernel
        poly_kernel_mask = cv_results_copy['param_kernel'] == 'poly'
        rbf_sigmoid_kernel_mask = cv_results_copy['param_kernel'].isin(['rbf', 'sigmoid']) & (cv_results_copy['param_gamma'].isin(['scale', 'auto']) == False)
        # Adjust the score based on the conditions
        cv_results_copy[f'mean_test_{GRIDSEARCH_METRIC}'] = (
            cv_results_copy[f'mean_test_{GRIDSEARCH_METRIC}']
            - alpha * (np.log(cv_results_copy['param_degree']) * poly_kernel_mask)
            - beta * (np.log(cv_results_copy[f'mean_test_{GRIDSEARCH_METRIC}'] + 1))
            - gamma * (np.log(cv_results_copy['param_gamma'] + 1) * rbf_sigmoid_kernel_mask)
        )

    best_row = utils.get_row_from_best_params(cv_results_copy, best_params, CHOSEN_TOL, CHOSEN_CLASS_WEIGHT, GRIDSEARCH_METRIC)
    best_row = utils.replace_column_prefix(best_row, ['classifier__'],  'param_')

    # Set parameters for the best estimator using the row with the best parameters making sure all parameters ('param_') are included
    del best_estimator  # Delete the previous best_estimator to free up memory
    best_estimator = deepcopy(gridsearch_cv.best_estimator_)
    for param in best_row.iloc[0, :].index:
        if param.startswith(PARAM_PREFIX) and param not in ['param_normalizer', 'normalizer']:
            if best_row[param].values[0] == 'None':
                best_row[param].values[0] = None
            # if param == 'param_gamma' and best_row[param].values[0] not in ['scale', 'auto'] and type(best_row[param].values[0]) is not np.float64:
            #     best_row[param].values[0] = np.float64(best_row[param].values[0])
            best_estimator.set_params(**{param.replace('param_', 'classifier__'): best_row[param].values[0]})
        elif param in ['param_normalizer', 'normalizer']:
            if best_row[param].values[0] == 'StandardScaler':
                best_estimator.set_params(**{'normalizer': StandardScaler(copy=False)})
            elif best_row[param].values[0] == 'MinMaxScaler':
                best_estimator.set_params(**{'normalizer': MinMaxScaler(copy=False)})
            else:
                warnings.warn(f"Unknown normalizer: {best_row[param].values[0]}")
                breakpoint()
    print("Winning parameters/model based on roc_auc:\n", best_estimator)

    # Fit the model with full training set and evaluate using test set
    best_estimator.fit(X_training, y_training['FT5'])
    final_roc_auc = roc_auc_score(y_testing['FT5'], best_estimator.predict(X_testing))
    final_M = confusion_matrix(y_testing['FT5'], best_estimator.predict(X_testing))

    # Print the calculated final test metrics
    print("Confusion Matrix (final, separate test set):\n", final_M)
    print("ROC AUC (final, separate test set): ", final_roc_auc)

    # Print the CV classification report (mean validation metrics)
    for label in best_row:
        score = best_row[label].values[0]
        if 'train' in label or 'param' in label:
            continue
        print(f"{label}={score}")

    breakpoint_ = 1


