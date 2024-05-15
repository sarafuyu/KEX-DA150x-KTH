#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Northstar Prediction Estimation

:Date: 2024-05-01
:Authors: Sara Rydell, Noah Hopkins

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""

# %% Imports

# Standard library imports
import time
from collections import defaultdict
from itertools import product

# External imports
import joblib
import numpy as np
from pathlib import Path
from sklearn import svm

# Local imports
import utils
from model_selection._search import CustomCvGridSearch  # noqa


# %% Setup

VERBOSE = utils.VERBOSITY_LEVEL  # get verbosity level
SEED = utils.RANDOM_SEED         # get random seed
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# %% Functions

# SVM Classifier Model
# TODO: later, try probability=True

def find_best_svm_model(pipeline_config,
                        dataset_dict,
                        C_params=(1.0,),
                        kernels=('poly',),
                        degree_params=(3,),
                        gamma_params=('scale',),
                        coef0_params=(0.0,),
                        shrinking=True,
                        probability=False,
                        tol_params=(0.001,),
                        cache_size_params=(200,),
                        class_weight=None,
                        verb=VERBOSE,
                        max_iter_params=(-1,),
                        decision_function_params=('ovr',),
                        break_ties=False,
                        random_state=SEED,
                        log=print,
                        grid_search_verbosity=0,
                        return_train_score=False,
                        grid_search_scoring=0,
                        k_cv_folds=5,
                        calc_final_scores=True):
    """
    Create a list of Support Vector Machine models for classification and regression with
    different configurations.

    Required imports:

    - ``sklearn.svm.SVC``
    - ``sklearn.model_selection.GridSearchCV``

    """
    # If no imputation has been done, return the dataset_dict as is
    if dataset_dict['type'] == 'NO_IMPUTATION':
        # TODO: Test run to see if SVC can handle NaN values. Probably we need to convert into a sparse matrix first.
        return dataset_dict
    if dataset_dict['type'] == 'sparse':
        # TODO(Sara): do we need to change any SVR/grid search params to make it work with sparse data?
        #             If so, we can do that here.
        pass

    # Extract the training and testing data
    X_training = dataset_dict['X_training']
    X_testing = dataset_dict['X_testing']
    y_training = dataset_dict['y_training']
    y_testing = dataset_dict['y_testing']

    # Construct parameter grid
    param_grid = [
        {
            'C': C_params,
            'kernel': kernels,
            'degree': degree_params,
            'gamma': gamma_params,
            'coef0': coef0_params,
            'shrinking': shrinking,
            'probability': probability,
            'tol': tol_params,
            'cache_size': cache_size_params,
            'verbose': [grid_search_verbosity],
            'max_iter': max_iter_params,
            'decision_function_shape': decision_function_params,
            'break_ties': break_ties,
            'random_state': [random_state],
        }
    ]
    if class_weight is not None:
        param_grid[0]['class_weight'] = [class_weight]

    # Perform grid search
    clf = CustomCvGridSearch(
        estimator=svm.SVC(),
        param_grid=param_grid,
        scoring=grid_search_scoring,
        cv=k_cv_folds,
        verbose=grid_search_verbosity,
        return_train_score=return_train_score,
    )

    if calc_final_scores:
        clf = clf.fit_calc_final_scores(X_training, y_training['FT5'], X_testing, y_testing['FT5'])
    else:
        clf = clf.fit(X_training, y_training['FT5'])

    # Calculate and log best score (accuracy)
    if hasattr(clf, 'score'):
        test_accuracy = clf.score(X_testing, y_testing['FT5'])
    else:
        Warning("The classifier does not have a 'score' attribute. Was it fitted?")
        test_accuracy = None
    if verb:
        utils.log_grid_search_results(
            pipeline_config, dataset_dict, protein_start_col=11, clf=clf, accuracy=test_accuracy, log=log
            )

    dataset_dict['svm'] = {'clf': clf, 'test_accuracy': test_accuracy}

    joblib.dump(clf, PROJECT_ROOT/'out'/Path(utils.get_file_name(dataset_dict, pipeline_config) + '_CLF꞉SVC.pkl'))

    return dataset_dict


# %% SVR Model

def find_best_svr_model(pipeline_config,
                        dataset_dict,
                        kernels='rbf',
                        degree_params=3,
                        gamma_params='scale',
                        coef0_params=0.0,
                        tol_params=0.001,
                        C_params=1.0,
                        epsilon_params=0.1,
                        shrinking_params=True,
                        cache_size_params=200,
                        verb=VERBOSE,
                        max_iter_params=-1,
                        log=print,
                        grid_search_verbosity=0,
                        return_train_score=False,
                        grid_search_scoring=0,
                        k_cv_folds=5,
                        calc_final_scores=False):
    """
    Create a list of Support Vector Machine models for classification and regression with
    different configurations.

    Required imports:

    - ``sklearn.svm.SVR``
    - ``sklearn.model_selection.GridSearchCV``

    """
    # If no imputation has been done, return the dataset_dict as is
    if dataset_dict['type'] == 'NO_IMPUTATION':
        # TODO: Test run to see if SVR can handle NaN values. Probably we need to convert into a sparse matrix first.
        return dataset_dict
    if dataset_dict['type'] == 'sparse':
        # TODO: do we need to change any SVR/grid search params to make it work with sparse data?
        #             If so, we can do that here.
        pass

    # Extract the training and testing data
    X_training = dataset_dict['X_training']
    X_testing = dataset_dict['X_testing']
    y_training = dataset_dict['y_training']
    y_testing = dataset_dict['y_testing']

    # Convert to float
    y_testing = y_testing.astype(float)
    y_training = y_training.astype(float)

    # Construct parameter grid
    param_grid = [
        {
            'kernel': kernels,
            'degree': degree_params,
            'gamma': gamma_params,
            'coef0': coef0_params,
            'tol': tol_params,
            'C': C_params,
            'epsilon': epsilon_params,
            'shrinking': shrinking_params,
            'cache_size': cache_size_params,
            'verbose': [grid_search_verbosity],
            'max_iter': max_iter_params,
        }
    ]

    # Perform grid search
    clf = CustomCvGridSearch(
        estimator=svm.SVR(),
        param_grid=param_grid,
        scoring=grid_search_scoring,
        cv=k_cv_folds,
        verbose=grid_search_verbosity,
        return_train_score=return_train_score,
    )
    if calc_final_scores:
        clf = clf.fit_calc_final_scores(X_training, y_training, X_testing, y_testing)
    else:
        clf = clf.fit(X_training, y_training)

    # Log best parameters
    if verb and hasattr(clf, 'best_params_'):
        log("BEST PARAMETERS COMBINATION FOUND:")
        best_parameters = clf.best_params_
        for param_name in sorted(best_parameters.keys()):
            log(f"{param_name}: {'-'*(36-len(param_name))} {best_parameters[param_name]}")

    # Calculate and log best score (neg mean squared error)
    if hasattr(clf, 'score'):
        test_accuracy = clf.score(X_testing, y_testing)
    else:
        Warning("The classifier does not have a 'score' attribute.")
        test_accuracy = None
    if verb:
        log(f"Test accuracy of best SVR classifier: {test_accuracy}")

    dataset_dict['svr'] = {'clf': clf, 'test_accuracy': test_accuracy}

    joblib.dump(clf, PROJECT_ROOT/'out'/Path(utils.get_file_name(dataset_dict, pipeline_config) + '.pkl'))

    return dataset_dict


# %% LinearSVC Model

linear_model = svm.LinearSVR(
    epsilon=0.0,
    tol=0.0001,
    C=1.0,
    loss='epsilon_insensitive',
    fit_intercept=True,
    intercept_scaling=1.0,
    dual='warn',
    verbose=0,
    random_state=SEED,
    max_iter=1000
)


# %%

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

"""
# Ridge Regression
ridge_model = Ridge(alpha=1.0)

# Lasso Regression
lasso_model = Lasso(alpha=0.1) # adjust for different sparsity levels

ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)

# Making predictions
ridge_predictions = ridge_model.predict(X_test)
lasso_predictions = lasso_model.predict(X_test)

# Calculating Mean Squared Error
ridge_mse = mean_squared_error(y_test, ridge_predictions)
lasso_mse = mean_squared_error(y_test, lasso_predictions)

print("Ridge MSE:", ridge_mse)
print("Lasso MSE:", lasso_mse)
"""

# %% Main

def main():
    pass


if __name__ == '__main__':
    main()
