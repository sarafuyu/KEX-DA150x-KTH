#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data imputation

:Date: 2024-05-01
:Authors: Sara Rydell, Noah Hopkins

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Imports

## Standard library imports
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import joblib
## External imports
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge

# Local imports
import utils


# %% Setup

VERBOSE = utils.VERBOSITY_LEVEL  # get verbosity level
SEED = utils.RANDOM_SEED         # get random seed
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# %% Option 1: Simple Imputer

def create_simple_imputers(add_indicator=False, copy=True, strategy=("mean",)):
    """
    Impute missing values (e.g., with simple statistical values of each column) using SimpleImputer.

    Required Imports

    - ``numpy``
    - ``sklearn.impute.SimpleImputer``

    Parameters
    ==========
    :param add_indicator: Whether to add a missing indicator column to the dataset.
    :param copy: Whether to create a copy of the dataset (`True`) or modify it in place (`False`).
    :param strategy: The imputation strategies to use.
    :return: A list of dictionaries, each containing an imputer object and its configuration.
    """
    from sklearn.impute import SimpleImputer
    
    simple_imputers = []
    for strat in strategy:
        imputer = SimpleImputer(
            missing_values=pd.NA, # changed from np.nan
            strategy=strat,
            fill_value=None,
            copy=copy,
            add_indicator=add_indicator,  # interesting for later, TODO: explore
            keep_empty_features=False,  # no effect: we have removed empty features in cleanup alrdy
        )
        imputer_dict = {
            "type": "SimpleImputer",
            "imputer": imputer,
            "strategy": strat,
            "add_indicator": add_indicator,
        }
        simple_imputers.append(imputer_dict)
    
    return simple_imputers


# %% Option 2: Iterative Imputer


def create_iterative_imputers(df, estimators=(BayesianRidge(),), estimator_criterion='squared_error', max_iter=10, tol=0.001,
                              initial_strategy=("mean",), n_nearest_features=(10,),
                              imputation_order=("ascending",), add_indicator=True,
                              min_value='stat', max_value='stat', verbose=VERBOSE):
    """
    Impute missing values using IterativeImputer (experimental feature).

    Required Imports

    - ``sklearn.experimental.enable_iterative_imputer``
    - ``sklearn.impute.IterativeImputer``
    - ``sklearn.linear_model.BayesianRidge``


    :param estimators: The estimators to use at each step of the round-robin imputation.
    :param estimator_criterion: The criterion to use for the estimator.
    :param max_iter: The maximum number of imputation rounds to perform.
    :param tol: The tolerance/stopping criterion for imputation.
    :param initial_strategy: The imputation strategy to use for the initial imputation.
    :param n_nearest_features: The number of nearest features to take into consideration for imputation.
    :param imputation_order: The order in which to impute missing values.
    :param add_indicator: Whether to add a missing indicator column to the dataset.
    :param min_value: A cap on the minimum value to impute. If 'stat', the 10th percentile of the dataset is used.
    :param max_value: A cap on the maximum value to impute. If 'stat', the 90th percentile of the dataset is used.
    :param df: The dataset to impute. Used to determine the min and max values for imputation.
    :param verbose: The verbosity level.
    :return: A list of dictionaries, each containing an imputer object and its configuration.
    """
    # Imports
    
    # Explicitly require experimental feature
    from sklearn.experimental import enable_iterative_imputer  # noqa
    # Now we can import normally from sklearn.impute
    from sklearn.impute import IterativeImputer  # noqa
    from sklearn.linear_model import BayesianRidge

    if min_value == 'stat':
        m = utils.summary_statistics(df, range(11, df.shape[1]))[1].min()
        min_value = 0.25 * m if m > 0 else 1.75 * m
    if max_value == 'stat':
        # We set max_value to 1.1*max value of the dataset, to avoid imputing values significantly
        # higher than the original data.
        max_value = 1.75 * utils.summary_statistics(df, range(11, df.shape[1]))[1].max()  # Ask Cristina for exact values


    ## Create imputers with different configurations
    iterative_imputers = []
    for estimator in estimators:
        for n in n_nearest_features:
            for strat in initial_strategy:
                for order in imputation_order:
                    imputer = IterativeImputer(
                        estimator=estimator,
                        missing_values=np.nan,        # pd.NA gives error do not do that!!
                        sample_posterior=False,       # TODO: should likely be set to True since there are multiple imputations but we need testing to evaluate return_std support. should be false for early stopping in max_iter
                        max_iter=max_iter,
                        tol=tol,
                        n_nearest_features=n,
                        initial_strategy=strat,       # Best: mean
                        fill_value=None,              # Default
                        imputation_order=order,       # Default
                        skip_complete=False,          # Default
                        min_value=min_value,          # TODO: use data stats to set limits
                        max_value=max_value,
                        verbose=verbose,
                        random_state=SEED,
                        add_indicator=add_indicator,  # interesting for later, TODO: explore
                        keep_empty_features=False,    # no effect: we have removed empty features in cleanup
                    )
                    imputer_dict = {
                        "type": "IterativeImputer",
                        "imputer": imputer,
                        "estimator": estimator,
                        "estimator_criterion": estimator_criterion,  # "squared_error" is default
                        "sample_posterior": False,
                        "max_iter": max_iter,
                        "tol": tol,
                        "n_nearest_features": n,
                        "initial_strategy": strat,
                        "imputation_order": order,
                        "min_value": min_value,
                        "max_value": max_value,
                        "random_state": SEED,
                        "add_indicator": add_indicator
                    }
                    iterative_imputers.append(imputer_dict)
    
    return iterative_imputers


# %% Option 3: KNN Imputer

def create_KNN_imputers(missing_values=np.nan, n_neighbours=(5,), weights=('uniform',),
                        metric=('nan_euclidean',), copy=True,
                        add_indicator=False, keep_empty_features=False):
    """
    Impute missing values using K-Nearest Neighbour Imputer.
    
    Required Imports:

    - ``sklearn.impute.KNNImputer``

    :param missing_values: The placeholder for missing values.
    :param n_neighbours: List of number of neighbours to use.
    :param weights: List of weighting schemes.
    :param metric: List of distance metrics.
    :param copy: Whether to create a copy of the dataset (`True`) or modify it in place (`False`).
    :param add_indicator: Whether to add a missing indicator column to the dataset.
    :param keep_empty_features: Whether to keep empty features in the dataset.
    :return: A list of dictionaries, each containing an imputer object and its configuration.
    """
    from sklearn.impute import KNNImputer

    knn_imputers = []
    for num in n_neighbours:
        for weight in weights:
            for met in metric:
                imputer = KNNImputer(
                    missing_values=missing_values,  # default
                    n_neighbors=num,  # default = 5
                    weights=weight,
                    metric=met,  # default, callable has potential for later fitting
                    copy=copy,  # default, best option for reuse of dataframe dataset
                    add_indicator=add_indicator,  # default, interesting for later, TODO: explore
                    keep_empty_features=keep_empty_features  # default, we have removed empty
                                                             # features in cleanup
                )
                imputer_dict = {
                    'type': 'KNNImputer',
                    'imputer': imputer,
                    'missing_values': missing_values,
                    'n_neighbors': num,
                    'weights': weight,
                    'metric': met,
                    'add_indicator': add_indicator,
                }
                knn_imputers.append(imputer_dict)
    
    return knn_imputers


# %% Option 4: NaN Elimination

def eliminate_nan(df, drop_cols=True):
    """
    Drop columns with any NaN values in the dataset.

    :param df: The dataset to remove NaN values from.
    :param drop_cols: Whether to drop columns with NaN values (True) or rows (False).
    :return: A dictionary containing the NaN-eliminated dataset and the date of elimination.
    """
    df_dropped = df.copy().dropna(axis=(1 if drop_cols else 0))
    
    return [{'type': 'NAN_ELIMINATION', 'dataset': df_dropped, 'date': pd.Timestamp.now()}]


# %% Option 5: No Imputation

def no_imputer(df, copy=True):
    """
    Drop rows with any NaN values in the dataset.

    :param df: The dataset to remove NaN values from.
    :param copy: Whether to drop columns with NaN values (True) or rows (False).
    """
    if copy:
        df = df.copy()  # TODO: copy or not? Probably not necessary, but we do it in other imputers.
    return [{'type': 'NO_IMPUTATION', 'dataset': df, 'date': pd.Timestamp.now()}]


def sparse_no_impute(data_dict: dict, protein_start_col=11):
    """
    Convert the data to a sparse format and return it as a dictionary with type `sparse`.

    :param data_dict: (dict) Dictionary of no imputation copy.
    :param protein_start_col: (int) Index of the first protein column.
    :return datadict_sparse: (list) Altered dictionary in list f
    """

    data_dict_sparse = deepcopy(data_dict)

    dataset, dataset_col_names = utils.dataframe_to_sparse(data_dict_sparse['dataset'], column_start=protein_start_col)
    X_train, X_train_col_names = utils.dataframe_to_sparse(data_dict_sparse['X_training'])
    X_test, X_test_col_names = utils.dataframe_to_sparse(data_dict_sparse['X_testing'])
    if type(data_dict_sparse['y_training']) == pd.Series:
        y_train, y_train_col_names = utils.dataframe_to_sparse(data_dict_sparse['y_training'].to_frame())
    else:
        y_train, y_train_col_names = utils.dataframe_to_sparse(data_dict_sparse['y_training'])
    if type(data_dict_sparse['y_testing']) == pd.Series:
        y_test, y_test_col_names = utils.dataframe_to_sparse(data_dict_sparse['y_testing'].to_frame())
    else:
        y_test, y_test_col_names = utils.dataframe_to_sparse(data_dict_sparse['y_testing'])

    data_dict_sparse['type'] = 'sparse'
    data_dict_sparse['dataset'] = dataset
    data_dict_sparse['dataset_col_names'] = dataset_col_names
    data_dict_sparse['X_training'] = X_train
    data_dict_sparse['X_train_col_names'] = X_train_col_names
    data_dict_sparse['X_testing'] = X_test
    data_dict_sparse['X_test_col_names'] = X_test_col_names
    data_dict_sparse['y_training'] = y_train
    data_dict_sparse['y_train_col_names'] = y_train_col_names
    data_dict_sparse['y_testing'] = y_test
    data_dict_sparse['y_test_col_names'] = y_test_col_names
    data_dict_sparse['date'] = pd.Timestamp.now()

    return [data_dict_sparse]


def impute_data(imp_dict, df, start_col_X=11, add_indicator=True, log=print):
    """
    Impute missing values in the dataset using the specified imputer.

    Normally, the imputer is applied, in place, to the protein intensities in the main dataframe
    (under the 'dataset' key), the key which is used to pass the dataset to the next step
    in the pipeline.

    If add_indicator is `True`, the imputation is performed in place in the 'dataset' dataframe as
    usual, but a missing indicator matrix is concatenated to the end of the whole dataframe.
    The missing value indicator matrix df and imputed X df are also saved separately under the
    dictionary keys 'X_missing_values' and 'X_imputed', respectively.

    Feel free to use any of these 'dataset', 'X_missing_values', 'X_imputed' keys as needed in the
    rest of the pipeline. We can also remove any of these keys if they are not needed. Keep in mind
    that the 'dataset' key will always be present and is currently what is used to pass the dataset
    to the next step in the pipeline. I therefore recommend keeping the 'dataset' key as the main
    dataset to be passed forward.

    :param imp_dict: A dictionary containing the imputer object and its configuration.
    :param df: The dataset to impute.
    :param start_col_X: The start index of the columns to impute.
    :param size_X: The number of columns to impute counting from the start index.
    :param add_indicator: Whether to add a missing indicator column to the dataset.
    :param log: A logging function.
    :return: A dictionary containing the type of imputation, the imputed dataset, and the date
    of imputation.
    """
    # Start time
    iter_imp_start_time = datetime.now()
    if VERBOSE and imp_dict['type'] == 'IterativeImputer':
        log('IMPUTATION')
        log(f'Imputation Method: {imp_dict["type"]}')
        log(f'Estimator: {imp_dict["estimator"]}')
        log(f'Estimator criterion: {imp_dict["estimator_criterion"]}')
        log(f'Random State (seed): {imp_dict["random_state"]}')
        log(f'Missing Indicator Method (add_indicators): {imp_dict["add_indicator"]}')
        log(f'Sample Posterior: {imp_dict["sample_posterior"]}')
        log(f'Imputation order: {imp_dict["imputation_order"]}')
        log(f'Initial imputation strategy: {imp_dict["initial_strategy"]}')
        log(f'Max iterations: {imp_dict["max_iter"]}')
        log(f'Tolerance: {imp_dict["tol"]}')
        log(f'Num nearest features to consider for each imputation: {imp_dict["n_nearest_features"]}')
        log(f'Starting {imp_dict["type"]} Imputation ...')

    if imp_dict['add_indicator']:
        add_indicator = True

    # Set size to the number of columns from start_col_X to the end
    size_X = df.shape[1] - start_col_X

    # Isolate relevant data
    df_imputed = df.copy()

    # Extract the protein intensities
    X = df_imputed.iloc[:, start_col_X:]

    # Impute missing values
    # A numpy ndarray is returned, not a dataframe, has no column names, need to convert back to dataframe
    imputer = imp_dict['imputer']
    X_imputed_arr = imputer.fit_transform(X)
    imputed_features_index = imputer.indicator_.features_
    imp_dict['imputer'] = imputer  # save the imputer object

    # Convert the imputed values separately back to a dataframe
    X_imputed = pd.DataFrame(X_imputed_arr[:, 0:size_X], columns=X.columns)

    # Save the imputed values as a separate dataframe in the dictionary
    imp_dict['X_imputed'] = X_imputed  # save the imputed values

    # Insert and replace the original values with the imputed values in the original dataframe
    df_imputed.iloc[:, start_col_X:] = X_imputed

    if add_indicator:
        if VERBOSE > 3:
            for i, f in np.ndenumerate(imputed_features_index):
                # Check if all features have missing value indicators
                print(f"Missing value indicator {i} corresponds to feature {f}.")
                if i == len(X.columns)-1:
                    print("All features have missing value indicators.")
                    print(f"Num missing should be {2*i} and was {X_imputed_arr.shape[1]}")


        # Generate column names for the missing value indicator features
        X_column_names = X.columns
        # Drop unused missing value indicator column labels (if any)
        X_indicator_column_names = [
            f"{col_name}_missing" for col_name in X_column_names
            if X.columns.get_loc(col_name) in imputed_features_index
        ]

        # Convert the missing value indicators separately into a dataframe
        X_indicators = pd.DataFrame(X_imputed_arr[:, size_X:], columns=X_indicator_column_names)

        imp_dict['X_missing_values'] = X_indicators  # save the missing value indicators

        # Drop rows outside the selection range
        df_imputed = df_imputed.drop(df_imputed.columns[start_col_X + size_X:], axis=1)

        # Concatenate missing value indicators with the rest of the dataset
        X_indicators.index = df_imputed.index  # Ensure row indices match
        df_imputed = pd.concat([df_imputed, X_indicators], axis=1)

    # Return the imputed dataset
    imp_dict['dataset'] = df_imputed

    # Save the date of imputation
    imp_dict['date'] = pd.Timestamp.now()

    # Log time taken to run
    iter_imp_end_time = imp_dict['date']
    timedelta = str(iter_imp_end_time - iter_imp_start_time).split('.')
    hms = timedelta[0].split(':')
    if VERBOSE:
        log(
            f"Imputation with {str(type(imputer)).split("'")[1].split('.')[-1]} using a {str(imputer.estimator).split("(")[0]} estimator finished {iter_imp_end_time.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"and took {hms[0]}h:{hms[1]}m:{hms[2]}s {timedelta[0]}s {timedelta[1][:3]}.{timedelta[1][3:]}ms to run."
        )

    # Pickle imputation dict to disk
    joblib.dump(deepcopy(imp_dict), PROJECT_ROOT/'out'/Path(utils.get_file_name(imp_dict) + '_Imputed_Datadict.pkl'))

    return imp_dict


# %% Main

def main():
    dataset = pd.read_csv(PROJECT_ROOT/'out'/'normalized_data.csv')
    dataset.head()


if __name__ == '__main__':
    main()
