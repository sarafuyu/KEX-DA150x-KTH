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
from IPython.core.display_functions import display
from sklearn.linear_model import BayesianRidge

# Local imports
import utils


# %% Setup

VERBOSE = utils.VERBOSITY_LEVEL  # get verbosity level
SEED = utils.RANDOM_SEED         # get random seed
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# %% Option 1: Simple Imputer

def create_simple_imputers(df, add_indicator=False, copy=True, strategy=("mean",)):
    """
    Impute missing values (e.g., with simple statistical values of each column) using SimpleImputer.

    Required Imports

    - ``numpy``
    - ``sklearn.impute.SimpleImputer``

    Parameters
    ==========
    :param df: The dataset to impute.
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
            "dataset": df,
        }
        simple_imputers.append(imputer_dict)
    
    return simple_imputers


# %% Option 2: Iterative Imputer


def create_iterative_imputers(df, estimators=(BayesianRidge(),), estimator_criterion='squared_error', max_iter=10, tol=0.001,
                              initial_strategy=("mean",), n_nearest_features=(10,),
                              imputation_order=("ascending",), add_indicator=True,
                              min_value='stat', max_value='stat', verbose=VERBOSE, imputers_only=False):
    """
    Impute missing values using IterativeImputer (experimental feature).

    Required Imports

    - ``sklearn.experimental.enable_iterative_imputer``
    - ``sklearn.impute.IterativeImputer``
    - ``sklearn.linear_model.BayesianRidge``

    :param df: The dataset to impute.
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
    :param verbose: The verbosity level.
    :param imputers_only: Whether to return only the imputers or the dataset dictionary as well.
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
                        "add_indicator": add_indicator,
                        "dataset": df,
                    }
                    iterative_imputers.append(imputer_dict)

    return iterative_imputers


# %% Option 3: KNN Imputer

def create_KNN_imputers(df, missing_values=np.nan, n_neighbours=(5,), weights=('uniform',),
                        metric=('nan_euclidean',), copy=True,
                        add_indicator=False, keep_empty_features=False):
    """
    Impute missing values using K-Nearest Neighbour Imputer.
    
    Required Imports:

    - ``sklearn.impute.KNNImputer``

    :param df: The dataset to impute.
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
                    "dataset": df,
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
    if type(data_dict_sparse['y_training']) is pd.Series:
        y_train, y_train_col_names = utils.dataframe_to_sparse(data_dict_sparse['y_training'].to_frame())
    else:
        y_train, y_train_col_names = utils.dataframe_to_sparse(data_dict_sparse['y_training'])
    if type(data_dict_sparse['y_testing']) is pd.Series:
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


def impute_data(data_dict, df, pipeline_start_time, imputer_dict=None, start_col_X=11, log=print):
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

    :param data_dict: A dictionary containing the imputer object and its configuration.
    :param df: The dataset to impute.
    :param pipeline_start_time: The start time of the pipeline.
    :param imputer_dict: A list of imputer dictionaries.
    :param start_col_X: The start index of the columns to impute.
    :param log: A logging function.
    :return: A dictionary containing the type of imputation, the imputed dataset, and the date
    of imputation.
    """
    # Start time
    iter_imp_start_time = datetime.now()

    # Extract the protein intensities
    X_selected = data_dict['X_training'].columns
    X = df.copy()[X_selected]

    # Impute missing values
    # A numpy ndarray is returned, not a dataframe, has no column names, need to convert back to dataframe
    if imputer_dict is None:
        imputer = data_dict['imputer']
    else:
        imputer = imputer_dict['imputer']
        data_dict['type'] = imputer_dict['type']
        data_dict['imputer'] = imputer_dict['imputer']
        data_dict['estimator'] = imputer_dict['estimator']
        data_dict['estimator_criterion'] = imputer_dict['estimator_criterion']
        data_dict['sample_posterior'] = imputer_dict['sample_posterior']
        data_dict['max_iter'] = imputer_dict['max_iter']
        data_dict['tol'] = imputer_dict['tol']
        data_dict['n_nearest_features'] = imputer_dict['n_nearest_features']
        data_dict['initial_strategy'] = imputer_dict['initial_strategy']
        data_dict['imputation_order'] = imputer_dict['imputation_order']
        data_dict['min_value'] = imputer_dict['min_value']
        data_dict['max_value'] = imputer_dict['max_value']
        data_dict['random_state'] = imputer_dict['random_state']
        data_dict['add_indicator'] = imputer_dict['add_indicator']

    if VERBOSE and data_dict['type'] == 'IterativeImputer':
        log('|--- IMPUTATION ---|')
        log(f'Imputation Method: ------------------ {data_dict["type"]}')
        log(f'Estimator: -------------------------- {data_dict["estimator"]}')
        log(f'Estimator criterion: ---------------- {data_dict["estimator_criterion"]}')
        log(f'Random State (seed): ---------------- {data_dict["random_state"]}')
        log(f'Missing Indicator Method: ----------- {data_dict["add_indicator"]}')
        log(f'Sample Posterior: ------------------- {data_dict["sample_posterior"]}')
        log(f'Imputation order: ------------------- {data_dict["imputation_order"]}')
        log(f'Initial imputation strategy: -------- {data_dict["initial_strategy"]}')
        log(f'Max iterations: --------------------- {data_dict["max_iter"]}')
        log(f'Tolerance: -------------------------- {data_dict["tol"]}')
        log(f'Num nearest features to consider: --- {data_dict["n_nearest_features"]}')
        log(f'Starting {data_dict["type"]} Imputation ...')

    X_imputed_arr = imputer.fit_transform(X)
    data_dict['imputer'] = imputer  # save the imputer object

    # Convert the imputed values separately back to a dataframe
    X_imputed = pd.DataFrame(X_imputed_arr, columns=X.columns)

    # Print the whole dataframe
    display(X_imputed.to_string())

    # Assert that there are no missing values in the imputed dataset
    # assert not any([not any(x) for x in X_imputed.isna()])

    # Save the imputed values as a separate dataframe in the dictionary
    data_dict['X_imputed'] = X_imputed  # save the imputed values

    # Save the date of imputation
    data_dict['date'] = pd.Timestamp.now()

    # Log time taken to run
    iter_imp_end_time = data_dict['date']
    timedelta = str(iter_imp_end_time - iter_imp_start_time).split('.')
    hms = timedelta[0].split(':')
    if VERBOSE:
        log(
            f"Imputation with {str(type(imputer)).split("'")[1].split('.')[-1]} using a {str(imputer.estimator).split("(")[0]} estimator finished {iter_imp_end_time.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"and took {hms[0]}h:{hms[1]}m:{hms[2]}s {timedelta[0]}s {timedelta[1][:3]}.{timedelta[1][3:]}ms to run.\n"
        )

    # Pickle imputation dict to disk
    try:
        # breakpoint()
        joblib.dump(imputer,
                    (PROJECT_ROOT/'out'/(pipeline_start_time.strftime("%Y-%m-%d-%H%M%S") + f'__{data_dict['type']}.pkl')))
    except:
        pass
    try:
        # breakpoint()
        pass
        X_imputed.to_csv(PROJECT_ROOT/'out'/(pipeline_start_time.strftime("%Y-%m-%d-%H%M%S")+f'__{data_dict['type']}_X_imputed.csv'))
        # joblib.dump((data_dict), PROJECT_ROOT/'out'/(pipeline_start_time.strftime("%Y-%m-%d-%H%M%S")+f'__{data_dict['type']}_imputed_data_dict.pkl'))
    except:
        pass
    try:
        X_imputed.to_csv(
            PROJECT_ROOT / 'out' / (
                        pipeline_start_time.strftime("%Y-%m-%d-%H%M%S") + f'__{data_dict['type']}_df.csv')
            )
    except:
        pass
    # breakpoint()

    return data_dict


# %% Main

def main():
    dataset = pd.read_csv(PROJECT_ROOT/'out'/'normalized_data.csv')
    dataset.head()


if __name__ == '__main__':
    main()
