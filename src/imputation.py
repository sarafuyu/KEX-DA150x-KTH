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
from pathlib import Path

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


def create_iterative_imputers(df, estimator=BayesianRidge(), max_iter=10, tol=1e-3,
                              initial_strategy=("mean",), n_nearest_features=(500,),
                              imputation_order=("ascending",), min_value=0,
                              max_value='10% higher than max'):
    """
    Impute missing values using IterativeImputer (experimental feature).

    Required Imports

    - ``sklearn.experimental.enable_iterative_imputer``
    - ``sklearn.impute.IterativeImputer``
    - ``sklearn.linear_model.BayesianRidge``

    :param estimator: The estimator to use at each step of the round-robin imputation.
    :param max_iter:
    :param tol:
    :param initial_strategy:
    :param n_nearest_features:
    :param imputation_order:
    :param min_value:
    :param max_value:
    :param df: The dataset to impute. Used to determine the min and max values for imputation.
    :return: A list of dictionaries, each containing an imputer object and its configuration.
    """
    # Imports
    
    # Explicitly require experimental feature
    from sklearn.experimental import enable_iterative_imputer  # noqa
    # Now we can import normally from sklearn.impute
    from sklearn.impute import IterativeImputer  # noqa
    from sklearn.linear_model import BayesianRidge
    
    if max_value == '10% higher than max':
        # We set max_value to 1.1*max value of the dataset, to avoid imputing values significantly
        # higher than the original data.
        max_value = 1.1 * utils.summary_statistics(df, range(11, df.shape[1]))[1].max()
    
    ## Create imputers with different configurations
    iterative_imputers = []
    for strat in initial_strategy:
        for order in imputation_order:
            imputer = IterativeImputer(
                estimator=estimator,
                missing_values=pd.NA,
                sample_posterior=False,  # TODO: should likely be set to True since there are multiple imputations but we need testing to evaluate return_std support. should be false for early stopping in max_iter
                max_iter=max_iter,
                tol=tol,
                n_nearest_features=n_nearest_features,
                initial_strategy=strat,
                imputation_order=order,
                skip_complete=False,
                min_value=min_value,
                max_value=max_value,
                verbose=VERBOSE,
                random_state=SEED,
                add_indicator=False,  # interesting for later, TODO: explore
                keep_empty_features=False,  # no effect: we have removed empty features in cleanup
            )
            imputer_dict = {
                "type": "IterativeImputer",
                "imputer": imputer,
                "max_iter": max_iter,
                "tol": tol,
                "n_nearest_features": n_nearest_features[1],
                "initial_strategy": strat,
                "imputation_order": order,
                "random_state": SEED,
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


def sparse_no_impute(data_dict: dict):
    """
    Convert the data to a sparse format and return it as a dictionary with type `sparse`.

    :param data_dict: (dict) Dictionary of no imputation copy.
    :return datadict_sparse: (list) Altered dictionary in list f
    """

    data_dict_sparse = deepcopy(data_dict)

    dataset, dataset_col_names = utils.dataframe_to_sparse(data_dict_sparse['dataset'])
    X_train, X_train_col_names = utils.dataframe_to_sparse(data_dict_sparse['X_training'])
    X_test, X_test_col_names = utils.dataframe_to_sparse(data_dict_sparse['X_testing'])
    y_train, y_train_col_names = utils.dataframe_to_sparse(data_dict_sparse['y_training'])
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


def impute_data(imp_dict, df, start_col=11):
    """
    Impute missing values in the dataset using the specified imputer.

    :param imp_dict: A dictionary containing the imputer object and its configuration.
    :param df: The dataset to impute.
    :param start_col: The start index of the columns to impute.
    :return: A dictionary containing the type of imputation, the imputed dataset, and the date
    of imputation.
    """
    # Isolate relevant data
    d_protein_intensities = df.iloc[:, start_col:]
    df_imputed = df.copy()
    df_imputed.iloc[:, start_col:] = pd.DataFrame(
        imp_dict['imputer'].fit_transform(d_protein_intensities),
        columns=d_protein_intensities.columns
    )
    
    # Add imputed dataset and date to dictionary
    imp_dict['dataset'] = df_imputed
    imp_dict['date'] = pd.Timestamp.now()
    
    return imp_dict


# %% Main

def main():
    dataset = pd.read_csv(PROJECT_ROOT/'out'/'normalized_data.csv')
    dataset.head()


if __name__ == '__main__':
    main()
