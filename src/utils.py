#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for Data Analysis

:Date: 2024-05-01
:Authors: Sara Rydell, Noah Hopkins

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
import math
# %% Imports

# Standard library imports
import re
import warnings
from copy import deepcopy
from decimal import Decimal
from functools import reduce
from pathlib import Path
from collections.abc import Sequence

## External library imports
import numpy as np
import pandas as pd
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from scipy.sparse import csr_matrix  # Needed for dataframe_to_sparse
from sklearn.feature_selection import f_classif
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import FunctionTransformer
from datetime import datetime


# %% Module-Global Variables

# Verbosity
# Is set in main. Module variable 'VERBOSITY_LEVEL' should be imported from other modules.
VERBOSITY_LEVEL = 1  # Default is 1. The higher, the more verbose. Can be 0, 1, 2, or 3.

# Random Seed
# Is set in main. Module variable 'RANDOM_SEED' should be imported from other modules.
RANDOM_SEED = 42  # Default is 42.


# %% Setup

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# %% Utility Functions

def make_binary(data, column_label, cutoffs, copy=True):
    """
    Bin the values in a column into a categorical variable based on cutoff values.
    Values below the cutoff are assigned 1, and values equal and above the cutoff are assigned 0.
    For N CUTOFFS, the data is divided into N+1 classes.
    
    Example:
    --------
    ``CUTOFFS=[a]`` will create a binary variable with the two classes:
        ``data.iloc[:,0:(a-1)]==1`` and
        ``data.iloc[:,a:]==0``.

    :param data: A pandas DataFrame containing the data.
    :param column_label: The name of the column to convert to binary.
    :param cutoffs: Values above which the binary value is 0, and below which the binary value is 1.
    :param copy: Whether to create a copy of the DataFrame (True) or modify it in place (False).
    :return: A pandas DataFrame with the specified column converted to binary.
    """
    # TODO: Add support for multiple cutoffs, only one cutoff supported for now. -- #
    if len(cutoffs) != 1:                                                           #
        raise ValueError("not implemented yet! "                                    #
                         "For now the `CUTOFFS` parameter "                         #
                         "must be a list with one element.")                        #
    else:                                                                           #
        cutoffs = cutoffs[0]                                                        #
    # TODO: ----------------------------------------------------------------------- #

    if type(data) is pd.DataFrame:
        df = data
    elif type(data) is dict:
        df = data['dataset']
    else:
        raise ValueError(
            "Argument data must be a pandas DataFrame or a dictionary "
            "with a 'dataset' key."
        )

    # Copy the data to avoid modifying the original DataFrame
    if copy:
        df_copy = df.copy()
    else:
        df_copy = df

    # [0, 21] [0, 22]
    # Convert the values in the specified column to binary
    df_copy[column_label] = df_copy[column_label].apply(lambda x: 1 if x < cutoffs else 0)

    if type(data) is dict:
        data['dataset'] = df_copy
        return data
    else:
        Warning(f"The data in {make_binary.__name__} is not a dictionary. Returning a tuple.")
        return df_copy


def make_continuous(data, column_label, copy=True):
    if type(data) is pd.DataFrame:
        df = data
    elif type(data) is dict:
        df = data['dataset']
    else:
        raise ValueError(
            "Argument data must be a pandas DataFrame or a dictionary "
            "with a 'dataset' key."
        )

    # Copy the data to avoid modifying the original DataFrame
    if copy:
        df_copy = df.copy()
    else:
        df_copy = df

    # Convert the values in the specified column float
    df_copy[column_label] = df_copy[column_label].astype(float)

    if type(data) is dict:
        data['dataset'] = df_copy
        return data
    else:
        Warning(f"The data in {make_binary.__name__} is not a dictionary. Returning a tuple.")
        return df_copy


def summary_statistics(data, cols):
    """
    Calculate summary statistics for the given data.

    :param data: A pandas DataFrame.
    :param cols: A list of column indices for which to calculate summary statistics.
    :return: A pandas DataFrame with summary statistics.
    """
    df = None
    if type(data) is dict:
        df = data['dataset']  # noqa
    elif type(data) is pd.DataFrame:
        df = data

    if type(cols[0]) is int:
        max_values = df.iloc[:, cols].max()
        min_values = df.iloc[:, cols].min()
        med_values = df.iloc[:, cols].median()
    else:
        max_values = df[cols].max()
        min_values = df[cols].min()
        med_values = df[cols].median()

    if VERBOSITY_LEVEL > 1:
        print(max_values)
        print(min_values)
        print(med_values)
    
    return min_values, max_values


def get_file_name(data_dict, pipeline_config=None):
    """
    Generate a file name based on the imputer type and configuration.

    If a pipeline configuration is provided, the file name will also include the pipeline
    configuration and the date will be the start time of the pipeline. Otherwise, the date
    will be the date of the imputation.

    :param data_dict: A dictionary containing the dataset, imputer type and configuration.
    :param pipeline_config: A dictionary containing the pipeline configuration.
    :return: A string with the file name.
    """
    fn_string = ''
    if data_dict['type'] == 'SimpleImputer':
        fn_string = (
                'TP꞉' + str(data_dict['type']) + '_' +
                'ST꞉' + str(data_dict['strategy']) + '_'
        )
    elif data_dict['type'] == 'IterativeImputer':
        fn_string = (
                'TP꞉' + str(data_dict['type']) + '_' +
                'IT꞉' + str(data_dict['max_iter']) + '_' +
                'TO꞉' + str(data_dict['tol']) + '_' +
                'NF꞉' + str(data_dict['n_nearest_features']) + '_' +
                'IS꞉' + str(data_dict['initial_strategy']) + '_' +
                'IM꞉' + str(data_dict['imputation_order']) + '_'
        )
    elif data_dict['type'] == 'KNNImputer':
        fn_string = (
                'TP꞉' + str(data_dict['type']) + '_' +
                'NE꞉' + str(data_dict['n_neighbors']) + '_' +
                'WE꞉' + str(data_dict['weights']) + '_' +
                'ME꞉' + str(data_dict['metric']) + '_'
        )
    elif data_dict['type'] == 'NAN_ELIMINATION':
        fn_string = (
                'TP꞉' + str(data_dict['type']) + '_'
        )
    elif data_dict['type'] == 'NO_IMPUTATION':
        fn_string = (
                'TP꞉' + str(data_dict['type']) + '_'
        )
    elif data_dict['type'] == 'sparse':
        # TODO: add info from sparse matrix type here later:
        fn_string = (
                'TP꞉' + str(data_dict['type']) + '_'
        )

    if pipeline_config:
        fn_string += (
             'CL꞉' + str(reduce(lambda a, b: str(a)+'｜'+str(b), pipeline_config['CUTOFFS'])) + '_' +
             'PR꞉' + str(pipeline_config['TEST_PROPORTION']) + '_'
             'RS꞉' + str(pipeline_config['SEED']) + '_'
        )
        if pipeline_config['SVC']:
            fn_string += 'CF꞉SVC' + '_'
            if 'clr' in pipeline_config:
                fn_string += (
                    'CP꞉' + str(pipeline_config['C']) + '_' +
                    'KP꞉' + str(pipeline_config['kernel']) + '_' +
                    'DE꞉' + str(pipeline_config['degree']) + '_' +
                    'GA꞉' + str(pipeline_config['gamma']) + '_' +
                    'CO꞉' + str(pipeline_config['coef0']) + '_' +
                    'TO꞉' + str(pipeline_config['tol']) + '_' +
                    'DF꞉' + str(pipeline_config['decision_function']) + '_'
                )
        elif pipeline_config['SVR']:
            fn_string += 'CF꞉SVR' + '_'
            if 'clr' in pipeline_config:
                fn_string += (
                    'CP꞉' + str(pipeline_config['C']) + '_' +
                    'KP꞉' + str(pipeline_config['kernel']) + '_' +
                    'DE꞉' + str(pipeline_config['degree']) + '_' +
                    'GA꞉' + str(pipeline_config['gamma']) + '_' +
                    'CO꞉' + str(pipeline_config['coef0']) + '_' +
                    'TO꞉' + str(pipeline_config['tol']) + '_' +
                    'ES꞉' + str(pipeline_config['epsilon']) + '_'
                )

    if pipeline_config:
        fn_string += 'DT꞉' + str(pipeline_config['START_TIME'].strftime('%Y-%m-%d-%H%M%S'))
    else:
        fn_string += 'DT꞉' + str(data_dict['date'].strftime('%Y-%m-%d-%H%M%S'))
    return fn_string


def export_imputed_data(data_dict, filename=None):
    """
    Export data to a CSV file.

    :param data_dict: A dictionary containing the imputer object, the imputed dataset,
        and the date of imputation.
    :param filename: The name of the file to which the imputed data will be saved. If None,
        the file will be saved with a name generated by the get_file_name function.
    """
    if filename is None:
        data_dict['dataset'].to_csv(PROJECT_ROOT/'out'/(get_file_name(data_dict)+'.csv'), index=False)
    else:
        data_dict['dataset'].to_csv(PROJECT_ROOT/'out'/filename, index=False)
    

def print_summary_statistics(data_dict, log=print, current_step='', start_column=11):
    """
    Print summary statistics for a given dataset.

    :param data_dict: A dictionary containing the imputer object, the imputed dataset,
        and the date of imputation.
    :param log: A logging and/or printing function.
    :param current_step: A string indicating the current step in the pipeline.
    :param start_column: The index of the first protein column in the dataset.
    """
    y = data_dict['dataset']['FT5'].to_frame(name='FT5')
    if 'X_imputed' in data_dict:
        X = data_dict['X_imputed']
    elif 'X' in data_dict:
        X = data_dict['X']
    elif 'X_selected' in data_dict:
        X = data_dict['X_selected']
    elif 'dataset' in data_dict:
        X = data_dict['dataset'].iloc[:, start_column:]
    else:
        warnings.warn("The data dictionary must contain a key 'X_imputed', 'X', 'X_selected', or 'dataset'.")
        return
    if VERBOSITY_LEVEL > 1:
        log(f"/__ SUMMARY STATISTICS {current_step}{'-'*(40-len(current_step))}")
        log(f"Dataset: ----------------------------- {get_file_name(data_dict)}")
        log(f"Number of entries (N): --------------- {X.shape[0]}")
        log(f"Number of features (X): -------------- {X.shape[1]}")
        if y['FT5'].dtypes in (np.float64, np.float32, float):
            log(f"Interval (y): ------------------------ [{y.min().min()}, {y.max().max()}]")
        else:
            log(f"Number of classes (y): --------------- {y.nunique()}")
            log(f"Classes: ----------------------------- {list(range(len(y.nunique())))}")
        log(f"Min y: ------------------------------- {y.min()}")
        log(f"Max y: ------------------------------- {y.max()}")
        log(f"Mean y: ------------------------------ {y.mean()}")
        log(f"Median y: ---------------------------- {y.median()}")
        log(f"Variance y: -------------------------- {y.var()}")
        log(f"Std deviation y: --------------------- {y.std()}")
        log(f"Min minimum X: ----------------------- {X.min().min()}\n")
        log(f"Max maximum X: ----------------------- {X.max().max()}")
        log(f"Mean mean X: ------------------------- {X.mean().mean()}")
        log(f"Mean median X: ----------------------- {X.median().mean()}")
        log(f"Mean variance X: --------------------- {X.var().mean()}")
        log(f"Mean std deviation X: ---------------- {X.std().mean()}\n")
    # if VERBOSITY_LEVEL > 2:
    #     log(f"Mean values: ------------------------- {df.mean()}")
    #     log(f"Median values: ----------------------- {df.median()}\n")


def dataframe_to_sparse(df, column_start=0):
    """
    Convert a Pandas DataFrame to a SciPy sparse matrix, preserving column names.

    Required imports:

    - ``pandas``
    - ``scipy.sparse.csr_matrix``

    :param df: The DataFrame to convert.
    :param column_start: The index of the first column to include in the sparse matrix.
    :return:
        sparse_matrix (scipy.sparse.csr_matrix): The converted sparse matrix.
        column_names (list): List of column names from the DataFrame.
    """
    sparse_matrix = csr_matrix(df.iloc[:, column_start:].values)  # Convert DataFrame to CSR sparse matrix
    column_names = df.iloc[:, column_start:].columns.tolist()  # Preserve the column names
    return sparse_matrix, column_names


def sparse_to_dataframe(sparse_matrix, column_names):
    """
    Convert a SciPy sparse matrix back to a Pandas DataFrame using provided column names.

    :param sparse_matrix: (scipy.sparse.csr_matrix) The sparse matrix to convert.
    :param column_names: (list) The column names for the DataFrame.
    :return df (pandas.DataFrame): The reconstructed DataFrame with original column names.
    """
    df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, columns=column_names)  # Convert sparse matrix to DataFrame
    return df


def get_dict_from_list_of_dict(dict_list, dict_key, dict_value):
    """
    Get a dictionary from a list of dictionaries based on a key-value pair.

    :param dict_list: A list of dictionaries.
    :param dict_key: The key to search for.
    :param dict_value: The value to search for.
    :return: The dictionary with the specified key-value pair.

    dict_key='type', dict_value='NO_IMPUTATION'
    """
    for d in dict_list:
        if d[dict_key] == dict_value:
            return d
    raise ValueError(f"Dictionary with key '{dict_key}' and value '{dict_value}' not found. "
                     f"Maybe you didn't set `NO_IMPUTATION = True`?")


def log_results(original_dataset, original_protein_start_col, config, data_dict, log=print):
    """
    Log the results of the pipeline.

    :param original_dataset: The original dataset.
    :param original_protein_start_col: The index of the first protein column in the original dataset.
    :param config: The configuration of the pipeline.
    :param data_dict: A dictionary containing the dataset, imputer type and configuration.
    :param log: A logging and/or printing function.
    :return: None
    """
    X = original_dataset.iloc[:, config['X_START_COLUMN_IDX']:]
    y = original_dataset['FT5'].to_frame(name='FT5')

    log(f"====================== PRE-PROCESSING ======================== |")
    log(f"/__ ORIGINAL DATASET {'_________' if X.isna().any().any() else '(IMPUTED)'}_________________________________")
    log(f"Missing indicators: ------------------ {config['ADD_INDICATORS']}")
    log(f"Shape (incl. misc columns): ---------- {original_dataset.shape}")
    log(f"X start column index: ---------------- {config['X_START_COLUMN_IDX']}")
    log(f"Number of entries (N): --------------- {X.shape[0]}")
    log(f"Number of features (X): -------------- {X.shape[1]}")
    log(f"Number of classes (y): --------------- {len(config['CUTOFFS']) + 1 if config['CUTOFFS'] else 'Continuous'}")
    log(f"Classes: ----------------------------- {list(range(len(config['CUTOFFS']) + 1)) if config['CUTOFFS'] else 'Continuous'}")
    log(f"Min y: ------------------------------- {y.min().mean()}")
    log(f"Max y: ------------------------------- {y.max().mean()}")
    log(f"Mean y: ------------------------------ {y.mean().mean()}")
    log(f"Median y: ---------------------------- {y.median().mean()}")
    log(f"Variance y: -------------------------- {y.var().mean()}")
    log(f"Std deviation y: --------------------- {y.std().mean()}")
    log(f"Min minimum X: ----------------------- {X.min().min()}")
    log(f"Max maximum X: ----------------------- {X.max().max()}")
    log(f"Mean mean X: ------------------------- {X.mean().mean()}")
    log(f"Mean median X: ----------------------- {X.median().mean()}")
    log(f"Mean variance X: --------------------- {X.var().mean()}")
    log(f"Mean std deviation X: ---------------- {X.std().mean()}\n")

    # Log categorization
    log(f"/__ CATEGORIZATION ____________________________________________")
    if config['PRECOMPUTED_XGB_SELECTED_DATA']:
        log(f"Pre-computed: ------------------------ True")
        log(f"Pre-computed file: ------------------- {str(config['PRECOMPUTED_XGB_SELECTED_DATA']).split('PycharmProjects\\')[-1]}")
    if not len(config['CUTOFFS']):
        log(f"Type: -------------------------------- No categorization (continuous)")
    else:
        log(f"Cut-offs: ---------------------------- {config['CUTOFFS']}")
        log(f"Number of classes: ------------------- {len(config['CUTOFFS']) + 1}\n")

    # Log training split
    log(f"/__ TRAINING SPLIT (FOR FEATURE SELECTION) ____________________")
    if config['PRECOMPUTED_XGB_SELECTED_DATA']:
        log(f"Seed: -------------------------------- {data_dict['random_state']}")
        log(f"Pre-computed: ------------------------ True")
        log(f"Pre-computed file: ------------------- {str(config['PRECOMPUTED_XGB_SELECTED_DATA']).split('PycharmProjects\\')[-1]}\n")

    # Log feature selection
    log(f"/__ FEATURE SELECTION _________________________________________")
    log(f"Feature selection: ------------------- {'XGB-RFECV' if config['SELECT_XGB'] else 'No info'}")
    if config['PRECOMPUTED_XGB_SELECTED_DATA']:
        log(f"Pre-computed: ------------------------ True")
        log(f"Pre-computed file: ------------------- {str(config['PRECOMPUTED_XGB_SELECTED_DATA']).split('PycharmProjects\\')[-1]}\n")
    else:
        log(f"Pre-computed: ------------------------ False\n")

    if config['STOP_AFTER_FEATURE_SELECTION']:
        return

    # Print imputation information
    log(f"/__ IMPUTATION ________________________________________________")
    if config['PRECOMPUTED_IMPUTED_X']:
        log(f"Pre-computed imputed data: ----------- True")
        log(f"Pre-computed imputed data file: ------ {str(config['PRECOMPUTED_IMPUTED_X']).split('PycharmProjects\\')[-1]}")
    if config['SIMPLE_IMPUTER']:
        log(f"Imputer: ----------------------------- SimpleImputer")
    if config['ITERATIVE_IMPUTER']:
        log(f"Imputer: ----------------------------- IterativeImputer")
    if config['KNN_IMPUTER']:
        log(f"Imputer: ----------------------------- KNNImputer")
    if config['NO_IMPUTATION']:
        log(f"Imputer: ----------------------------- No Imputation")
    log(f"Sparse: ------------------------------ {config['SPARSE_NO_IMPUTATION']}")
    if config['NAN_ELIMINATION']:
        log(f"NaN Elimination: --------------------- {config['NAN_ELIMINATION']}")
        log(f"NaN elimination drop: \n{'columns' if config['DROP_COLS_NAN_ELIM'] else 'rows'}\n")
    else:
        log(f"NaN Elimination: --------------------- {config['NAN_ELIMINATION']}\n")

    if not config['PRECOMPUTED_IMPUTED_X']:
        log(f"Date: ------------------------------- {data_dict['date']}")
        log(f"Imputer type: ----------------------- {data_dict['type']}")
        if data_dict['type'] == 'SimpleImputer':
            log(f"ADD_INDICATOR_SIMPLE_IMP: ------------ {config['ADD_INDICATOR_SIMPLE_IMP']}")
            log(f"COPY_SIMPLE_IMP: --------------------- {config['COPY_SIMPLE_IMP']}")
            log(f"STRATEGY_SIMPLE_IMP: ----------------- {config['STRATEGY_SIMPLE_IMP']}")
        elif data_dict['type'] == 'IterativeImputer':
            log(f"ESTIMATOR_ITER_IMP: ------------------ {config['ESTIMATOR_ITER_IMP']}")
            log(f"MAX_ITER_ITER_IMP: ------------------- {config['MAX_ITER_ITER_IMP']}")
            log(f"TOL_ITER_IMP: ------------------------ {config['TOL_ITER_IMP']}")
            log(f"INITIAL_STRATEGY_ITER_IMP: ----------- {config['INITIAL_STRATEGY_ITER_IMP']}")
            log(f"N_NEAREST_FEATURES_ITER_IMP: --------- {config['N_NEAREST_FEATURES_ITER_IMP']}")
            log(f"IMPUTATION_ORDER_ITER_IMP: ----------- {config['IMPUTATION_ORDER_ITER_IMP']}")
            log(f"MIN_VALUE_ITER_IMP: ------------------ {config['MIN_VALUE_ITER_IMP']}")
            log(f"MAX_VALUE_ITER_IMP: ------------------ {config['MAX_VALUE_ITER_IMP']}")
        elif data_dict['type'] == 'KNNImputer':
            log(f"N_NEIGHBOURS_KNN_IMP: ---------------- {config['N_NEIGHBOURS_KNN_IMP']}")
            log(f"WEIGHTS_KNN_IMP: --------------------- {config['WEIGHTS_KNN_IMP']}")
            log(f"METRIC_KNN_IMP: ---------------------- {config['METRIC_KNN_IMP']}")
            log(f"ADD_INDICATOR_KNN_IMP: --------------- {config['ADD_INDICATOR_KNN_IMP']}")
        elif data_dict['type'] == 'NAN_ELIMINATION':
            log(f"DROP_COLS_NAN_ELIM: ------------------ {config['DROP_COLS_NAN_ELIM']}")
        log(f"Shape: ------------------------------- {data_dict['dataset'].shape}\n")

    if config['STOP_AFTER_IMPUTATION']:
        return

    # Log training split
    log(f"/__ TRAINING SPLIT (FOR GRID SEARCH) __________________________")
    log(f"Pre-computed: ------------------------ False")
    log(f"Seed: -------------------------------- {config['SEED']}")
    log(f"Test proportion: --------------------- {config['TEST_PROPORTION']}")
    log(f"Training proportion: ----------------- {1 - config['TEST_PROPORTION']}")
    log(f"X start column index: ---------------- {config['X_START_COLUMN_IDX']}")
    log(f"y column label: ---------------------- {repr(config['Y_COLUMN_LABEL'])}\n")

    return


def log_grid_search_results(pipeline_config, dataset_dict, clf, accuracy, log=print):
    """
    Log the results of the grid search.

    :return: None
    """
    dataset = dataset_dict['dataset']

    # TODO: print feature scores from dataset_dict

    # Print summary statistics
    X_imputed = dataset_dict['X_imputed']
    y = dataset_dict['dataset']['FT5'].to_frame(name='FT5')
    log(f"/__ SUMMARY STATISTICS FINAL DATASET (UN-NORMALIZED) __________")
    log(f"Number of entries (N): --------------- {X_imputed.shape[0]}")
    log(f"Number of features (X): -------------- {X_imputed.shape[1]}")
    log(f"Number of classes (y): --------------- {len(pipeline_config['CUTOFFS']) + 1}")
    log(f"Classes: ----------------------------- {list(range(len(pipeline_config['CUTOFFS']) + 1))}")
    log(f"Min y: ------------------------------- {y.min().min()}")
    log(f"Max y: ------------------------------- {y.max().max()}")
    log(f"Mean y: ------------------------------ {y.mean().mean()}")
    log(f"Median y: ---------------------------- {y.median().mean()}")
    log(f"Variance y: -------------------------- {y.var().mean()}")
    log(f"Std deviation y: --------------------- {y.std().mean()}")
    log(f"Min minimum X: ----------------------- {X_imputed.min().min()}")
    log(f"Max maximum X: ----------------------- {X_imputed.max().max()}")
    log(f"Mean mean X: ------------------------- {X_imputed.mean().mean()}")
    log(f"Mean median X: ----------------------- {X_imputed.median().mean()}")
    log(f"Mean variance X: --------------------- {X_imputed.var().mean()}")
    log(f"Mean std deviation X: ---------------- {X_imputed.std().mean()}\n")

    log(f"======================== GRID SEARCH ========================= |")
    log(f"Grid search finished: ----- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if accuracy is not None:
        # Log best parameters
        if hasattr(clf, 'best_params_'):
            log("/__ BEST PARAMETER COMBINATION FOUND __________________________")
            best_parameters = clf.best_params_
            n = len(best_parameters)
            for i, param_name in enumerate(sorted(best_parameters.keys())):
                log(f"{param_name}: {'-'*(36-len(param_name))} {best_parameters[param_name]}{'\n' if i == n-1 else ''}")

        # Print classifier information
        if hasattr(clf.best_estimator_, '_final_estimator'):
            final_clf = repr(clf.best_estimator_._final_estimator).replace("\n", "")  # noqa
        else:
            final_clf = repr(clf.best_estimator_).replace("\n", "")
        final_clf = re.sub(pattern=' +', repl=' ', string=final_clf)
        log(f"/__ BEST CLASSIFIER ___________________________________________")
        log(f"Classifier: -------------------------- {final_clf}")
        log(f"Test accuracy: ----------------------- {accuracy}\n")

        # Print normalized data with chosen best normalization strategy
        if hasattr(clf.best_estimator_.named_steps, 'normalizer'):
            normalizer = clf.best_estimator_.named_steps['normalizer']
            if normalizer and type(normalizer) is not FunctionTransformer:
                log(f"/__ BEST NORMALIZER & FINAL NORMALIZED DATA ___________________")
                X_normalized = pd.DataFrame(normalizer.transform(X_imputed), columns=X_imputed.columns)
                log(f"Normalizer: -------------------------- {repr(normalizer).replace("\n", "")}")
                log(f"Number of entries (N): --------------- {X_normalized.shape[0]}")
                log(f"Number of features (X): -------------- {X_normalized.shape[1]}")
                log(f"Number of classes (y): --------------- {len(pipeline_config['CUTOFFS']) + 1}")
                log(f"Classes: ----------------------------- {list(range(len(pipeline_config['CUTOFFS']) + 1))}")
                log(f"Min y: ------------------------------- {y.min().min()}")
                log(f"Max y: ------------------------------- {y.max().max()}")
                log(f"Mean y: ------------------------------ {y.mean().mean()}")
                log(f"Median y: ---------------------------- {y.median().mean()}")
                log(f"Variance y: -------------------------- {y.var().mean()}")
                log(f"Std deviation y: --------------------- {y.std().mean()}")
                log(f"Min minimum X: ----------------------- {X_normalized.min().min()}")
                log(f"Max maximum X: ----------------------- {X_normalized.max().max()}")
                log(f"Mean mean X: ------------------------- {X_normalized.mean().mean()}")
                log(f"Mean median X: ----------------------- {X_normalized.median().mean()}")
                log(f"Mean variance X: --------------------- {X_normalized.var().mean()}")
                log(f"Mean std deviation X: ---------------- {X_normalized.std().mean()}\n")
            else:
                log(f"/__ BEST NORMALIZER ___________________________________________ ")
                log(f"Normalizer: -------------------------- None (best without)\n")

    # Save cross-validation results
    if clf and hasattr(clf, 'cv_results_'):
        cv_results = pd.DataFrame(clf.cv_results_)
        # cv_results = cv_results['final_accuracy'] = accuracy  # TODO: fix with individual accuracy scores from custom model
        # Save cross-validation results as CSV file
        grid_search_file_name = f'{pipeline_config['START_TIME'].strftime("%Y-%m-%d-%H%M%S")}__GridSearch_cv_results.csv'
        cv_results.to_csv(PROJECT_ROOT/'out'/grid_search_file_name, index=False)
        # Also log cross-validation results
        if VERBOSITY_LEVEL > 2:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                log(f"Cross-validation results:")
                n = len(cv_results.columns)
                for i, col in enumerate(cv_results.columns):
                    if col == 'params':
                        for param in cv_results[col]:
                            log(f"params: {param}{'\n' if i == n-1 else ''}")
                    else:
                        log(f"{col}: {cv_results[col]}{'\n' if i == n-1 else ''}")
        log(f"CV results saved to: ----------------- {grid_search_file_name}\n")
    return


def log_time(start_time, end_time, log=print, logfile: Path = None):
    """
    Log the time taken for the pipeline to run.

    :param logfile:
    :param start_time: The start time of the pipeline.
    :param end_time: The end time of the pipeline.
    :param log: A logging and/or printing function.
    """
    timedelta = str(end_time - start_time).split('.')
    hms = timedelta[0].split(':')
    if logfile:
        log(f"Logs saved to: ----------------------- {logfile}\n")
    log(
        f"Pipeline finished {end_time.strftime('%Y-%m-%d %H:%M:%S')}, "
        f"and took {hms[0]}h:{hms[1]}m:{hms[2]}s {timedelta[1][:3]}.{timedelta[1][3:]}ms to run.\n"
    )


def replace_column_prefix(d: pd.DataFrame | dict, prefixes: Sequence[str], repl: str = '') -> pd.DataFrame:
    """
    Strip the 'param_' prefix from the column names in a DataFrame or dict.

    :param d: The DataFrame or dict to process.
    :param prefixes: The prefixes to remove from the column names.
    :param repl: The replacement string.
    :return: The DataFrame with the 'param_' prefix removed from the column names.
    """
    for prefix in prefixes:
        if type(d) is dict:
            d = {k.replace(prefix, repl): v for k, v in d.items()}
        else:
            d.columns = d.columns.str.replace(prefix, repl)
    return d


def is_power_of_10(n):
    """Check if a number is a power of 10, considering the absolute value for negative numbers."""
    n = abs(n)  # Take the absolute value to handle negative numbers
    if n <= 0:
        return False
    log10_n = np.log10(n)
    return log10_n == int(log10_n)


def round_to_1(x):
    if x == 0:
        return x
    try:
        return round(x, -int(np.floor(np.log10(abs(x)))))
    except (OverflowError, RuntimeWarning, ZeroDivisionError) as e:
        if x != 0:
            print(f'Error successfully caught and handled in utils.round_to_1({x}):\n', e)
        return f"{x}"


def round_to_n(x, n):
    if x == 0:
        return x
    try:
        return round(x, -int(np.floor(np.log10(abs(x)))) + n)
    except (OverflowError, RuntimeWarning, ZeroDivisionError) as e:
        if x != 0:
            print(f'Error successfully caught and handled in utils.round_to_1({x}):\n', e)
        return f"{x}"


def scientific_notation_formatter(val, pos):
    """Formatter for scientific notation with superscript."""
    if val == 0:
        return "$0$"
    if val < 0:
        return f"-{scientific_notation_formatter(-val, pos)}"
    exponent = int(np.log10(val))
    coefficient = val / 10**exponent
    if coefficient == 1:
        return f"$10^{{{exponent}}}$"
    else:
        return f"${coefficient:.1f} \\cdot 10^{{{exponent}}}$"


def get_exponent(num):
    """Get back the exponent of a number generated by logspace for the parameter search.
    """
    if num == 0:
        return 'undefined'
    # Check for positive and negative numbers
    abs_num = abs(num)
    # Check for the exponent n where 10**n == abs_num
    n = math.log10(abs_num)
    # Round to the nearest 0.25
    # n_rounded = round(n * 4) / 4
    return n


def format_number(num, latex=False):
    if type(num) is str:
        return num
    if num == 0:
        return f'{repr(num)}'
    if num in range(0, 20):
        return f'{repr(num)}'
    if latex:
        exponent = get_exponent(num)
        return f'$10^{{{exponent}}}$'
    else:
        return f'{num:.2e}'


def get_best_params(cv_results_: pd.DataFrame, metric: str, parameters: list[str], return_all=False, force_select=False,
                    default_params=None, alpha=0, beta=0, gamma=0) -> pd.Series:
    """
Extracts the row of cv_results with the highest value in the specified metric column
and returns the values of the specified parameters from that row.

:param cv_results_: pd.DataFrame - The cross-validation results.
:param metric: str - The name of the metric column to find the highest value.
:param parameters: list[str] - The list of parameter names to extract from the best row.
:param return_all: bool - Whether to return all rows with the best rank.
:param force_select: bool - Whether to do a reverse search through fallback metrics if still tied.
:param default_params: dict - A dictionary with default parameters to use if no best row is found.
:param alpha: float - The penalty for higher degrees in the score.
:param beta: float - The penalty for higher standard deviations in the score.
:param gamma: float - The penalty for higher gamma values in the score.
:return: pd.Series - A series containing the best parameters.


SVC_DEFAULT_PARAMS = {'param_C': 1.0, 'param_degree': 3, 'param_coef0': 0.0}
"""
    cv_results_ = deepcopy(cv_results_)

    # Define the order of fallback metrics to resolve ties
    fallback_metrics = ('roc_auc', 'precision', 'recall')
    metric_name = f'mean_test_{metric}'

    # Penalize score for higher degrees according to:
    #   Adjusted Score = Score − α×ln(degree) − β×ln(std(Score)+1) − γ×ln(gamma+1)
    #   where α, β, and γ are the penalty coefficients (α only for polynomial kernels, γ only for rbf and sigmoid)
    if alpha > 0 or beta > 0 or gamma > 0:
        # Create a mask for the polynomial kernel
        poly_kernel_mask = cv_results_['param_kernel'] == 'poly'
        rbf_sigmoid_kernel_mask = ~poly_kernel_mask
        # Adjust the score based on the conditions
        cv_results_[f'mean_test_{metric}'] = (
                cv_results_[f'mean_test_{metric}']
                - alpha * (np.log(cv_results_['param_degree']) * poly_kernel_mask)
                - beta * (np.log(cv_results_[f'mean_test_{metric}'] + 1))
                - gamma * (np.log(cv_results_['gamma_float'] + 1) * rbf_sigmoid_kernel_mask)
        )

    # Helper function to find the best row based on a primary metric and its std deviation
    def find_best_row(primary_metric, df):
        max_value = df[f'mean_test_{primary_metric}'].max()
        best_rows_ = df[df[f'mean_test_{primary_metric}'] == max_value]

        # Sort df by the primary metric and std deviation
        best_rows_ = best_rows_.sort_values(by=[f'mean_test_{primary_metric}', f'std_test_{primary_metric}'], ascending=False)

        # If there is only one row with the best value, return it
        if best_rows_.shape[0] == 1:
            return best_rows_, True

        # If tied, try to break the tie by looking at the std deviation
        std_metric = f'std_test_{primary_metric}'
        if std_metric in best_rows_.columns:
            min_std = best_rows_[std_metric].min()
            best_rows_ = best_rows_[best_rows_[std_metric] == min_std]
            if best_rows_.shape[0] == 1:
                return best_rows_, True

        return best_rows_, False

    # ---------------------- HIGHEST PERFORMANCE METRIC ------------------------

    # Start by trying to find the best row based on the specified metric
    best_rows, best_found = find_best_row(metric, cv_results_)
    if best_found:
        return best_rows[parameters].iloc[0]

    # Check for ties by iterating over fallback metrics if there are still ties
    for fallback_metric in fallback_metrics:
        fallback_metric_name = f'mean_test_{fallback_metric}'
        if fallback_metric_name not in cv_results_.columns:
            continue
        best_rows, best = find_best_row(fallback_metric, best_rows)
        if best:
            return best_rows[parameters] if return_all else best_rows[parameters].iloc[0]

    # ------------------------ LOWEST MODEL COMPLEXITY -------------------------

    # If still tied, try to break ties using the lowest degree
    min_degree = best_rows['param_degree'].min()
    best_rows = best_rows[best_rows['param_degree'] == min_degree]
    if best_rows.shape[0] == 1:
        return best_rows[parameters] if return_all else best_rows[parameters].iloc[0]

    # If still tied, try to break ties using the coef0 closest to 0
    best_rows['distance'] = abs(best_rows['param_coef0'])
    best_rows = best_rows.sort_values(by='distance', ascending=True)
    best_rows = best_rows[best_rows['distance'] == best_rows.iloc[0]['distance']]
    if best_rows.shape[0] == 1:
        return best_rows[parameters] if return_all else best_rows[parameters].iloc[0]

    # If still tied, try to break ties using C closest to 1
    best_rows = best_rows.sort_values(by='param_C', ascending=True)
    best_rows = best_rows[best_rows['param_C'] == best_rows.iloc[0]['param_C']]
    if best_rows.shape[0] == 1:
        return best_rows[parameters] if return_all else best_rows[parameters].iloc[0]

    # ------------------------ CLOSEST TO THE MEAN -----------------------------

    # If still tied, see which row is closest to the mean metric value
    # mean_metric = cv_results_[metric_name].mean()
    # best_rows['distance'] = abs(best_rows[metric_name] - mean_metric)
    # best_rows = best_rows.sort_values(by='distance', ascending=True)
    # best_rows = best_rows[best_rows['distance'] == best_rows.iloc[0]['distance']]
    # if best_rows.shape[0] == 1:
    #     return best_rows[parameters] if return_all else best_rows[parameters].iloc[0]

    # ------------------------- CLOSEST TO DEFAULTS ----------------------------

    # If still tied, see which row is closest to the default parameters in the specified order
    if default_params:
        for param_name, default_value in default_params.items():
            if type(default_value) is str:
                best_rows['distance'] = pd.Series(best_rows[param_name] != default_value).astype(int)
            else:
                best_rows['distance'] = abs(best_rows[param_name] - default_value)
            best_rows = best_rows.sort_values(by='distance', ascending=True)
            best_rows = best_rows[best_rows['distance'] == best_rows.iloc[0]['distance']]
            if best_rows.shape[0] == 1:
                return best_rows[parameters] if return_all else best_rows[parameters].iloc[0]

    # ----------------------------- FIRST ROW ----------------------------------

    if force_select:
        # If still tied, try to break ties by the main metric again
        best_rows, best_found = find_best_row(metric, best_rows)
        if best_found:
            return best_rows[parameters] if return_all else best_rows[parameters].iloc[0]

        # If still tied, try to break ties by running through the whole process again for each fallback metric backwards
        for fallback_metric in reversed(fallback_metrics):
            fallback_metric_name = f'mean_test_{fallback_metric}'
            if fallback_metric_name not in cv_results_.columns:
                continue
            best_rows, best = find_best_row(fallback_metric, best_rows)
            if best:
                return best_rows[parameters] if return_all else best_rows[parameters].iloc[0]

    # If still tied, extract the first row
    warnings.warn(
        Warning(
            f"More than one parameter set still has the best rank when getting best according to {metric} and trying all fallback methods.\nExtracting the first row..."
            )
        )
    if return_all:
        return best_rows[parameters]
    else:
        return best_rows[parameters].iloc[0]


# %% Prestudy plotter helper functions

def adjust_color(color, amount=0.1):
    """
    Lightens or darkens the given color. The amount parameter specifies the amount of lightening or darkening.
    Set amount to a positive value to lighten the color, or a negative value to darken it.

    """
    from matplotlib.colors import to_rgba

    c = np.array(to_rgba(color))
    c[:3] = np.clip(c[:3] + amount, 0, 1)
    return c

def add_label_band(ax, top, bottom, label, *, spine_pos=-0.05, tip_pos=-0.02):
    """
    Helper function to add bracket around y-tick labels.

    Author: @tacaswell
    Source: https://stackoverflow.com/questions/67235301/vertical-grouping-of-labels-with-brackets-on-matplotlib

    Parameters
    ----------
    ax : matplotlib.Axes
        The axes to add the bracket to

    top, bottom : floats
        The positions in *data* space to bracket on the y-axis

    label : str
        The label to add to the bracket

    spine_pos, tip_pos : float, optional
        The position in *axes fraction* of the spine and tips of the bracket.
        These will typically be negative

    Returns
    -------
    bracket : matplotlib.patches.PathPatch
        The "bracket" Aritst.  Modify this Artist to change the color etc of
        the bracket from the defaults.

    txt : matplotlib.text.Text
        The label Artist.  Modify this to change the color etc of the label
        from the defaults.

    """
    import numpy.typing
    # grab the yaxis blended transform
    transform = ax.get_yaxis_transform()

    # add the bracket
    bracket = mpatches.PathPatch(
        mpath.Path(
            [  # noqa
                (tip_pos, top),
                (spine_pos, top),
                (spine_pos, bottom),
                (tip_pos, bottom),
            ]
        ),
        transform=transform,
        clip_on=False,
        facecolor="none",
        edgecolor="k",
        linewidth=2,
    )
    ax.add_artist(bracket)

    # add the label
    txt = ax.text(
        spine_pos - 0.01,
        (top + bottom) / 2,
        label,
        ha="right",
        va="center",
        rotation="vertical",
        clip_on=False,
        transform=transform,
    )

    return bracket, txt


def get_row_from_best_params(cv_results_, best_params_, chosen_tol, chosen_class_weight, gridsearch_metric):
    """
    Get the row from the cv_results_ DataFrame that matches the best parameters.

    :param cv_results_: (pd.DataFrame) The DataFrame containing the cross-validation results.
    :param best_params_: (dict) The best parameters for the model.
    :param chosen_tol: (float) The chosen tolerance for the model.
    :param chosen_class_weight: (str) The chosen class weight for the model.
    :param gridsearch_metric: (str) The metric used for the grid search.
    :return: (pd.Series) The row from the cv_results_ DataFrame that matches the best parameters.
    """
    # Convert best_params_ to a dictionary if it's not already
    if isinstance(best_params_, pd.Series):
        best_params_dict = best_params_.to_dict()
    elif isinstance(best_params_, pd.DataFrame):
        best_params_dict = best_params_.iloc[:, 0].to_dict()
    else:
        best_params_dict = best_params_

    # Create a mask to find the row in cv_results that matches best_params_ and the chosen tol and class_weight
    mask = pd.Series(True, index=cv_results_.index)  # Initialize the mask with True values
    for param_name, param_value in best_params_dict.items():
        mask &= cv_results_[param_name] == param_value
    mask &= cv_results_['param_tol'] == chosen_tol
    mask &= cv_results_['param_class_weight'] == chosen_class_weight

    # Pick the row with the best parameters and lowest rank
    best_rows = cv_results_[mask].sort_values(f'rank_test_{gridsearch_metric}')
    lowest_rank = best_rows[f'rank_test_{gridsearch_metric}'].min(numeric_only=True)
    best_row_ = best_rows[best_rows[f'rank_test_{gridsearch_metric}'] == lowest_rank]

    # If there is only one row with the best value, return it
    if best_row_.shape[0] != 1:
        warnings.warn(f"Multiple rows with the best value found!")
        breakpoint()

    return best_row_

def print_differences(df):
    differences = []
    num_rows = df.shape[0]

    # Iterate over each pair of rows using iloc for integer-based indexing
    for i in range(num_rows - 1):
        for j in range(i + 1, num_rows):
            # Iterate over each column
            for col in df.columns:
                if not pd.api.types.is_dict_like(df.iloc[i][col]):
                    if df.iloc[i][col] != df.iloc[j][col]:
                        differences.append((col, df.iloc[i][col], df.iloc[j][col]))
                else:
                    if df.iloc[i][col] != df.iloc[j][col]:
                        differences.append((col, df.iloc[i][col], df.iloc[j][col]))

    # Print the differences
    for diff in differences:
        print(f"'{diff[0]}': {repr(diff[1])} | {repr(diff[2])}")
