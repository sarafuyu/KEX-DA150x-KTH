#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for Data Analysis

:Date: 2024-05-01
:Authors: Sara Rydell, Noah Hopkins

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Imports

# Standard library imports
from functools import reduce
from pathlib import Path

## External library imports
import pandas as pd
from scipy.sparse import csr_matrix  # Needed for dataframe_to_sparse
from sklearn.feature_selection import f_classif
from sklearn.linear_model import BayesianRidge
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
        raise ValueError("Argument data must be a pandas DataFrame or a dictionary "
                         "with a 'dataset' key.")

    # Copy the data to avoid modifying the original DataFrame
    if copy:
        df_copy = df.copy()
    else:
        df_copy = df
    
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
    
    max_values = df.iloc[:, cols].max()
    min_values = df.iloc[:, cols].min()
    med_values = df.iloc[:, cols].median()
    
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
             'KF꞉' + str(pipeline_config['K_FEATURES']) + '_' +
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
    

def print_summary_statistics(data_dict, log=print, start_column=11):
    """
    Print summary statistics for a given dataset.

    :param data_dict: A dictionary containing the imputer object, the imputed dataset,
        and the date of imputation.
    :param log: A logging and/or printing function.
    :param start_column:
    """
    df = data_dict['dataset']
    if VERBOSITY_LEVEL > 1:
        log("|--- SUMMARY STATISTICS (POST-IMPUTATION) ---|")
        log(f"Dataset: {get_file_name(data_dict)}")
        log(f"Number of features (X): {df.shape[1]}")
        log(f"Number of entries (N): {df.shape[0]}")
        log(f"Northstar Score (y) mean: {df['FT5'].mean()}")
        log(f"Northstar Score (y) median: "
            f"{df['FT5'].median()}")
        log(f"Northstar Score (y) variance: "
            f"{df['FT5'].var()}")
        log(f"Northstar Score (y) std deviation: "
            f"{df['FT5'].std()}")
        log(f"Northstar Score (y) max: "
            f"{df['FT5'].max()}")
        log(f"Northstar Score (y) min: "
            f"{df['FT5'].min()}")
        log(f"Protein intensities (X) global mean: "
            f"{df.iloc[:, start_column:].mean().mean()}")
        log(f"Protein intensities (X) global median: "
             f"{df.iloc[:, start_column:].median().mean()}")
        log(f"Protein intensities (X) global variance: "
            f"{df.iloc[:, start_column:].var().mean()}")
        log(f"Protein intensities (X) std deviation: "
            f"{df.iloc[:, start_column:].std().mean()}")
        log(f"Protein intensities (X) max: "
            f"{df.iloc[:, start_column:].max().max()}")
        log(f"Protein intensities (X) min: "
            f"{df.iloc[:, start_column:].min().min()}")
    # if VERBOSITY_LEVEL > 2:
        # log(f"Mean values: {df.mean()}")
        # log(f"Median values: {df.median()}")
    if VERBOSITY_LEVEL > 1:
        log("\n")


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


def log_results(original_dataset, original_protein_start_col, config, log=print):
    """
    Log the results of the pipeline.

    :param original_dataset: The original dataset.
    :param original_protein_start_col: The index of the first protein column in the original dataset.
    :param config: The configuration of the pipeline.
    :param log: A logging and/or printing function.
    :return: None
    """
    log("|--- RESULTS ---|")
    log(f"ORIGINAL DATASET")
    log(f"File: {config['DATA_FILE']}")
    log(f"shape: {original_dataset.shape}")
    log(f"mean FT5: {original_dataset['FT5'].mean().mean()}")
    log(f"median FT5: {original_dataset['FT5'].median().mean()}")
    log(f"variance FT5: {original_dataset['FT5'].var().mean()}")
    log(f"std deviation FT5: {original_dataset['FT5'].std().mean()}")
    log(f"max FT5: {original_dataset['FT5'].max().max()}")
    log(f"min FT5: {original_dataset['FT5'].min().min()}")
    log(f"mean protein intensities: {original_dataset.iloc[:, original_protein_start_col:].mean().mean()}")
    log(f"median protein intensities: {original_dataset.iloc[:, original_protein_start_col:].median().mean()}")
    log(f"variance protein intensities: {original_dataset.iloc[:, original_protein_start_col:].var().mean()}")
    log(f"std deviation protein intensities: {original_dataset.iloc[:, original_protein_start_col:].std().mean()}")
    log(f"max protein intensities: {original_dataset.iloc[:, original_protein_start_col:].max().max()}")
    log(f"min protein intensities: {original_dataset.iloc[:, original_protein_start_col:].min().min()}\n")

    # Log imputation modes
    log(f"IMPUTATION MODES")
    log(f"SimpleImputer: {config['SIMPLE_IMPUTER']}")
    log(f"IterativeImputer: {config['ITERATIVE_IMPUTER']}")
    log(f"KNN_Imputer: {config['KNN_IMPUTER']}")
    log(f"NO_IMPUTATION: {config['NO_IMPUTATION']}")
    log(f"Sparse: {config['SPARSE_NO_IMPUTATION']}")
    log(f"NAN_ELIMINATION: {config['NAN_ELIMINATION']}")
    if config['NAN_ELIMINATION']:
        log(f"nan elimination drop: {'columns' if config['DROP_COLS_NAN_ELIM'] else 'rows'}\n")


    # Log data normalization
    log(f"DATA NORMALIZATION")
    log(f"FIRST_COLUMN_TO_NORMALIZE: {config['FIRST_COLUMN_TO_NORMALIZE']}\n")

    # Log categorization
    log(f"CATEGORIZATION")
    if not len(config['CUTOFFS']):
        log(f"No categorization (continuous variable)")
    else:
        log(f"Cut-offs: {config['CUTOFFS']}")
        log(f"Number of classes: {len(config['CUTOFFS']) + 1}")


    # Log training split
    log(f"Test proportion: {config['TEST_PROPORTION']}")
    log(f"Training proportion: {1 - config['TEST_PROPORTION']}")
    log(f"X start column index: {config['X_START_COLUMN_IDX']}")
    log(f"y column label: {repr(config['Y_COLUMN_LABEL'])}")

    # Log feature selection
    log(f"FEATURE SELECTION")
    log(f"Feature selection score function for KBest: {repr(config['SCORE_FUNC_FEATURES'])} (not necessarily used)")
    log(f"KBest k: {config['K_FEATURES']}\n")
    log(f"Sparse: {config['SPARSE_NO_IMPUTATION']}")
    if config['SPARSE_NO_IMPUTATION']:
        log(f"Not performing feature selection with KBest on sparse data.")
        log()

    # Log classifier
    log(f"CLASSIFIERS")
    log(f"SVC: {config['SVC']}\n")
    log(f"SVR: {config['SVR']}")

    return None


def log_grid_search_results(pipeline_config, dataset_dict, protein_start_col, clf, accuracy, log=print):
    """
    Log the results of the grid search.

    :return: None
    """
    log(f"|--- PROCESSED DATASET ---|")
    dataset = dataset_dict['dataset']

    # Print imputation information
    log(f"IMPUTATION")
    log(f"Date: {dataset_dict['date']}")
    log(f"Imputer type: {dataset_dict['type']}")
    if dataset_dict['type'] == 'SimpleImputer':
        log(f"ADD_INDICATOR_SIMPLE_IMP: {pipeline_config['ADD_INDICATOR_SIMPLE_IMP']}")
        log(f"COPY_SIMPLE_IMP: {pipeline_config['COPY_SIMPLE_IMP']}")
        log(f"STRATEGY_SIMPLE_IMP: {pipeline_config['STRATEGY_SIMPLE_IMP']}")
    elif dataset_dict['type'] == 'IterativeImputer':
        log(f"ESTIMATOR_ITER_IMP: {pipeline_config['ESTIMATOR_ITER_IMP']}")
        log(f"MAX_ITER_ITER_IMP: {pipeline_config['MAX_ITER_ITER_IMP']}")
        log(f"TOL_ITER_IMP: {pipeline_config['TOL_ITER_IMP']}")
        log(f"INITIAL_STRATEGY_ITER_IMP: {pipeline_config['INITIAL_STRATEGY_ITER_IMP']}")
        log(f"N_NEAREST_FEATURES_ITER_IMP: {pipeline_config['N_NEAREST_FEATURES_ITER_IMP']}")
        log(f"IMPUTATION_ORDER_ITER_IMP: {pipeline_config['IMPUTATION_ORDER_ITER_IMP']}")
        log(f"MIN_VALUE_ITER_IMP: {pipeline_config['MIN_VALUE_ITER_IMP']}")
        log(f"MAX_VALUE_ITER_IMP: {pipeline_config['MAX_VALUE_ITER_IMP']}")
    elif dataset_dict['type'] == 'KNNImputer':
        log(f"N_NEIGHBOURS_KNN_IMP: {pipeline_config['N_NEIGHBOURS_KNN_IMP']}")
        log(f"WEIGHTS_KNN_IMP: {pipeline_config['WEIGHTS_KNN_IMP']}")
        log(f"METRIC_KNN_IMP: {pipeline_config['METRIC_KNN_IMP']}")
        log(f"ADD_INDICATOR_KNN_IMP: {pipeline_config['ADD_INDICATOR_KNN_IMP']}")
    elif dataset_dict['type'] == 'NAN_ELIMINATION':
        log(f"DROP_COLS_NAN_ELIM: {pipeline_config['DROP_COLS_NAN_ELIM']}")
    elif dataset_dict['type'] == 'NO_IMPUTATION':
        log(f"No imputation performed.\n")
    log(f"shape: {dataset.shape}\n")

    # TODO: print feature scores from dataset_dict

    # Print summary statistics
    log(f"SUMMARY STATISTICS")
    log(f"Mean FT5: {dataset['FT5'].mean().mean()}")
    log(f"Median FT5: {dataset['FT5'].median().mean()}")
    log(f"Variance FT5: {dataset['FT5'].var().mean()}")
    log(f"Std deviation FT5: {dataset['FT5'].std().mean()}")
    log(f"Max FT5: {dataset['FT5'].max().max()}")
    log(f"Min FT5: {dataset['FT5'].min().min()}")
    log(f"Mean protein intensities: {dataset.iloc[:, protein_start_col:].mean().mean()}")
    log(f"Median protein intensities: {dataset.iloc[:, protein_start_col:].median().mean()}")
    log(f"Variance protein intensities: {dataset.iloc[:, protein_start_col:].var().mean()}")
    log(f"Std deviation protein intensities: {dataset.iloc[:, protein_start_col:].std().mean()}")
    log(f"Max protein intensities: {dataset.iloc[:, protein_start_col:].max().max()}")
    log(f"Min protein intensities: {dataset.iloc[:, protein_start_col:].min().min()}\n")

    # Print classifier information
    log(f"CLASSIFIER")
    log(f"Classifier: {repr(clf.best_estimator_)}")
    log(f"Test accuracy: {accuracy}")

    # Log best parameters
    if hasattr(clf, 'best_params_'):
        log("Best parameters combination found:")
        best_parameters = clf.best_params_
        for param_name in sorted(best_parameters.keys()):
            log(f"{param_name}: {best_parameters[param_name]}")

    # Save cross-validation results
    if clf and hasattr(clf, 'cv_results_'):
        cv_results = pd.DataFrame(clf.cv_results_)
        # cv_results = cv_results['final_accuracy'] = accuracy  # TODO: fix with individual accuracy scores from custom model
        # Save cross-validation results as CSV file
        grid_search_file_name = get_file_name(dataset_dict, pipeline_config) + '.csv'
        cv_results.to_csv(PROJECT_ROOT/'out'/grid_search_file_name, index=False)
        # Also log cross-validation results
        if VERBOSITY_LEVEL > 2:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                log(f"Cross-validation results:")
                for col in cv_results.columns:
                    if col == 'params':
                        for param in cv_results[col]:
                            log(f"params: {param}")
                    else:
                        log(f"{col}: {cv_results[col]}\n")

        log(f"Grid search completed!")
        log(f"Results saved to: {grid_search_file_name}\n")
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
    log(
        f"Pipeline finished {end_time.strftime('%Y-%m-%d %H:%M:%S')}, "
        f"and took {hms[0]}h:{hms[1]}m:{hms[2]}s {timedelta[0]}s {timedelta[1][:3]}.{timedelta[1][3:]}ms to run."
    )
    if logfile:
        log(f"Logs saved to: {logfile}\n")
