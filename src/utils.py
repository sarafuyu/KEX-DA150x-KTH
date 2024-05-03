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

## External library imports
import pandas as pd
from scipy.sparse import csr_matrix # Needed for dataframe_to_sparse


# %% Verbosity

# Is set in main.
# Module variable 'verbosity_level' should be imported from other modules.
verbosity_level = 1  # Default is 1. The higher, the more verbose. Can be 0, 1, 2, or 3.


# %% Random Seed

# Is set in main.
# Module variable 'random_seed' should be imported from other modules.
random_seed = 42  # Default is 42.


# %% Utility Functions

def make_binary(data, column_label, cutoffs, copy=True):
    """
    Bin the values in a column into a categorical variable based on cutoff values.
    Values below the cutoff are assigned 1, and values equal and above the cutoff are assigned 0.
    For N cutoffs, the data is divided into N+1 classes.
    
    Example:
    --------
    ``cutoffs=[a]`` will create a binary variable with the two classes:
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
        raise ValueError("The cutoffs parameter must be a list with one element.")  #
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
    
    if verbosity_level > 1:
        print(max_values)
        print(min_values)
        print(med_values)
    
    return min_values, max_values


def get_file_name(data_dict, pipeline_config=None):
    """
    Generate a file name based on the imputer type and configuration.
    :param data_dict: A dictionary containing the imputer type and configuration.
    :param pipeline_config: A dictionary containing the pipeline configuration.
    :return: A string with the file name.
    """
    # svr = data_dict['svr']
    # cls = svr['clf']

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
    elif data_dict['type'] == 'nan_elimination':
        fn_string = (
                'TP꞉' + str(data_dict['type']) + '_'
        )
    elif data_dict['type'] == 'no_imputation':
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
             'CL꞉' + str(reduce(lambda a, b: str(a)+'｜'+str(b), pipeline_config['cutoffs'])) + '_' +
             'KF꞉' + str(pipeline_config['k_features']) + '_' +
             'PR꞉' + str(pipeline_config['test_proportion']) + '_'
             'RS꞉' + str(pipeline_config['seed']) + '_'
        )
        if pipeline_config['SVC']:
            fn_string += 'CL꞉SVC' + '_'
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
            fn_string += 'CL꞉SVR' + '_'
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
        fn_string += 'DT꞉' + str(data_dict['start_time'].strftime('%Y-%m-%d-%H%M%S'))
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
        data_dict['dataset'].to_csv(get_file_name(data_dict) + '.csv', index=False)
    else:
        data_dict['dataset'].to_csv(filename, index=False)
    

def print_summary_statistics(data_dict, logger=print, start_column=11):
    """
    Print summary statistics for a given dataset.

    :param data_dict: A dictionary containing the imputer object, the imputed dataset,
        and the date of imputation.
    :param logger: A logging and/or printing function.
    :param start_column:
    """
    df = data_dict['dataset']
    if verbosity_level > 1:
        logger("|--- SUMMARY STATISTICS (POST-IMPUTATION) ---|")
        logger(f"Dataset: {get_file_name(data_dict)}")
        logger(f"Number of features (X): {df.shape[1]}")
        logger(f"Number of entries (N): {df.shape[0]}")
        logger(f"Northstar Score (y) mean: {df['FT5'].mean()}")
        logger(f"Northstar Score (y) median: "
                    f"{df['FT5'].median()}")
        logger(f"Northstar Score (y) variance: "
                    f"{df['FT5'].var()}")
        logger(f"Northstar Score (y) std devition: "
                    f"{df['FT5'].std()}")
        logger(f"Northstar Score (y) max: "
                    f"{df['FT5'].max()}")
        logger(f"Northstar Score (y) min: "
                    f"{df['FT5'].min()}")
        
        logger(f"Protein intensities (X) global mean: "
                    f"{df.iloc[:, start_column:].mean().mean()}")
        logger(f"Protein intensities (X) global median: "
                    f"{df.iloc[:, start_column:].median().mean()}")
        logger(f"Protein intensities (X) global variance: "
                    f"{df.iloc[:, start_column:].var().mean()}")
        logger(f"Protein intensities (X) std deviation: "
                    f"{df.iloc[:, start_column:].std().mean()}")
        logger(f"Protein intensities (X) max: "
                    f"{df.iloc[:, start_column:].max().max()}")
        logger(f"Protein intensities (X) min: "
                    f"{df.iloc[:, start_column:].min().min()}")
    # if verbosity_level > 2:
        # logger(f"Mean values: {df.mean()}")
        # logger(f"Median values: {df.median()}")
    if verbosity_level > 1:
        logger("\n")


def dataframe_to_sparse(df):
    """
    Convert a Pandas DataFrame to a SciPy sparse matrix, preserving column names.

    Required imports:

    - ``pandas``
    - ``scipy.sparse.csr_matrix``

    :param df: The DataFrame to convert.
    :return:
        sparse_matrix (scipy.sparse.csr_matrix): The converted sparse matrix.
        column_names (list): List of column names from the DataFrame.
    """
    sparse_matrix = csr_matrix(df.iloc[:, 11:].values)  # Convert DataFrame to CSR sparse matrix
    column_names = df.columns.tolist()  # Preserve the column names
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

    dict_key='type', dict_value='no_imputation'
    """
    for d in dict_list:
        if d[dict_key] == dict_value:
            return d
    raise ValueError(f"Dictionary with key '{dict_key}' and value '{dict_value}' not found. "
                     f"Maybe you didn't set `no_imputation = True`?")


def log_results(original_dataset, original_protein_start_col, config, logger=print):
    """
    Log the results of the pipeline.

    :param original_dataset: The original dataset.
    :param original_protein_start_col: The index of the first protein column in the original dataset.
    :param config: The configuration of the pipeline.
    :param logger: A logging and/or printing function.
    :return: None
    """
    logger("|--- RESULTS ---|")
    logger(f"ORIGINAL DATASET")
    logger(f"File: {config['path']}")
    logger(f"shape: {original_dataset.shape}")
    logger(f"mean FT5: {original_dataset['FT5'].mean().mean()}")
    logger(f"median FT5: {original_dataset['FT5'].median().mean()}")
    logger(f"variance FT5: {original_dataset['FT5'].var().mean()}")
    logger(f"std deviation FT5: {original_dataset['FT5'].std().mean()}")
    logger(f"max FT5: {original_dataset['FT5'].max().max()}")
    logger(f"min FT5: {original_dataset['FT5'].min().min()}")
    logger(f"mean protein intensities: {original_dataset.iloc[:, original_protein_start_col:].mean().mean()}")
    logger(f"median protein intensities: {original_dataset.iloc[:, original_protein_start_col:].median().mean()}")
    logger(f"variance protein intensities: {original_dataset.iloc[:, original_protein_start_col:].var().mean()}")
    logger(f"std deviation protein intensities: {original_dataset.iloc[:, original_protein_start_col:].std().mean()}")
    logger(f"max protein intensities: {original_dataset.iloc[:, original_protein_start_col:].max().max()}")
    logger(f"min protein intensities: {original_dataset.iloc[:, original_protein_start_col:].min().min()}\n")

    # Log imputation modes
    logger(f"IMPUTATION MODES")
    logger(f"SimpleImputer: {config['simple_imputer']}")
    logger(f"IterativeImputer: {config['iterative_imputer']}")
    logger(f"KNN_Imputer: {config['KNN_imputer']}")
    logger(f"no_imputation: {config['no_imputation']}")
    logger(f"nan_elimination: {config['nan_elimination']}")
    logger(f"nan elimination drop: {'columns' if config['drop_cols_nan_elim'] else 'rows'}\n")

    # Log data normalization
    logger(f"DATA NORMALIZATION")
    logger(f"first_column_to_normalize: {config['first_column_to_normalize']}\n")

    # Log categorization
    logger(f"CATEGORIZATION")
    if not len(config['cutoffs']):
        logger(f"No categorization (continuous variable)")
    else:
        logger(f"Number of classes: {len(config['cutoffs']) + 1}")
        logger(f"Cut-offs: {config['cutoffs']}")

    # Log training split
    logger(f"Test_proportion: {config['test_proportion']}")
    logger(f"Training proportion: {1 - config['test_proportion']}")
    logger(f"X_start_column_idx: {config['X_start_column_idx']}")
    logger(f"y_column_label: {config['y_column_label']}")

    # Log feature selection
    logger(f"FEATURE SELECTION")
    logger(f"score_func: {repr(config['score_func'])}")
    logger(f"k: {config['k_features']}\n")

    # Log classifier
    logger(f"CLASSIFIERS")
    logger(f"SVR: {config['SVR']}")
    logger(f"SVC: {config['SVC']}\n")

    return None


def log_grid_search_results(pipeline_config, dataset_dict, protein_start_col, clf, accuracy, logger=print):
    """
    Log the results of the grid search.

    :return: None
    """
    logger(f"|--- PROCESSED DATASET ---|")
    dataset = dataset_dict['dataset']

    # Print imputation information
    logger(f"IMPUTATION")
    logger(f"Date: {dataset_dict['date']}")
    logger(f"Imputer type: {dataset_dict['type']}")
    if dataset_dict['type'] == 'SimpleImputer':
        logger(f"add_indicator_simple_imp: {pipeline_config['add_indicator_simple_imp']}")
        logger(f"copy_simple_imp: {pipeline_config['copy_simple_imp']}")
        logger(f"strategy_simple_imp: {pipeline_config['strategy_simple_imp']}")
    elif dataset_dict['type'] == 'IterativeImputer':
        logger(f"estimator_iter_imp: {pipeline_config['estimator_iter_imp']}")
        logger(f"max_iter_iter_imp: {pipeline_config['max_iter_iter_imp']}")
        logger(f"tol_iter_imp: {pipeline_config['tol_iter_imp']}")
        logger(f"initial_strategy_iter_imp: {pipeline_config['initial_strategy_iter_imp']}")
        logger(f"n_nearest_features_iter_imp: {pipeline_config['n_nearest_features_iter_imp']}")
        logger(f"imputation_order_iter_imp: {pipeline_config['imputation_order_iter_imp']}")
        logger(f"min_value_iter_imp: {pipeline_config['min_value_iter_imp']}")
        logger(f"max_value_iter_imp: {pipeline_config['max_value_iter_imp']}")
    elif dataset_dict['type'] == 'KNNImputer':
        logger(f"n_neighbours_KNN_imp: {pipeline_config['n_neighbours_KNN_imp']}")
        logger(f"weights_KNN_imp: {pipeline_config['weights_KNN_imp']}")
        logger(f"metric_KNN_imp: {pipeline_config['metric_KNN_imp']}")
        logger(f"add_indicator_KNN_imp: {pipeline_config['add_indicator_KNN_imp']}")
    elif dataset_dict['type'] == 'nan_elimination':
        logger(f"drop_cols_nan_elim: {pipeline_config['drop_cols_nan_elim']}")
    elif dataset_dict['type'] == 'no_imputation':
        logger(f"No imputation performed.\n")
    logger(f"shape: {dataset.shape}\n")

    # TODO: print feature scores from dataset_dict

    # Print summary statistics
    logger(f"SUMMARY STATISTICS")
    logger(f"mean FT5: {dataset['FT5'].mean().mean()}")
    logger(f"median FT5: {dataset['FT5'].median().mean()}")
    logger(f"variance FT5: {dataset['FT5'].var().mean()}")
    logger(f"std deviation FT5: {dataset['FT5'].std().mean()}")
    logger(f"max FT5: {dataset['FT5'].max().max()}")
    logger(f"min FT5: {dataset['FT5'].min().min()}")
    logger(f"mean protein intensities: {dataset.iloc[:, protein_start_col:].mean().mean()}")
    logger(f"median protein intensities: {dataset.iloc[:, protein_start_col:].median().mean()}")
    logger(f"variance protein intensities: {dataset.iloc[:, protein_start_col:].var().mean()}")
    logger(f"std deviation protein intensities: {dataset.iloc[:, protein_start_col:].std().mean()}")
    logger(f"max protein intensities: {dataset.iloc[:, protein_start_col:].max().max()}")
    logger(f"min protein intensities: {dataset.iloc[:, protein_start_col:].min().min()}\n")

    # Print classifier information
    logger(f"CLASSIFIER")
    logger(f"Classifier: {repr(clf.best_estimator_)}")
    logger(f"Test accuracy: {accuracy}")

    # Log best parameters
    if hasattr(clf, 'best_params_'):
        logger("Best parameters combination found:")
        best_parameters = clf.best_params_
        for param_name in sorted(best_parameters.keys()):
            logger(f"{param_name}: {best_parameters[param_name]}")

    # Save cross-validation results
    if clf and hasattr(clf, 'cv_results_'):
        cv_results = pd.DataFrame(clf.cv_results_)
        # Save cross-validation results as CSV file
        grid_search_file_name = get_file_name(dataset_dict, pipeline_config) + '.csv'
        cv_results.to_csv(grid_search_file_name, index=False)
        # Also log cross-validation results
        if verbosity_level > 2:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                logger(f"Cross-validation results:")
                for col in cv_results.columns:
                    if col == 'params':
                        for param in cv_results[col]:
                            logger(f"params: {param}")
                    else:
                        logger(f"{col}: {cv_results[col]}\n")

        logger(f"Grid search completed. Grid search results saved to {grid_search_file_name}!\n")
    return


def log_time(start_time, end_time, logger=print):
    """
    Log the time taken for the pipeline to run.

    :param start_time: The start time of the pipeline.
    :param end_time: The end time of the pipeline.
    :param logger: A logging and/or printing function.
    """
    timedelta = str(end_time - start_time).split('.')
    hms = timedelta[0].split(':')
    logger(
        f"Pipeline finished {end_time.strftime('%Y-%m-%d %H:%M:%S')}, "
        f"and took {hms[0]}h:{hms[1]}m:{hms[2]}s {timedelta[0]}s {timedelta[1][:3]}.{timedelta[1][3:]}ms to run.\n"
    )