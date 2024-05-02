"""
Utilities for Data Analysis

:Date: 2024-05-01
:Authors: Sara Rydell, Noah Hopkins

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Imports

## External imports
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


def get_file_name(data_dict):
    """
    Generate a file name based on the imputer type and configuration.
    :param data_dict: A dictionary containing the imputer type and configuration.
    :return: A string with the file name.
    """
    fn_string = ''
    if data_dict['type'] == 'SimpleImputer':
        fn_string = (
                'ty-' + data_dict['type'] + '_' +
                'st-' + data_dict['strategy'] + '_' +
                'ai-' + str(data_dict['add_indicator']) + '_' +
                'dt-' + data_dict['date'].strftime('%Y-%m-%d-%H%M%S'))
    elif data_dict['type'] == 'IterativeImputer':
        fn_string = (
                'ty-' + data_dict['type'] + '_' +
                'it-' + data_dict['max_iter'] + '_' +
                'tl-' + str(data_dict['tol']) + '_' +
                'nf-' + data_dict['n_nearest_features'] + '_' +
                'is-' + data_dict['initial_strategy'] + '_' +
                'ip-' + data_dict['imputation_order'] + '_' +
                'rs-' + data_dict['random_state'] + '_' +
                'ai-' + str(data_dict['add_indicator']) + '_' +
                'dt-' + data_dict['date'].strftime('%Y-%m-%d-%H%M%S'))
    elif data_dict['type'] == 'KNNImputer':
        fn_string = (
                'ty-' + data_dict['type'] + '_' +
                'ne-' + data_dict['n_neighbors'] + '_' +
                'we-' + data_dict['weights'] + '_' +
                'me-' + data_dict['metric'] + '_' +
                'ai-' + str(data_dict['add_indicator']) + '_' +
                'dt-' + data_dict['date'].strftime('%Y-%m-%d-%H%M%S'))
    elif data_dict['type'] == 'nan_elimination':
        fn_string = (
                'ty-' + data_dict['type'] + '_' +
                'dt-' + data_dict['date'].strftime('%Y-%m-%d-%H%M%S'))
    elif data_dict['type'] == 'no_imputation':
        fn_string = (
                'ty-' + data_dict['type'] + '_' +
                'dt-' + data_dict['date'].strftime('%Y-%m-%d-%H%M%S'))
    # TODO: add sparse matrix type here later: elif data_dict['type'] == 's'
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


def shorten_param(param_name):
    """
    Remove components' prefixes in param_name.

    TODO: This function is not used in the current implementation. It is needed when using scikit learn's Pipelines
          but since we are not using them, we can probably remove this function.

    :param param_name: A string with the parameter name with a prefix to be removed. We assume the prefix is
        separated by a '__' separator.
    """
    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name


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
