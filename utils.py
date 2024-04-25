"""
Utilities for Data Analysis

Authors:
Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Imports

## External imports
import pandas as pd


# %% Verbosity
# Is set in main. Module variable 'verbosity_level' should be imported from other modules.
verbosity_level = 1  # Default is 1. The higher, the more verbose. Can be 0, 1, 2, or 3.


# %% Random Seed
# Is set in main. Module variable 'random_seed' should be imported from other modules.
random_seed = 42  # Default is 42.


# %% Utility Functions

def make_binary(data, column, cutoff, copy=True):
    """
    Convert the values in a column to binary based on a cutoff value.

    :param data: A pandas DataFrame containing the data.
    :param column: The name of the column to convert to binary.
    :param cutoff: The value above which the binary value is 0, and below which the binary value is 1.
    :param copy: Whether to create a copy of the DataFrame (True) or modify it in place (False).
    :return: A pandas DataFrame with the specified column converted to binary.
    """
    # Copy the data to avoid modifying the original DataFrame
    if copy:
        data_copy = data.copy()
    else:
        data_copy = data
    
    # Convert the values in the specified column to binary
    data_copy[column] = data_copy[column].apply(lambda x: 1 if x < cutoff else 0)
    
    return data_copy


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
