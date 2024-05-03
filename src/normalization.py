#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Normalization

:Date: 2024-05-01
:Authors: Sara Rydell, Noah Hopkins

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Imports

# Standard library imports
from pathlib import Path

# External library imports
import numpy as np
import pandas as pd

# Local imports
import utils


# %% Setup

VERBOSE = utils.VERBOSITY_LEVEL  # get verbosity level
SEED = utils.RANDOM_SEED         # get random seed


# %% Dataset Normalization


def min_max_normalization(data, start_column, min_val=0, max_val=1):
    """
    Normalize given columns in the given DataFrame using the Min-Max Normalization technique.

    Formula:
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min
             
    """
    from sklearn.preprocessing import MinMaxScaler
    
    df = None
    if type(data) is dict:
        df = data['dataset']
    elif type(data) is pd.DataFrame:
        df = data
    
    # Create a copy of the DataFrame
    df_normalized = df.copy()
    
    # Extract the protein intensities
    d_protein_intensities = df.copy().iloc[:, start_column:]
    
    # Create and fit MinMaxScaler
    scaler = MinMaxScaler(feature_range=(min_val, max_val), copy=False, clip=False)
    
    # Normalize protein intensities
    df_normalized.iloc[:, start_column:] = scaler.fit_transform(d_protein_intensities)
    
    # Return normalized dataset
    if type(data) is dict:
        data['dataset'] = df_normalized
        return data
    else:
        Warning(f"The data in {min_max_normalization.__name__} is not a dictionary. Returning a tuple.")
        return df_normalized
    

def std_normalization(data, start_column):
    """
    Normalize given columns in the given DataFrame using the Standardization technique.

    Formula:
        X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    
    """
    from sklearn.preprocessing import StandardScaler
    
    df = None
    if type(data) is dict:
        df = data['dataset']
    elif type(data) is pd.DataFrame:
        df = data
    
    # Create a copy of the DataFrame
    df_normalized = df.copy()
    
    # Extract the protein intensities
    d_protein_intensities = df.copy().iloc[:, start_column:]
    
    # Create and fit StandardScaler
    scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
    
    # Normalize protein intensities
    df_normalized.iloc[:, start_column:] = scaler.fit_transform(d_protein_intensities)
    
    # Return normalized dataset
    if type(data) is dict:
        data['dataset'] = df_normalized
        return data
    else:
        Warning(f"The data in {std_normalization.__name__} is not a dictionary. Returning a tuple.")
        return df_normalized
    
    
def log2_normalization(data, start_column):
    """
    Normalize given columns in the given DataFrame using the Log2 Normalization technique.

    Formula:
        X_log2 = log2(X + 1)
    
    """
    df = None
    if type(data) is dict:
        df = data['dataset']
    elif type(data) is pd.DataFrame:
        df = data
    
    # Create a copy of the DataFrame
    df_normalized = df.copy()
    
    # Extract the protein intensities
    d_protein_intensities = df.copy().iloc[:, start_column:]
    
    min_val = d_protein_intensities.min().min()
    def log2(x): return np.log2(x)
    if min_val == 0:
        def log2(x): return np.log2(x + 1)
    if min_val < 0:
        def log2(x): return np.log2(x - min_val)
    
    # Normalize protein intensities
    df_normalized.iloc[:, start_column:] = d_protein_intensities.applymap(log2)
    
    # Return normalized dataset
    if type(data) is dict:
        data['dataset'] = df_normalized
        return data
    else:
        Warning(f"The data in {log2_normalization.__name__} is not a dictionary. Returning a tuple.")
        return df_normalized
    
    
# %% Main
def main():
    
    
    # %% Configuration
    cleaned_data_path = 'cleaned_data.csv'
    output_data_path = 'normalized_data.csv'
    start_col = 11
    
    
    # %% Data Initialization
    
    # Load the data
    dataset = pd.read_csv(cleaned_data_path)
    if verbose:
        print("Data loaded successfully.")
    if verbose > 1:
        dataset.head()
        
    
    # %% Data Normalization
    
    # columns_to_normalize = list(range(start_col, dataset.shape[1]))
    
    ## Run statistics part for normalization
    stats = utils.summary_statistics(dataset, start_col)
    min_values, max_values = stats
    if verbose > 2:
        print(min_values)
        print(max_values)
    
    # Perform Normalization
    dataset_normalized = std_normalization(data=dataset, start_column=start_col)


    # %% Export Normalized Data
    
    # Save the normalized data to a CSV file
    dataset_normalized.to_csv(output_data_path, index=False)
    if verbose > 0:
        dataset_normalized  # noqa


# %%
if __name__ == '__main__':
    main()
