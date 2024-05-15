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
from sklearn.base import TransformerMixin, BaseEstimator, OneToOneFeatureMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer

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

    # If missing indicator columns are present, seek the first one
    end_col = len(df.columns)
    for col in df.columns:
        if 'missing' in col:
            end_col = df.columns.get_loc(col)
            break

    # Create a copy of the DataFrame
    df_normalized = df.copy()

    # Extract the protein intensities
    d_protein_intensities = df.copy().iloc[:, start_column:end_col]

    # Create and fit StandardScaler
    scaler = StandardScaler(copy=False, with_mean=True, with_std=True)

    # Normalize protein intensities (Note: not the missing indicator columns!)
    df_normalized.iloc[:, start_column:end_col] = scaler.fit_transform(d_protein_intensities)
    
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


# def identity(X):
#     return X
#
#
# class Normalizer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):  # Maybe OneToOneFeatureMixin is better?
#     def __init__(self, model=None):
#         self.model = model
#
#     def __repr__(self, N_CHAR_MAX=700):
#         if self.model is None or type(self.model) is FunctionTransformer:
#             return 'No Normalization'
#         else:
#             return repr(self.model)
#
#     def set_params(self, **parameters):
#         if hasattr(parameters, 'model'):
#             self.model = parameters['model']
#         elif self.model is not None:
#             self.model.set_params(**parameters)
#         return self
#
#     def fit(self, X, y=None):
#         if self.model is None:
#             raise ValueError("No Normalization selected")
#         if self.model is not None:
#             self.model.fit(X)
#         self.is_fitted_ = True
#         return self
#
#     def transform(self, X):
#         if not self.is_fitted_:
#             raise NotFittedError("This Normalizer instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
#         if self.model is None or type(self.model) is FunctionTransformer:
#             return X
#         return self.model.transform(X)



    # def get_feature_names_out(self, input_features=None) -> None:
    #     """Enables inheritance of the set_output() method so that output format can be set
    #
    #     Set the output format of a transformer to pd.DataFrame instead of np.array by
    #     calling set_output(transform_output="pandas") on the transformer object.
    #     """
    #     return super().get_feature_names_out(self)



# %% Main
def main():
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(Normalizer(normalizer_type='StandardScaler'))

# %%
if __name__ == '__main__':
    main()
