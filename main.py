"""
This is the main file for the project. It will be used to run the project.

Authors:
Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Configuration

## Verbosity level
# The higher, the more verbose. Can be 0, 1, 2, or 3.
# Level 0: No prints.
# Level 1: Only essential prints.
# Level 2: Essential prints and previews of data.
# Level 3: All prints.
verbose = 1

## Logging status
# Set to True to log the output to a file.
log = False

## Data file path
path = 'normalised_data_all_w_clinical_kex_20240321.csv'

## Data imputation use status
# For detailed configuration for each imputation mode, see imputation.py
simple_imputer = False
iterative_imputer = False
KNN_imputer = False
nan_elimination = True
no_imputation = False

## Train-test split
test_proportion = 0.2

## Randomness seed
# Used for consistent test and training splits between models.
seed = 42

## Feature selection
# For detailed configuration for each feature selection mode, see features.py
k = 100

## Data normalization
# Will split the data [cols:] into input and target variables.
start_column = 11 # noqa


# %% Imports

## External imports
import pandas as pd
import numpy as np
import logging as log

## Local imports
import utils
utils.verbosity_level = verbose  # Set verbosity level for all modules
utils.random_seed = seed         # Set random seed for all modules

## Module imports
import cleaning
import imputation
import normalization
import features
import svm


# %% Data Loading

dataset = pd.read_csv(path)

if verbose:
    print("Data loaded successfully.")
if verbose == 2:
    dataset.head() # Pre-view first five rows


# %% Data Cleaning

# Clean and preprocess the data
dataset = cleaning.clean_data(dataset)


# %% Data Imputation

# Create data imputers
dataset_dicts = []
if simple_imputer:
    dataset_dicts = dataset_dicts + imputation.create_simple_imputers()
if iterative_imputer:
    dataset_dicts = dataset_dicts + imputation.create_iterative_imputers(dataset)
if KNN_imputer:
    dataset_dicts = dataset_dicts + imputation.create_KNN_imputers()

# Create imputed datasets using generated imputers
dataset_dicts = [imputation.impute_data(imputer_dict, dataset, 11)
                 for imputer_dict in dataset_dicts]

# Add NaN-eliminated and un-imputed datasets
if nan_elimination:
    dataset_dicts = dataset_dicts + imputation.eliminate_nan(dataset)
if no_imputation:
    dataset_dicts = dataset_dicts + imputation.no_imputer(dataset)
    

# %% Data Normalization

"""
# Columns to normalize
columns_to_normalize = list(range(11, dataset.shape[1]))
# Note: we only normalize the antibody/protein intensity columns (cols 11 and up)
# age, disease, FTs not normalized

# Normalize the datasets
dataset_dicts = [
    normalization.normalize(
        data_dict,
        columns_to_normalize,
        utils.summary_statistics(data_dict, columns_to_normalize)
    )
    for data_dict in dataset_dicts
]
"""

# Normalize the datasets

"""
dataset_dicts = [
    normalization.normalize(
        data_dict,
        start_column,
    )
    for data_dict in dataset_dicts
]
"""

from sklearn.preprocessing import StandardScaler

for data_dict in dataset_dicts:

    scaler = StandardScaler()
    d_protein_intensities = data_dict['dataset'].copy().iloc[:, start_column:]
    scaler.fit(d_protein_intensities)
    data_dict['dataset'].iloc[:, start_column:] = scaler.transform(d_protein_intensities)

# %% Categorization of Northstar score (y)

from utils import make_binary

# TODO: Fix bug in pipeline df_normalized['FT5'] = ... should be df_normalized = ...
# TODO: validate change
dataset_dicts = [
    utils.make_binary(df_normalized, column='FT5', cutoff=17, copy=False)
    for df_normalized in dataset_dicts
]

# %% Train-test Split

# Split data
# TODO: super wrong since split returns the four part parameters, should be in tuple format as well as be with keays as values
dataset_dicts = [
    features.split_data(
        dataset_dict,
        test_size=test_proportion,
        random_state=seed,
        col=start_column
    )
    for dataset_dict in dataset_dicts
]


# %% Feature Selection

dataset_dicts = [
    features.select_KBest(
        dataset_dict=data_dict,
        k=k
    )
    for data_dict in dataset_dicts
]

# dataset_dicts is now a list that contains dict with the following:
# 1. Imputed and normalized data sets, date of imputation, type of imputation, imputer objects,
#    summary statistics, ANOVA P-values and feature selection scores (F-values).
# 2. Input and target variables for the feature-selected training and testing data.


# %% Model Training & Fitting

# Create SVM models
dataset_dicts = [svm.create_svm_models(dataset_dict) for dataset_dict in dataset_dicts]


# %%

breakpoint()
