"""
This is the main file for the project. It will be used to run the project.

Authors:
Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Configuration

## Verbosity
verbose = 1  # The higher, the more verbose. Can be 0, 1, 2, or 3.

## Randomness seed
seed = 42

## Data extraction
path = 'normalised_data_all_w_clinical_kex_20240321.csv'

## Data imputation
# For detailed configuration for each imputation mode, see imputation.py
# Pick imputers to use:
simple_imputer = True
iterative_imputer = False
KNN_imputer = False
nan_elimination = True
no_imputation = True

## Train-test split & Feature selection
# For detailed configuration for each feature selection mode, see features.py
test_proportion = 0.2
k = 100
start_column = 11  # Column index to start from. Will split the data [cols:] into input
                   # and target variables. # noqa


# %% Imports

## External imports
import pandas as pd
import numpy as np

## Local imports
import utils
utils.verbosity_level = verbose  # Set verbosity level for all modules
utils.random_seed = seed         # Set random seed for all modules

import cleaning
import imputation
import normalization
import features
import svm


# %% Load Data

# Load the data
dataset = pd.read_csv(path)

if verbose:
    print("Data loaded successfully.")
if verbose == 2:
    dataset.head()  # Pre-view first five rows


# %% Data Cleaning

# Clean and preprocess the data
dataset = cleaning.clean_data(dataset)


# %% Data Imputation

# Create imputers
dataset_dicts = []
if simple_imputer:
    dataset_dicts = dataset_dicts + imputation.create_simple_imputers()
if iterative_imputer:
    dataset_dicts = dataset_dicts + imputation.create_iterative_imputers(dataset)
if KNN_imputer:
    dataset_dicts = dataset_dicts + imputation.create_KNN_imputers()

# Impute data using generated imputers
dataset_dicts = [imputation.impute_data(imputer_dict, dataset, 11)
                 for imputer_dict in dataset_dicts]

# Add NaN-eliminated and un-imputed datasets
if nan_elimination:
    dataset_dicts = dataset_dicts + imputation.eliminate_nan(dataset)
if no_imputation:
    dataset_dicts = dataset_dicts + imputation.no_imputer(dataset)
    

# %% Data Normalization

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


# %% Train-test Split & Feature selection

# Split data
dataset_dicts = [
    features.split_data(
        dataset_dict, test_size=test_proportion,
        random_state=seed, col=start_column
    )
    for dataset_dict in dataset_dicts
]

# Feature selection
dataset_dicts = [features.select_KBest(dataset_dict=data_dict, k=k, northstar_cutoff=0)
                 for data_dict in dataset_dicts]

# dataset_dicts is now a list that contains dict with the following:
# 1. Imputed and normalized data sets, date of imputation, type of imputation, imputer objects,
#    summary statistics, ANOVA P-values and feature selection scores (F-values).
# 2. Input and target variables for the feature-selected training and testing data.


# %% Model Training & Fitting

# Create SVM models
dataset_dicts = [svm.create_svm_models(dataset_dict) for dataset_dict in dataset_dicts]


# %%

breakpoint()
