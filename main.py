"""
This is the main file for the project. It will be used to run the project.

Authors:
Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""

# %%
## Imports & Data Initialization
import pandas as pd
import numpy as np

dataset = pd.read_csv('normalised_data_all_w_clinical_kex_20240321.csv')
dataset.head()  # Pre-view first five rows

# %%
# Local imports
from data_cleaning import clean_data
from data_imputation import (
    create_simple_imputers,
    create_iterative_imputers,
    create_KNN_imputers,
    eliminate_nan,
    no_imputer,
    impute_data,
)

# %%
## Data cleaning
dataset = clean_data(dataset)

# %%
## Data imputation

# Configuration
# Pick the imputers to use
simple_imputer = True
iterative_imputer = True
KNN_imputer = True
nan_elimination = True
no_imputation = True
# For detailed configuration, see data_imputation.py

# Create imputers
imputed_dataset_dicts = []
if simple_imputer:
    imputed_dataset_dicts = imputed_dataset_dicts + create_simple_imputers()
if iterative_imputer:
    imputed_dataset_dicts = imputed_dataset_dicts + create_iterative_imputers()
if KNN_imputer:
    imputed_dataset_dicts = imputed_dataset_dicts + create_KNN_imputers()

# %%
## Impute data

# Impute data using the previously generated imputers
imputed_dataset_dicts = [impute_data(imputer_dict, dataset, 10) for imputer_dict in imputed_dataset_dicts]

# Add NaN-eliminated and un-imputed datasets
if nan_elimination:
    imputed_dataset_dicts = imputed_dataset_dicts + eliminate_nan(dataset)
if no_imputation:
    imputed_dataset_dicts = imputed_dataset_dicts + no_imputer(dataset)
    
# %%
## Feature selection

