"""
This is the main file for the project. It will be used to run the project.
"""

# Imports
import pandas as pd
import numpy as np

# Local imports
from data_cleaning import clean_data
from data_imputation import (create_simple_imputers, create_iterative_imputers,
                             create_KNN_imputers, nan_elimination, impute_data)


# %%
## Data cleaning
dataset = clean_data()

# %%
# Data imputation

# Create imputers
simple_imputers = create_simple_imputers()
iterative_imputers = create_iterative_imputers()
KNN_imputers = create_KNN_imputers()


# %%
# Pick the imputers to use
imputed_dataset_dicts = simple_imputers + iterative_imputers + KNN_imputers

imputed_dataset_dicts = [impute_data(imputer_dict, dataset, 10) for imputer_dict in imputed_dataset_dicts]

# Handle Missing Values:
# Before anything can be done, ensure all NaN values are handled appropriately, either by
# imputation or by removing rows/columns with NaN values.
imputed_dataset_dicts + nan_elimination(dataset)

# %%

