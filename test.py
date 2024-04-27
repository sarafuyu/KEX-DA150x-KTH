"""
Testing the pipeline on the KEX dataset.

Authors:
Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Configuration

## Verbosity
# The higher, the more verbose. Can be 0, 1, 2, or 3.
verbose = 1
# Level 0: No prints.
# Level 1: Only essential prints.
# Level 2: Essential prints and previews of data.
# Level 3: All prints.

## Logging
# Set to True to log the output to a file.
log = False

## Randomness seed
seed = 42

## Data extraction
path = 'normalised_data_all_w_clinical_kex_20240321.csv'

## Data imputation (SimpleImputer)
data_imputation = False  # If false, we skip imputation and remove rows with NaNs
# TODO: Need to implement smart NaN-elimination where we drop rows in valuable columns but
#  otherwise drop columns. Ask Cristina for valuable columns (biomarkers/promising biomarker
#  candidates)

# Vi har endast lagt in SimpleImputer här än så länge:
add_indicator = False  # interesting for later, TODO: explore
copy = True  # so we can reuse dataframe with other imputers
strategy = "mean"  # ["mean", "median", "most_frequent"]



## Data normalization
# See further down for the columns to normalize

## Train-test split & Feature selection
# For detailed configuration for each feature selection mode, see features.py
test_proportion = 0.2
k = 100
start_column = 11  # Column index to start from. Will split the data [cols:] into input
# and target variables. # noqa

## Feature selection
# See further down.

## Model training & fitting
# See further down.


# %% Imports

## External imports
import pandas as pd
import numpy as np
# import logging as log

## Local imports
import utils

utils.verbosity_level = verbose  # Set verbosity level for all modules
utils.random_seed = seed  # Set random seed for all modules

import cleaning


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

if data_imputation:
    from sklearn.impute import SimpleImputer
    
    imputer = SimpleImputer(
        missing_values=np.nan, strategy=strategy, copy=copy,
        add_indicator=add_indicator,  # interesting for later, TODO: explore
        keep_empty_features=False,
        # no effect: we have removed empty features in cleanup alrdy
    )
    
    data_dict = {"type": "SimpleImputer", "imputer": imputer, "strategy": strategy,
                    "add_indicator": add_indicator}
    
    # Isolate relevant data
    d_protein_intensities = dataset.iloc[:, start_column:]
    df_imputed = dataset.copy()
    df_imputed.iloc[:, start_column:] = pd.DataFrame(
        data_dict['imputer'].fit_transform(d_protein_intensities), columns=d_protein_intensities.columns
    )
    data_dict['dataset'] = df_imputed
    
else:
    df_imputed = dataset.copy().dropna(axis=1)
    data_dict = {'type': 'nan_elimination', 'dataset': df_imputed, 'date': pd.Timestamp.now()}
    
# Add imputed dataset and date to dictionary
data_dict['date'] = pd.Timestamp.now()

print("Validating imputation... NaNs in protein intensity columns:",
      df_imputed.iloc[:, start_column:].isna().any(axis=None))


# %% Data Normalization

# Columns to normalize
columns_to_normalize = list(range(11, df_imputed.shape[1]))
# Note: we only normalize the antibody/protein intensity columns (cols 11 and up)
# age, disease, FTs not normalized


# Calculate summary statistics
min_vals, max_vals = utils.summary_statistics(data_dict, columns_to_normalize)

# Create a copy of the DataFrame
df_normalized = df_imputed.copy()

## Try StandardScaler instead
# Docs: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
d_protein_intensities = df_imputed.copy().iloc[:, start_column:]


scaler.fit(d_protein_intensities)
print("Mean before:", scaler.mean_.mean())

df_normalized.iloc[:, start_column:] = scaler.transform(d_protein_intensities)
if verbose > 1:
    print("Scaled data:", df_normalized.iloc[:, start_column:])

d_protein_intensities_normalized = df_normalized.iloc[:, start_column:]
print("Variance after (should be ~1):", d_protein_intensities_normalized.var(axis=0).mean())
print("Mean after: (should be ~1)", d_protein_intensities_normalized.mean(axis=0).mean())
print("NaNs in normalized data?:",
      d_protein_intensities_normalized.isna().any(axis=None))


# # Normalize the specified columns
# TODO: try StandarScaler instead
# TODO: probably some bug here, rewrite [0,1]-interval normalization with a pandas Scaler instead
# index = dataset.shape[1] - len(columns_to_normalize)
# for col in columns_to_normalize:
#     col_range = max_vals.iloc[col - index] - min_vals.iloc[col - index]
#     # Normalize the current column using scaling formula
#     df_normalized.iloc[:, col] = (dataset.iloc[:, col] - min_vals.iloc[col - index]) / col_range
    
data_dict['dataset'] = df_normalized


# %% Train-test Split & Feature selection

from sklearn.model_selection import train_test_split

y_data = df_normalized['FT5']  # Vector for the target variable
X_data = df_normalized.iloc[:, start_column:]  # Matrix with variable input

# Splitting the dataset into training and testing sets (80% - 20%)
X_training, X_testing, y_training, y_testing = train_test_split(
    X_data, y_data,
    test_size=test_proportion, random_state=seed
)

data_dict['X_training'] = X_training
data_dict['X_testing'] = X_testing
data_dict['y_training'] = y_training
data_dict['y_testing'] = y_testing

print("Validating imputation... NaNs in X_training:",
      X_training.isna().any(axis=None))
print("Validating imputation... NaNs in X_testing:",
      X_testing.isna().any(axis=None))
print("Validating imputation... NaNs in y_training:",
      y_training.isna().any(axis=None))
print("Validating imputation... NaNs in y_testing:",
      y_testing.isna().any(axis=None))


# %% Feature Selection

from sklearn.feature_selection import SelectKBest, f_classif

# # Make the Northstar score binary
# northstar_cutoff = 0
# if northstar_cutoff:
#     from utils import make_binary
#
#     y_train = make_binary(y_train, column='FT5', cutoff=northstar_cutoff)

# Configure the SelectKBest selector (default: f_classif ANOVA F-test)
k_best_selector = SelectKBest(score_func=f_classif, k=k)


# TODO: FIX IN PIPELINE!!! (Convert back to DataFrame)
# Apply score function to data and store results in k_best_selector SelectKBest class instance
k_best_selector.fit(X_training, y_training)
# Get columns to keep and create new dataframe with those only
# Use evaluated scores to select the best k features
cols_idxs = k_best_selector.get_support(indices=True)
X_training_selected = X_training.iloc[:, cols_idxs]
X_testing_selected = X_testing.iloc[:, cols_idxs]


# Assert that the number of selected features is equal to k
if X_training_selected.shape[1] != k:
    raise ValueError(f"Selected Features Shape {X_training_selected.shape[1]} "
                     f"is not equal to k ({k})!")

# Update the dataset dictionary with the selected features
data_dict['feature_scores'] = k_best_selector
del data_dict['X_training']
del data_dict['X_testing']
data_dict['X_training'] = X_training_selected
data_dict['X_testing'] = X_testing_selected


# %% Model Training & Fitting

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Configuration
# for poly specification for degree will be needed
kernel_type = 'poly'  # 'linear', 'poly', 'sigmoid', 'precomputed'

# If no imputation has been done, return the dataset_dict as is
if data_dict['type'] == 'no_imputation':
    # TODO: Look into if SVC can handle NaN values.
    exit()

X_training = data_dict['X_training']
X_testing = data_dict['X_testing']
y_training = data_dict['y_training']
y_testing = data_dict['y_testing']


# Create the SVM model
svm_model = SVC(
    C=1.0,  # default, Regularization parameter
    kernel=kernel_type, degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False,
    tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
    decision_function_shape='ovr', break_ties=False, random_state=seed,
)
# Train SVM model
svm_model.fit(X_training, y_training)
if verbose:
    print("Finished with model training!")

# Validate SVM model
y_prediction = svm_model.predict(X_testing)
if verbose:
    print("Prediction finished!")

accuracy = accuracy_score(y_testing, y_prediction)
if verbose:
    print(f"Accuracy calculated: {accuracy}")

# Append SVM model to list
data_dict['svm'] = {'model': svm_model, 'accuracy': accuracy}


# %%

breakpoint()
