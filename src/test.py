#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing the pipeline on the KEX dataset.

:Date: 2024-05-01
:Authors: Sara Rydell, Noah Hopkins

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
from datetime import datetime
from pathlib import Path

# %% Configuration

## Verbosity
# The higher, the more verbose. Can be 0, 1, 2, or 3.
verbose = 1
# Level 0: No prints.
# Level 1: Only essential prints.
# Level 2: Essential prints and previews of data.
# Level 3: All prints.

## Randomness seed
seed = 42

## Data extraction
# Path to dataset
START_TIME = datetime.now()
PROJECT_ROOT = Path(__file__).parents[1]
DATA_FILE: Path = PROJECT_ROOT/'data'/'dataset'/'normalised_data_all_w_clinical_kex_20240321.csv'
# Clean and preprocess the data

path = DATA_FILE

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

## Logging
# Set to True to log the output to a file.
log = True
logfile: str = 'test-log-' + datetime.now().strftime("%Y-%m-%d-%H%M%S") + '.log'


# %% Imports

## External imports
import pandas as pd
import numpy as np
import logging
import sys

## Local imports
import cleaning
import utils
utils.VERBOSITY_LEVEL = verbose  # Set verbosity level for all modules
utils.RANDOM_SEED = seed         # Set random seed for all modules


# %% Logging

# Set up logging

handlers = []
if log:
    file_handler = logging.FileHandler(filename=logfile)  # TODO: generate unique filename
    handlers.append(file_handler)
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers.append(stdout_handler)

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s] {%(filename)s:%(lineno)d}: %(message)s',
                    handlers=handlers)

logger = logging.getLogger('LOGGER_NAME')

if verbose:
    logger.info("/__ SVM CLASSIFICATION ________________________________________")
    logger.info("|--- HYPERPARAMETERS ---|")
    logger.info(f"Dataset: {path}")
    logger.info(f"Impute: {data_imputation}")
    logger.info(f"add_indicator: {add_indicator}")
    logger.info(f"Random seed: {seed}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Test proportion: {test_proportion}")
    logger.info(f"K: {k}")
    logger.info(f"Start column: {start_column}")
    

# %% Load Data

# Load the data
dataset = pd.read_csv(path)

if verbose > 1:
    logger.info("Data loaded successfully.")
if verbose == 2:
    dataset.head()  # Pre-view first five rows
if verbose:
    logger.info("\n")


# %% Data Cleaning

dataset = cleaning.clean_data(dataset, verbose=1, log=logger.info, date=START_TIME, dataset_path=DATA_FILE)


def make_binary(data, column_label, cutoffs, copy=True):
    """
    Bin the values in a column into a categorical variable based on cutoff values.
    Values below the cutoff are assigned 1, and values equal and above the cutoff are assigned 0.
    For N CUTOFFS, the data is divided into N+1 classes.

    Example:
    --------
    ``CUTOFFS=[a]`` will create a binary variable with the two classes:
        ``data.iloc[:,0:(a-1)]==1`` and
        ``data.iloc[:,a:]==0``.

    :param data: A pandas DataFrame containing the data.
    :param column_label: The name of the column to convert to binary.
    :param cutoffs: Values above which the binary value is 0, and below which the binary value is 1.
    :param copy: Whether to create a copy of the DataFrame (True) or modify it in place (False).
    :return: A pandas DataFrame with the specified column converted to binary.
    """
    # TODO: Add support for multiple cutoffs, only one cutoff supported for now. -- #
    if len(cutoffs) != 1:  #
        raise ValueError(
            "not implemented yet! "  #
            "For now the `CUTOFFS` parameter "  #
            "must be a list with one element."
            )  #
    else:  #
        cutoffs = cutoffs[0]  #
    # TODO: ----------------------------------------------------------------------- #

    if type(data) is pd.DataFrame:
        df = data
    elif type(data) is dict:
        df = data['dataset']
    else:
        raise ValueError(
            "Argument data must be a pandas DataFrame or a dictionary "
            "with a 'dataset' key."
            )

    # Copy the data to avoid modifying the original DataFrame
    if copy:
        df_copy = df.copy()
    else:
        df_copy = df

    # Convert the values in the specified column to binary based on the cutoff
    df_copy[column_label] = df_copy[column_label].apply(lambda x: 1 if x < cutoffs else 0)

    if type(data) is dict:
        data['dataset'] = df_copy
        return data
    else:
        Warning(f"The data in {make_binary.__name__} is not a dictionary. Returning a tuple.")
        return df_copy


dataset = make_binary(dataset, column_label='FT5', cutoffs=[22], copy=False)
total_rows = len(dataset)
nan_count = dataset['Age'].isna().sum()
percentage_nan = (nan_count / total_rows) * 100

print(f"The percentage of NaN values in the 'Age' column is {percentage_nan:.2f}%")




x = 1

def make_continuous(data, column_label, copy=True):
    if type(data) is pd.DataFrame:
        df = data
    elif type(data) is dict:
        df = data['dataset']
    else:
        raise ValueError(
            "Argument data must be a pandas DataFrame or a dictionary "
            "with a 'dataset' key."
        )

    # Copy the data to avoid modifying the original DataFrame
    if copy:
        df_copy = df.copy()
    else:
        df_copy = df

    # Convert the values in the specified column float
    df_copy[column_label] = df_copy[column_label].astype(float)

    if type(data) is dict:
        data['dataset'] = df_copy
        return data
    else:
        Warning(f"The data in {make_binary.__name__} is not a dictionary. Returning a tuple.")
        return df_copy


# %% Data Imputation

if data_imputation:
    from sklearn.impute import SimpleImputer
    
    imputer = SimpleImputer(
        missing_values=np.nan,
        strategy=strategy,
        copy=copy,
        add_indicator=add_indicator,  # interesting for later, TODO: explore
        keep_empty_features=False,  # no effect: we have removed empty features in cleanup alrdy
    )
    
    data_dict = {
        "type": "SimpleImputer",
        "imputer": imputer,
        "strategy": strategy,
        "add_indicator": add_indicator,
    }
    
    # Isolate relevant data
    d_protein_intensities = dataset.iloc[:, start_column:]
    df_imputed = dataset.copy()
    df_imputed.iloc[:, start_column:] = pd.DataFrame(
        data_dict['imputer'].fit_transform(d_protein_intensities),
        columns=d_protein_intensities.columns
    )
    data_dict['dataset'] = df_imputed
    data_dict['date'] = pd.Timestamp.now()
    
else:
    df_imputed = dataset.copy().dropna(axis=1)
    data_dict = {'type': 'NAN_ELIMINATION', 'dataset': df_imputed, 'date': pd.Timestamp.now()}
    
# Add imputed dataset and date to dictionary
data_dict['date'] = pd.Timestamp.now()

logger.info("/__ DATA IMPUTATION ___________________________________________")
logger.info(
    f"Imputed protein intensity columns free from NaN values: "
    f"{not bool(df_imputed.iloc[:, start_column:].isna().any(axis=None))}"
)
logger.info("\n")


# %% Summary Statistics

if verbose:
    logger.info("/__ SUMMARY STATISTICS (POST-IMPUTATION) ______________________")
    logger.info(f"Number of features (X): {df_imputed.shape[1]}")
    logger.info(f"Number of entries (N): {df_imputed.shape[0]}")
    logger.info(f"Northstar Score (y) mean: {df_imputed['FT5'].mean()}")
    logger.info(f"Northstar Score (y) median: "
                f"{df_imputed['FT5'].median()}")
    logger.info(f"Northstar Score (y) variance: "
                f"{df_imputed['FT5'].var()}")
    logger.info(f"Northstar Score (y) std devition: "
                f"{df_imputed['FT5'].std()}")
    logger.info(f"Northstar Score (y) max: "
                f"{df_imputed['FT5'].max()}")
    logger.info(f"Northstar Score (y) min: "
                f"{df_imputed['FT5'].min()}")
    
    logger.info(f"Protein intensities (X) global mean: "
                f"{df_imputed.iloc[:, start_column:].mean().mean()}")
    logger.info(f"Protein intensities (X) global median: "
                f"{df_imputed.iloc[:, start_column:].median().mean()}")
    logger.info(f"Protein intensities (X) global variance: "
                f"{df_imputed.iloc[:, start_column:].var().mean()}")
    logger.info(f"Protein intensities (X) std deviation: "
                f"{df_imputed.iloc[:, start_column:].std().mean()}")
    logger.info(f"Protein intensities (X) max: "
                f"{df_imputed.iloc[:, start_column:].max().max()}")
    logger.info(f"Protein intensities (X) min: "
                f"{df_imputed.iloc[:, start_column:].min().min()}")
if verbose > 1:
    logger.info(f"Mean values: {df_imputed.mean()}")
    logger.info(f"Median values: {df_imputed.median()}")
if verbose:
    logger.info("\n")


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

if verbose:
    logger.info("/__ NORMALIZATION _____________________________________________")
    logger.info(f"Mean before normalization: {scaler.mean_.mean()}")
    logger.info("Variance before normalization: {scaler.var_.mean()}")

df_normalized.iloc[:, start_column:] = scaler.transform(d_protein_intensities)
if verbose > 1:
    logger.info("Scaled data:", df_normalized.iloc[:, start_column:])

d_protein_intensities_normalized = df_normalized.iloc[:, start_column:]
if verbose:
    logger.info(f"Variance after (should be ~1):"
                f"{d_protein_intensities_normalized.var(axis=0).mean()}")
    logger.info(f"Mean after: (should be ~1):"
                f"{d_protein_intensities_normalized.mean(axis=0).mean()}")
    logger.info(f"Normalized protein intensity columns free from NaN values:"
                f"{not bool(d_protein_intensities_normalized.isna().any(axis=None))}")
    logger.info("\n")


# # Normalize the specified columns
# TODO: try StandarScaler instead
# TODO: probably some bug here, rewrite [0,1]-interval normalization with a pandas Scaler instead
# index = dataset.shape[1] - len(columns_to_normalize)
# for col in columns_to_normalize:
#     col_range = max_vals.iloc[col - index] - min_vals.iloc[col - index]
#     # Normalize the current column using scaling formula
#     df_normalized.iloc[:, col] = (dataset.iloc[:, col] - min_vals.iloc[col - index]) / col_range
    
data_dict['dataset'] = df_normalized


# %% Categorization of Northstar score (y)

"""
Temp test for bin classes
[0,34], break at 17
"""

from utils import make_binary

# TODO: Fix bug in pipeline df_normalized['FT5'] = ... should be df_normalized = ...
df_normalized = make_binary(df_normalized, column_label='FT5', cutoffs=[17], copy=False)


# %% Train-test Split & Feature selection

from sklearn.model_selection import train_test_split

y_data = df_normalized['FT5']  # Vector for the target variable
X_data = df_normalized.iloc[:, start_column:]  # Matrix with variable input

# Splitting the dataset into training and testing sets (80% - 20%)
X_training, X_testing, y_training, y_testing = train_test_split(
    X_data, y_data,
    test_size=test_proportion,
    random_state=seed
)

data_dict['X_training'] = X_training
data_dict['X_testing'] = X_testing
data_dict['y_training'] = y_training
data_dict['y_testing'] = y_testing

if verbose:
    logger.info("|--- TRAIN-TEST SPLIT ---|")
    logger.info(f"X_training free from NaN values:{not bool(X_training.isna().any(axis=None))}")
    logger.info(f"X_testing free from NaN values:{not bool(X_testing.isna().any(axis=None))}")
    logger.info(f"y_training free from NaN values:{not bool(y_training.isna().any(axis=None))}")
    logger.info(f"y_testing free from NaN values:{not bool(y_testing.isna().any(axis=None))}")
    logger.info("\n")


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

# %% Naïve Bayes

if verbose:
    logger.info("/__ MODEL TRAINING & FITTING (BAYES)  _________________________")
    logger.info("\n")

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

if verbose > 1:
    print("Start naive test...")
    
gnb = GaussianNB()
gnb.fit(X_training, y_training)
y_prediction = gnb.predict(X_testing)

res = sum((a != b) for (a, b) in zip(y_testing, y_prediction))
accuracy_bayes = accuracy_score(y_testing, y_prediction)

if verbose:
    logger.info(f"Number of mislabeled out of a total {X_testing.shape[0]} points: {res}")
    logger.info(f"Accuracy: {accuracy_bayes}")
    logger.info("\n")


# %% SVM Classifier Model

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Configuration
# for poly specification for degree will be needed
kernel_type = 'poly'  # 'linear', 'poly', 'sigmoid', 'precomputed'

# If no imputation has been done, return the dataset_dict as is
if data_dict['type'] == 'NO_IMPUTATION':
    # TODO: Look into if SVC can handle NaN values.
    exit()

X_training = data_dict['X_training']
X_testing = data_dict['X_testing']
y_training = data_dict['y_training']
y_testing = data_dict['y_testing']

if verbose:
    logger.info("#--- MODEL TRAINING & FITTING (SVC) ---#")
    logger.info("\n")
    
# Create the SVM model
models = []
i = 0
for c in [0.1, 1.0, 10.0]:
    svm_model = SVC(
        C=c,  # default, Regularization parameter
        kernel=kernel_type,
        degree=3,
        gamma='scale',
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=0.001,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape='ovr',
        break_ties=False,
        random_state=seed,
    )
    
    # C & kernel
    if verbose:
        logger.info("/__ SVM CLASSIFICATION ________________________________________")
        logger.info(f"/__ SVM MODEL PARAMETERS {i} ___________________________________")
        
        logger.info(
            f"SVC(C={c}, kernel={kernel_type}, degree=2, gamma='scale', coef0=0.0, "
            f"shrinking=True, probability=False, tol=0.001, cache_size=200, "
            f"class_weight=None, verbose=False, max_iter=-1,"
            f"decision_function_shape='ovr', break_ties=False, random_state={seed})"
        )
        
        logger.info(f"c: {c}")
        logger.info(f"kernel: {kernel_type}")
        logger.info(f"degree: 3")
        logger.info(f"gamma: 'scale'")
        logger.info(f"coef0: 0.0")
        logger.info(f"shrinking: True")
        logger.info(f"probability: False")
        logger.info(f"tol: 0.001")
        logger.info(f"cache_size: 200")
        logger.info(f"class_weight: None")
        logger.info(f"verbose: False")
        logger.info(f"max_iter: -1")
        logger.info(f"decision_function_shape: 'ovr'")
        logger.info(f"break_ties: False")
        logger.info(f"random_state: {seed}")

    
    # Train SVM model
    svm_model.fit(X_training, y_training)
    if verbose > 1:
        print("Finished with model training!")
    
    # Validate SVM model
    y_prediction = svm_model.predict(X_testing)
    if verbose:
        logger.info("Prediction finished!")
    
    if verbose:
        logger.info(f"SVM model: {svm_model}")
    
    accuracy = accuracy_score(y_testing, y_prediction)
    if verbose:
        logger.info(f"Model accuracy: {accuracy}")
    
    models.append({'model': svm_model, 'accuracy': accuracy})
    i += 1
    
    if verbose:
        logger.info("") # formatting
    
# Append SVM model to list
data_dict['svm'] = models

if verbose:
    logger.info("Finished with model training and validation!")

# %%

# breakpoint()
