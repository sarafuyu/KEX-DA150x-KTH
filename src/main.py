"""
This is the main file for the project. It will be used to run the project.

:Date: 2024-05-01
:Authors: Sara Rydell, Noah Hopkins

Currently supported: Binary classification using SVM classifier.

TODOs
=====

TODO: Add final accuracy to the cv_results csv file

------ MVP for today ------
TODO(priority 1): Add filtering of dataset dicts in each find_best_Xmodel function
                  so that only compatible datasets are used during gridsearch
TODO(priority 1): Add SVR classifier
------ Nice to have for today ------
TODO(priority 2): Implement SVM classifier with NaN
TODO(priority 2): Validate that multiclass SVM still works in the pipeline
------ MVP for tomorrow ------
TODO(priority 3): Add Ridge Regression
TODO(priority 3): Add Lasso Regression
------ Nice to have for tomorrow ------
TODO(priority 4): Add Decision Tree classifier
TODO(priority 4): Add Naive Bayes classifier
TODO(priority 4): Refactor cleaning.py
TODO(priority 5): Refactor main.py pipeline to make it more DRY (construct grid_param in main and 
                  pass to a function that calls the various grid search functions for the various 
                  models depending on the different
                  dataset dict attributes)
--------------------------------

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Imports

## Standard library imports
import sys
import logging
from collections.abc import Sequence, Callable
from datetime import datetime

## External library imports
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import f_classif
from sklearn.linear_model import BayesianRidge


# %% Start Time

start_time = datetime.now()


# %% Configuration

# **********----------------------------------------------------------------------------********** #
# |                                        ~~ General ~~                                         | #
# **********----------------------------------------------------------------------------********** #

# ----------
# Verbosity
# ----------

# The higher, the more verbose. Can be 0, 1, 2, 3, or 4.
verbose: int = 1
# Level 0: No prints.
# Level 1: Only essential prints. Minimal prints for each individual hyperparameter/data set.
# Level 2: Essential prints and additional prints for each hyperparameter config/data set.
# Level 3: All prints. Maybe some previews of data.
# Level 4: All prints and all previews of data.

# --------
# Logging
# --------

# Set to True to log the output to a file.
log: bool = True
# Set the name of the log file.
logfile: str = 'pipline-log-' + start_time.strftime("%Y-%m-%d-%H%M%S") + '.log'

# -----------
# Randomness
# -----------

# Randomness seed for all methods that take a seed in all files of the project
seed: int = 42

# ----------------
# Data Extraction
# ----------------

# Path to dataset
path: str = 'normalised_data_all_w_clinical_kex_20240321.csv'


# **********----------------------------------------------------------------------------********** #
# |                                     ~~ Data imputation ~~                                    | #
# **********----------------------------------------------------------------------------********** #

# Pick imputation modes to use:
simple_imputer: bool = True
iterative_imputer: bool = False
KNN_imputer: bool = False
nan_elimination: bool = True
no_imputation: bool = False
sparse_no_imputation: bool = False  # Note: if `True`, `no_imputation` must be set to `True`.

# -----------------------------
# Simple imputer configuration
# -----------------------------

# Strategy for imputing missing values
strategy_simple_imp: Sequence[str] = ['mean']  # ["mean", "median", "most_frequent"]
# Add indicator for missing values
add_indicator_simple_imp: bool = False

# Should always be True, since the implementation expects a copy of the data
copy_simple_imp: bool = True

# --------------------------------
# Iterative imputer configuration
# --------------------------------

# Estimator, e.g. a BayesianRidge() object or an estimator object from scikit-learn.
# Can probably be customized, but leave default for now.
# For future type hints, see: https://stackoverflow.com/a/60542986/6292000
estimator_iter_imp = BayesianRidge()
# Maximum number of imputation rounds to perform. The imputer will stop iterating after this many iterations.
max_iter_iter_imp: int = 100  # try low number of iterations first, see if converges, then try higher numbers
tol_iter_imp: float = 1e-3  # might need to adjust
initial_strategy_iter_imp: Sequence[str] = ["mean"]  # ["mean", "median", "most_frequent", "constant"]
# Number of other features to use to estimate the missing values of each feature column.
# None means all features, which might be too many.
n_nearest_features_iter_imp: Sequence[int] = [5, 20, 50, None]  # [10, 100, 500, None]
imputation_order_iter_imp: Sequence[str] = ["ascending"]  # ["ascending", "descending" "random"]
# ascending: From the features with the fewest missing values to those with the most
min_value_iter_imp: int = 0  # no features have negative values, adjust tighter for prot intensities?
max_value_iter_imp: str | int = '10% higher than max'

# --------------------------
# KNN imputer configuration
# --------------------------

# Missing values to impute
missing_values_KNN_imp = pd.NA
# Initial span of neighbours considering dataset size
n_neighbours_KNN_imp: Sequence[int] = [5, 20, 50]
# default='uniform', callable has potential for later fitting
weights_KNN_imp: Sequence[str] = ['uniform', 'distance']
metric_KNN_imp: Sequence[str] = ['nan_euclidean']
add_indicator_KNN_imp = False
keep_empty_features_KNN_imp = False

# Should always be True, since the implementation expects a copy of the data
copy_KNN_imp = True

# ------------------------------
# NaN elimination configuration
# ------------------------------

# If True, drop all columns with NaN values. If False, drop rows with NaN values.
drop_cols_nan_elim = True


# **********----------------------------------------------------------------------------********** #
# |                                   ~~ Data normalization ~~                                   | #
# **********----------------------------------------------------------------------------********** #

# First column to normalize. Will normalize all columns from this index and onwards.
first_column_to_normalize: int = 11
# TODO: check if this is the correct index. Should it be 10 instead?
# Note: we only normalize the antibody/protein intensity columns (cols 11 and up).
#       Age, disease, FTs not normalized.


# **********----------------------------------------------------------------------------********** #
# |                        ~~ Re-categorization of Northstar score (y) ~~                        | #
# **********----------------------------------------------------------------------------********** #
# Northstar score (y) is originally a discrete variable on [0,34]
#
# Bin Northstar score into a categorical variable.
# For N cutoffs, the data is divided into N+1 classes.
# For binary classification, use one cutoff, [a].
#
# Example:
# cutoffs=[a] will create a binary variable with classes:
#   df.iloc[:, 0:(a - 1)] == 1
#   df.iloc[:, a:       ] == 0
# i.e. the variable is 1 if (x < a) else 0.
#
cutoffs: Sequence[int] | bool = [17]  # False  # [17]  # break into classes at 17
column_to_categorize: str = 'FT5'


# **********----------------------------------------------------------------------------********** #
# |                                   ~~ Train-test split ~~                                     | #
# **********----------------------------------------------------------------------------********** #

# For detailed configuration for each feature selection mode, see features.py
test_proportion: float = 0.2

# Column index to start from. Will split the selected X and y data, into the two data subsets:
#   X_train, y_train,
#   X_test, y_test,
# where X_train and X_test are the input variables and y_train and y_test are the target variables.
X_start_column_idx: int = 11  # X columns are from this column index and onwards TODO: check if 11 is the correct index.
y_column_label: str = 'FT5'   # y column label


# **********----------------------------------------------------------------------------********** #
# |                                   ~~ Feature selection ~~                                     | #
# **********----------------------------------------------------------------------------********** #

# score_func is a function taking in X, an array of columns (features), and y, a target column,
# and returning a pair of arrays (scores, p-values).
score_func: Callable[[Sequence, Sequence], tuple[Sequence, Sequence]] = f_classif
k_features: int = 60  # 216  # 100 # TODO: add different levels: 30, 60, 90, 120, 150, 200 ...


# **********----------------------------------------------------------------------------********** #
# |                              ~~ Model training & fitting ~~                                  | #
# **********----------------------------------------------------------------------------********** #

verbose_grid_search: int = 0

# ---------------
# SVM Classifier
# ---------------

# Enable SVC
try_SVC = True

# Hyperparameters:            # np.logspace(start, stop, num=50)
C_params_SVC: Sequence[float] = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  # np.linspace(0.00001, 3, num=10)  # np.linspace(0.001, 100, num=60)
kernels_SVC: Sequence[str] = ['poly', 'sigmoid', 'rbf']  # 'linear','rbf', 'precomputed'
degree_params_SVC: Sequence[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
gamma_params_SVC: Sequence[str] = ['auto']  # scale not needed since normalization X_var
coef0_params_SVC: Sequence[float] = [-100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0]  # np.linspace(-2, 4, num=10)  # np.linspace(-10, 10, num=60)
shrinking_SVC: Sequence[bool] = [True]
probability_SVC: Sequence[bool] = [False]
tol_params_SVC: Sequence[float] = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  # np.linspace(0.01, 0.0001, 10)  # np.linspace(0.01, 0.0001, 10)
cache_size_params_SVC: Sequence[int] = [500]
class_weight_SVC: dict | None = None
verb_SVC: int = verbose
max_iter_params_SVC: Sequence[int] = [1_000_000]  # [-1]
decision_function_shape_params_SVC: Sequence[str] = ['ovr'] # onli ovo if multi class
break_ties_params_SVC: Sequence[bool] = [False]

# return_train_score_SVC:
# Include scores in `cv_results_`. Computing training scores is used to get insights on how different parameter settings
# impact the overfitting/underfitting trade-off. However, computing the scores on the training set can be
# computationally expensive and is not strictly required to select the parameters that yield the best generalization
# performance.
return_train_score_SVC = False

# SVR Classifier
# --------------

# Enable SVR
try_SVR = False

kernels_SVR: Sequence[str] = ['rbf']
degree_params_SVR: Sequence[int] = [3]
gamma_params_SVR: Sequence[str] = ['scale']
coef0_params_SVR: Sequence[float] = [0.0]
tol_params_SVR: Sequence[float] = [0.001]
C_params_SVR: Sequence[float] = [1.0, 10.0]
epsilon_params_SVR: Sequence[float] = [0.1]
shrinking_params_SVR: Sequence[bool] = [True]
cache_size_params_SVR: Sequence[int] = [200]
verb_SVR: int = verbose
max_iter_params_SVR: Sequence[int] = [-1]

# return_train_score_SVC:
# Include scores in `cv_results_`. Computing training scores is used to get insights on how different parameter settings
# impact the overfitting/underfitting trade-off. However, computing the scores on the training set can be
# computationally expensive and is not strictly required to select the parameters that yield the best generalization
# performance.
return_train_score_SVR: bool = False


# %% Local imports

# The utils module need to be imported first to set verbosity level and random seed
import utils
utils.verbosity_level = verbose  # Set verbosity level for all modules
utils.random_seed = seed         # Set random seed for all modules

# Import the other modules (they import utils to get verbosity level and random seed)
import cleaning
import imputation
import normalization
import features
import classifier


# %% Logging

# Set up logging

handlers = []
if log:
    file_handler = logging.FileHandler(filename=logfile)  # TODO: generate unique filename
    handlers.append(file_handler)
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers.append(stdout_handler)

logging.basicConfig(level=logging.DEBUG,
                    format=('[%(asctime)s] '
                            '{%(filename)s:%(lineno)d} : '
                            '%(levelname)s | '
                            '%(message)s'),
                    handlers=handlers)

logger = logging.getLogger('LOGGER_NAME')
    
pipeline_config = {
    'seed':                        seed,
    'verbose':                     verbose,
    'path':                        path,
    'simple_imputer':              simple_imputer,
    'iterative_imputer':           iterative_imputer,
    'KNN_imputer':                 KNN_imputer,
    'nan_elimination':             nan_elimination,
    'no_imputation':               no_imputation,
    'sparse_no_imputation':        sparse_no_imputation,
    'add_indicator_simple_imp':    add_indicator_simple_imp,
    'copy_simple_imp':             copy_simple_imp,
    'strategy_simple_imp':         strategy_simple_imp,
    'estimator_iter_imp':          estimator_iter_imp,
    'max_iter_iter_imp':           max_iter_iter_imp,
    'tol_iter_imp':                tol_iter_imp,
    'initial_strategy_iter_imp':   initial_strategy_iter_imp,
    'n_nearest_features_iter_imp': n_nearest_features_iter_imp,
    'imputation_order_iter_imp':   imputation_order_iter_imp,
    'min_value_iter_imp':          min_value_iter_imp,
    'max_value_iter_imp':          max_value_iter_imp,
    'n_neighbours_KNN_imp':        n_neighbours_KNN_imp,
    'weights_KNN_imp':             weights_KNN_imp,
    'metric_KNN_imp':              metric_KNN_imp,
    'drop_cols_nan_elim':          drop_cols_nan_elim,
    'first_column_to_normalize':   first_column_to_normalize,
    'cutoffs':                     cutoffs,
    'column_to_categorize':        column_to_categorize,
    'test_proportion':             test_proportion,
    'X_start_column_idx':          X_start_column_idx,
    'y_column_label':              y_column_label,
    'score_func':                  score_func,
    'k_features':                  k_features,
    'try_SVC':                     try_SVC,
    'try_SVR':                     try_SVR,
}


# %% Load Data

# Load the data
dataset = pd.read_csv(path)

if verbose:
    logger.info("Data loaded successfully.")
if verbose == 2:
    dataset.head()  # Pre-view first five rows


# %% Data Cleaning

# Clean and preprocess the data
dataset = cleaning.clean_data(dataset, logger=logger.info)


# %% Data Imputation

# Create imputers
dataset_dicts = []
if simple_imputer:
    dataset_dicts = dataset_dicts + imputation.create_simple_imputers(
        add_indicator=add_indicator_simple_imp,
        copy=copy_simple_imp,
        strategy=strategy_simple_imp,
    )
if iterative_imputer:
    dataset_dicts = dataset_dicts + imputation.create_iterative_imputers(
        df=dataset,
        estimator=estimator_iter_imp,
        max_iter=max_iter_iter_imp,
        tol=tol_iter_imp,
        initial_strategy=initial_strategy_iter_imp,
        n_nearest_features=n_nearest_features_iter_imp,
        imputation_order=imputation_order_iter_imp,
        min_value=min_value_iter_imp,
        max_value=max_value_iter_imp,
    )
if KNN_imputer:
    dataset_dicts = dataset_dicts + imputation.create_KNN_imputers(
        missing_values=missing_values_KNN_imp,
        n_neighbours=n_neighbours_KNN_imp,
        weights=weights_KNN_imp,
        metric=metric_KNN_imp,
        copy=copy_KNN_imp,
        add_indicator=add_indicator_KNN_imp,
        keep_empty_features=keep_empty_features_KNN_imp,
    )

# Impute data using generated imputers
dataset_dicts = [imputation.impute_data(imputer_dict, dataset, 11)
                 for imputer_dict in dataset_dicts]

# Add NaN-eliminated and un-imputed datasets
if nan_elimination:
    dataset_dicts = dataset_dicts + imputation.eliminate_nan(dataset, drop_cols=drop_cols_nan_elim)
if no_imputation:
    dataset_dicts = dataset_dicts + imputation.no_imputer(dataset, copy=True)


# %% Report Summary Statistics

# Report summary statistics for the imputed datasets
if verbose:
    for data_dict in dataset_dicts:
        utils.print_summary_statistics(data_dict, logger=logger.info, start_column=11)


# %% Data Normalization

# Columns to normalize
# Note: we only normalize the antibody/protein intensity columns (cols 11 or 10 and up)
# age, disease, FTs not normalized
# TODO: instead iterate over the column labels until we reach the first protein intensity column
#       then normalize all columns from there and onwards instead of hardcoding the index.
# TODO: Right now, during

# Normalize the datasets
dataset_dicts = [
    normalization.std_normalization(
        data=data_dict,
        start_column=first_column_to_normalize,
    )
    for data_dict in dataset_dicts
]


# %% Categorization of Northstar score (y)

if cutoffs:
    dataset_dicts = [
        utils.make_binary(
            data_dict,
            column_label=column_to_categorize,
            cutoffs=cutoffs, copy=False
        )
        for data_dict in dataset_dicts
    ]


# %% Train-test Split

# Split data
dataset_dicts = [
    features.split_data(
        data=dataset_dict,
        test_size=test_proportion,
        random_state=seed,
        start_col=X_start_column_idx,
        y_col_label=y_column_label,
    )
    for dataset_dict in dataset_dicts
]


# %% Sparse Data Handling

if sparse_no_imputation:
    dataset_dicts = dataset_dicts + imputation.sparse_no_impute(
        utils.get_dict_from_list_of_dict(dataset_dicts, dict_key='type', dict_value='no_imputation')
    )


# %% Feature selection

# Feature selection
dataset_dicts = [
    features.select_KBest(
        data_dict=data_dict,
        score_func=score_func,
        k=k_features
    )
    for data_dict in dataset_dicts
]

# dataset_dicts is now a list that contains dict with the following:
# 1. Imputed and normalized data sets, date of imputation, type of imputation, imputer objects,
#    summary statistics, ANOVA P-values and feature selection scores (F-values).
# 2. Input and target variables for the feature-selected training and testing data.


# %% Log results

utils.log_results(
    original_dataset=dataset,
    config=pipeline_config,
    original_protein_start_col=first_column_to_normalize,
    dataset_dicts=dataset_dicts,
    logger=logger.info
)


# %% Model Training & Fitting

# Create Naive Bayes models
# dataset_dicts = [classifier.add_naive_bayes_models(dataset_dict) for dataset_dict in dataset_dicts]

# Find best SVM model
if try_SVC:
    dataset_dicts = [
        classifier.find_best_svr_model(
            pipeline_config=pipeline_config,
            dataset_dict=dataset_dict,
            C_params=C_params_SVC,
            kernels=kernels_SVC,
            degree_params=degree_params_SVC,
            gamma_params=gamma_params_SVC,
            coef0_params=coef0_params_SVC,
            shrinking=shrinking_SVC,
            probability=probability_SVC,
            tol_params=tol_params_SVC,
            cache_size_params=cache_size_params_SVC,
            class_weight=class_weight_SVC,
            verb=verb_SVC,
            max_iter_params=max_iter_params_SVC,
            decision_function_shape_params=decision_function_shape_params_SVC,
            break_ties=break_ties_params_SVC,
            random_state=seed,
            verbose_grid_search=verbose_grid_search,
            logger=logger.info,
            return_train_score=return_train_score_SVC,
        )
        for dataset_dict in dataset_dicts
    ]

# Find best SVR model
if try_SVR:
    dataset_dicts = [
        classifier.find_best_svr_model(
            pipeline_config=pipeline_config,
            dataset_dict=dataset_dict,
            kernels=kernels_SVR,
            degree_params=degree_params_SVR,
            gamma_params=gamma_params_SVR,
            coef0_params=coef0_params_SVR,
            tol_params=tol_params_SVR,
            C_params=C_params_SVR,
            epsilon_params=epsilon_params_SVR,
            shrinking_params=shrinking_params_SVR,
            cache_size_params=cache_size_params_SVR,
            verb=verb_SVR,
            max_iter_params=max_iter_params_SVR,
            verbose_grid_search=verbose_grid_search,
            logger=logger.info,
            return_train_score=return_train_score_SVR,
        )
        for dataset_dict in dataset_dicts
    ]

# Find best Ridge Regression model
# TODO

# Find best Lasso Regression model
# TODO

# Find best Decision Tree model
# TODO


# %% Breakpoint

logger.info(f"Pipeline finished {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} which took {datetime.now() - start_time}")

# breakpoint()
