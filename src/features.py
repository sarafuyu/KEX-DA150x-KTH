#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
## Feature Selection Techniques

:Date: 2024-05-01
:Authors: Sara Rydell, Noah Hopkins

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Imports

# Standard library imports
from pathlib import Path

# External imports
import pandas as pd
import scipy
from sklearn.feature_selection import f_classif

# Local imports
import utils


# %% Setup

VERBOSE = utils.VERBOSITY_LEVEL  # get verbosity level
SEED = utils.RANDOM_SEED         # get random seed
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# %% Data Splitting

def split_data(data, test_size=0.2, random_state=42, start_col=11, y_col_label='FT5'):
    """
    Split the data into input and target variables.

    :param data: A pandas DataFrame or a dictionary with a 'data' key.
    :param test_size: The proportion of the dataset to include in the test split.
    :param random_state: The seed used by the random number generator.
    :param start_col: Column index to start from. Will split the data [cols:] into input and target variables.
    :param y_col_label: The label of the target variable column.
    :return: A tuple of input and target variables.
    """
    from sklearn.model_selection import train_test_split

    if type(data) is pd.DataFrame:
        df = data
    elif type(data) is dict:
        df = data['dataset']
    else:
        raise ValueError("Argument data must be a pandas DataFrame or a dictionary "
                         "with a 'dataset' key.")
    
    y_data = df[y_col_label]         # Vector for the target variable
    X_data = df.iloc[:, start_col:]  # Matrix with variable input

    # Split the dataset into training and testing sets (default 80% - 20%)
    X_training, X_testing, y_training, y_testing = train_test_split(
        X_data, y_data,
        test_size=test_size,
        random_state=random_state
    )
    
    if type(data) is dict:
        data['X_training'] = X_training
        data['X_testing'] = X_testing
        data['y_training'] = y_training
        data['y_testing'] = y_testing
        return data
    else:
        Warning(f"The data in {split_data.__name__} is not a dictionary. Returning a tuple.")
        return X_training, X_testing, y_training, y_testing


# %% 1. Filter Methods for Feature Selection
## 1.1 Univariate Feature Selection
#
# Univariate feature selection works by selecting the best features based on univariate statistical tests.
# It can be seen as a preprocessing step to an estimator. There are three options.
#
#  - SelectKBest removes all but the highest scoring features.
#
#  - SelectPercentile removes all but a user-specified highest scoring percentage of features using
#    common univariate statistical tests for each feature: false positive rate SelectFpr,
#    false discovery rate SelectFdr, or family wise error SelectFwe.
#
#  - GenericUnivariateSelect allows to perform univariate feature selection with a configurable
#    strategy. This allows to select the best univariate selection strategy with hyper-parameter
#    search estimator.

def select_KBest(data_dict, score_func=f_classif, k=100):
    """
    Does KBest feature selection on given test data.
    
    This method selects the best features based on univariate statistical tests. It can be seen as a preprocessing step
    to an estimator.
    
    :param data_dict: A dictionary containing the dataset, its type and other metadata.
    :param score_func: Function taking two arrays X and y, and returning a pair of arrays (scores, p-values).
    :param k: Number of features to select.
    :return: dataset_dict
    """
    from sklearn.feature_selection import SelectKBest
    
    if data_dict['type'] == 'NO_IMPUTATION':
        return data_dict
    X_train = data_dict['X_training']
    X_test = data_dict['X_testing']
    y_train = data_dict['y_training']
    
    # Configure the SelectKBest selector (default: f_classif ANOVA F-test)
    k_best_selector = SelectKBest(score_func=score_func, k=k)
    
    # Apply score function to data and store results in k_best_selector SelectKBest class instance
    k_best_selector.fit(X_train, y_train)
    
    # Use evaluated scores to select the best k features
    # BUG: HERE WE HAD THE BUG WHERE THE transform METHOD DIDN'T RETURN
    #      A NEW DATAFRAME, BUT INSTEAD RETURNED A NUMPY ARRAY WITHOUT LABELS
    #
    #   x_train_selected_features = k_best_selector.transform(x_train)
    #   x_test_selected_features = k_best_selector.transform(x_test)
    #
    # FIX: SELECT THE COLUMNS FROM ORIGINAL DF BASED ON THE INDICES OF
    #      THE SELECTED COLUMNS INSTEAD OF USING THE TRANSFORM METHOD:
    cols_idxs = k_best_selector.get_support(indices=True)
    X_train_selected_features = X_train.iloc[:, cols_idxs]
    X_test_selected_features = X_test.iloc[:, cols_idxs]
    
    # Assert that the number of selected features is equal to k
    if X_train_selected_features.shape[1] != k:
        raise ValueError(f"Selected Features Shape {X_train_selected_features.shape[1]} "
                         f"is not equal to k ({k})!")
    
    # Update the dataset dictionary with the selected features
    data_dict['feature_scores'] = k_best_selector
    del data_dict['X_training']
    del data_dict['X_testing']
    data_dict['X_training'] = X_train_selected_features
    data_dict['X_testing'] = X_test_selected_features
    
    return data_dict


# %% Main
def main():
    
    # %% Load Data
    
    df_imputed = pd.read_csv(PROJECT_ROOT/'out'/'imputed_data.csv')
    if VERBOSE:
        print("Data loaded successfully.")
    if VERBOSE > 1:
        df_imputed.head()
    global SEED
    
    
    # %% Split Data
    
    # Splitting the dataset into training and testing sets (80% - 20%)
    X_train, X_test, y_train, y_test = split_data(df_imputed, test_size=0.2, random_state=SEED)
    
    
    # %% 2. Feature Selection Using Model
    
    '''You can use a model to determine the importance of each feature and select the most important features accordingly. Here, we'll use ExtraTreesClassifier as an example for classification. For regression tasks, you could use ExtraTreesRegressor.'''
    
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    
    model = ExtraTreesClassifier(n_estimators=50)
    model = model.fit(X_train, y_train)
    
    # Model-based feature selection
    model_select = SelectFromModel(model, prefit=True)
    X_train_model = model_select.transform(X_train)
    X_test_model = model_select.transform(X_test)  # noqa
    
    if VERBOSE:
        print("Model Selected Features Shape:", X_train_model.shape)
    
    
    # %% 3. Recursive Feature Elimination (RFE)
    
    '''RFE works by recursively removing the least important feature and building a model on those features that remain.'''
    
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    
    # Initialize the model to be used
    model = LogisticRegression(max_iter=1000)
    
    # Initialize RFE and select the top 100 features
    rfe = RFE(estimator=model, n_features_to_select=100, step=1)
    X_train_rfe = rfe.fit_transform(X_train, y_train)
    X_test_rfe = rfe.transform(X_test)  # noqa
    
    if VERBOSE:
        print("RFE Selected Features Shape:", X_train_rfe.shape)
    
    
    # %% Feature Selection using SelectKBest
    
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Select the top 100 features based on ANOVA F-value
    select_k_best = SelectKBest(f_classif, k=100)
    X_train_selected = select_k_best.fit_transform(X_train, y_train)
    X_test_selected = select_k_best.transform(X_test)  # noqa
    
    if VERBOSE:
        print("Selected Features Shape:", X_train_selected.shape)
    
    
    # %% Feature Selection using ExtraTreesClassifier
    
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    
    model = ExtraTreesClassifier(n_estimators=50)
    model = model.fit(X_train, y_train)
    
    # Model-based feature selection
    model_select = SelectFromModel(model, prefit=True)
    X_train_model = model_select.transform(X_train)
    X_test_model = model_select.transform(X_test)  # noqa
    
    if VERBOSE:
        print("Model Selected Features Shape:", X_train_model.shape)
    
    
    # %% Feature Selection using Recursive Feature Elimination (RFE)

    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    
    # Initialize the model to be used
    model = LogisticRegression(max_iter=1000)
    
    # Initialize RFE and select the top 100 features
    rfe = RFE(estimator=model, n_features_to_select=100, step=1)
    X_train_rfe = rfe.fit_transform(X_train, y_train)
    X_test_rfe = rfe.transform(X_test)  # noqa
    
    if VERBOSE:
        print("RFE Selected Features Shape:", X_train_rfe.shape)


# %%
if __name__ == '__main__':
    main()
    