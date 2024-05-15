#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
## Feature Selection Techniques

:Date: 2024-05-01
:Authors: Sara Rydell, Noah Hopkins

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
from copy import deepcopy
from datetime import datetime
# %% Imports

# Standard library imports
from pathlib import Path

import joblib
import numpy as np
# External imports
import pandas as pd
import scipy
from sklearn import svm
from sklearn.feature_selection import f_classif, f_regression, SelectFromModel

# Local imports
import utils


# %% Setup

VERBOSE = utils.VERBOSITY_LEVEL  # get verbosity level
SEED = utils.RANDOM_SEED         # get random seed
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# %% Data Splitting

def split_data(data, X=None, y=None, test_size=0.2, random_state=42, start_col=11, y_col_label='FT5'):
    """
    Split the data into input and target variables.

    :param X: A pandas DataFrame containing the input variables.
    :param y: A pandas DataFrame containing the target variable.
    :param data: A pandas DataFrame or a dictionary with a 'data' key. If X and y are not provided,
        the full dataset under the 'dataset' key will be split into input and target variables.
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

    if X is not None and y is not None:
        X_data = X
        y_data = y
    else:
        X_data = df.iloc[:, start_col:]  # Matrix with variable input
        y_data = df[y_col_label]         # Vector for the target variable


    if test_size:
        # Split the dataset into training and testing sets (default 80% - 20%)
        X_training, X_testing, y_training, y_testing = train_test_split(
            X_data, y_data,
            test_size=test_size,
            random_state=random_state
        )
    else:
        # Note: If test_size is 0, the data will not be split into training and testing sets
        #       For compatibility with the rest of the code, we set the testing data to
        #       be the same as the training data:
        X_training, X_testing, y_training, y_testing = X_data, X_data, y_data, y_data

    if type(y_testing) is pd.Series:
        y_testing = y_testing.to_frame()
    if type(y_training) is pd.Series:
        y_training = y_training.to_frame()
    
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


# %% Feature Selection using Recursive Feature Elimination (RFE)
def select_RFE(data_dict, score_func, k):
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression

    # Initialize the model to be used
    svc = svm.SVC(
        kernel='poly', degree=3, gamma='scale', coef0=0.0, tol=1e-3, C=1.0, shrinking=True, cache_size=200,
        verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=SEED,
    )

    # Initialize RFE and select the top 100 features
    rfe = RFE(estimator=svc, n_features_to_select=k, step=1, verbose=0, importance_getter='auto')
    X_train_rfe = rfe.fit_transform(data_dict['X_training'], data_dict['y_training'])

    if VERBOSE:
        print("RFE Selected Features Shape:", X_train_rfe.shape)

    # Use evaluated scores to select the best k features
    cols_idxs = rfe.get_support(indices=True)
    X_train_selected_features = data_dict['X_training'].iloc[:, cols_idxs]
    X_test_selected_features = data_dict['X_testing'].iloc[:, cols_idxs]

    # Assert that the number of selected features is equal to k
    if X_train_selected_features.shape[1] != k:
        raise ValueError(f"Selected Features Shape {X_train_selected_features.shape[1]} " # alt. attribute n_features_
                         f"is not equal to k ({k})!")

    # Update the dataset dictionary with the selected features
    data_dict['feature_scores'] = rfe
    del data_dict['X_training']
    del data_dict['X_testing']
    data_dict['X_training'] = X_train_selected_features
    data_dict['X_testing'] = X_test_selected_features

    return data_dict


def select_mutual_info_regression(data_dict, k):
    # mutual_info_regression(X, y, *, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)

    # Estimate mutual information for a continuous target variable.
    # Mutual information (MI) [1] between two random variables is a non-negative value, which measures the dependency between the variables.
    # It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.

    from sklearn.feature_selection import mutual_info_regression

    if data_dict['type'] == 'NO_IMPUTATION':
        return data_dict
    X_train = data_dict['X_training']
    X_test = data_dict['X_testing']
    y_train = data_dict['y_training']

    # Calculate the mutual information between each feature and the target variable
    mi_list = []
    for i in range(X_train.shape[1]):
        # discrete_features{‘auto’, bool, array-like}, default='auto'
        # If bool, then determines whether to consider all features discrete or continuous. If array, then it should be either a boolean mask with shape (n_features,) or array with indices of discrete features. If ‘auto’, it is assigned to False for dense X and to True for sparse X.
        mi = mutual_info_regression(X_train, y_train, discrete_features=False, n_neighbors=3, copy=True, random_state=SEED)
        print(f"Feature {i} has MI: {mi}")
        mi_list.append(mi)

    # Sort the features based on mutual information
    mi_sorted = sorted(range(len(mi_list)), key=lambda i: mi_list[i], reverse=True)

    # Select the top k features
    X_train_selected_features = X_train.iloc[:, mi_sorted[:k]]
    X_test_selected_features = X_test.iloc[:, mi_sorted[:k]]

    # Update the dataset dictionary with the selected features
    del data_dict['X_training']
    del data_dict['X_testing']
    data_dict['X_training'] = X_train_selected_features
    data_dict['X_testing'] = X_test_selected_features

    return data_dict


def select_XGB(data_dict, log=print, n_estimators=-1, verbosity=0, use_label_encoder=False, validate_parameters=True,
               missing=np.nan, objective='binary:logistic', n_jobs_xgb=6, n_jobs_rfecv=2, scoring='accuracy', cv=5,
               min_features_to_select=1, step=1,
               original_dataset=None, original_protein_start_col=11,
               config=None, start_time=None, logfile=None):
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    from sklearn.metrics import f1_score, confusion_matrix
    from sklearn.metrics import accuracy_score
    import seaborn as sns
    from sklearn.feature_selection import RFECV

    XGB_start_time = datetime.now()

    X_train = data_dict['X_training']
    X_test = data_dict['X_testing']
    y_train = data_dict['y_training']
    y_test = data_dict['y_testing']

    # Set n_estimators to size of training set
    if n_estimators == -1:
        n_estimators = X_train.shape[1]

    # Setup XGB feature selection
    xgbclf = xgb.XGBClassifier(
        random_state=SEED,
        n_jobs=n_jobs_xgb,
        n_estimators=n_estimators,
        verbosity=verbosity,
        use_label_encoder=use_label_encoder,
        validate_parameters=validate_parameters,
        missing=missing,
        objective=objective,
    )
    xgbclf.fit(X_train, y_train)
    rfecv = RFECV(estimator=xgbclf, step=step, min_features_to_select=min_features_to_select, cv=cv, scoring=scoring, verbose=verbosity, n_jobs=n_jobs_rfecv, importance_getter='auto')
    min_features_to_select = rfecv.min_features_to_select
    xgbclf = rfecv.estimator

    if VERBOSE:
        log('/__ FEATURE SELECTION _________________________________________')
        log(f"Mean std deviation X: ---------------- {X_imputed.std().mean()}\n")
        log('Feature Selection Method: ------------- XGB-RFECV')
        log(f'Objective function: ------------------ {objective}')
        log(f'Minimum features to select: ---------- {min_features_to_select}')
        log(f'Step size: --------------------------- {step}')
        log(f'Num estimators/trees: ---------------- {(str(n_estimators) + ' (1 per feature)') if n_estimators == X_train.shape[1] else n_estimators}')
        log(f'Num CV folds: {rfecv.cv}')
        log(f'Starting XGB-RFE-CV Feature Selection ...')

    # Start the XGB feature selection
    rfecv.fit(X_train, y_train)
    xgbclf = rfecv.estimator_

    # Log time taken to run
    XGB_end_time = datetime.now()
    timedelta = str(XGB_end_time - XGB_start_time).split('.')
    hms = timedelta[0].split(':')
    if VERBOSE:
        log(f"Finished: ---------------------------- {XGB_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"Run time: ---------------------------- {hms[0]}h:{hms[1]}m:{hms[2]}s {timedelta[1][:3]}.{timedelta[1][3:]}ms to run.")

    # Log results
    opt_features = rfecv.n_features_
    best_features = X_train.columns[rfecv.support_]
    accuracy = accuracy_score(y_test, rfecv.predict(X_test))
    if VERBOSE:
        log(f'Classes: ----------------------------- {rfecv.classes_}')
        log(f'Accuracy score: ---------------------- {accuracy}')
        log(f'Num optimal features found: ---------- {opt_features}')
    if VERBOSE > 2:
        log(f'Optimal features found: -------------- {best_features}')
    if VERBOSE:
        utils.print_summary_statistics(data_dict, log=log, current_step='(POST-FEATURE SELECTION) ')

    # Save best_features to file
    try:
        X_train[best_features].to_csv(PROJECT_ROOT/'out'/(utils.get_file_name(data_dict)+'_FeatureSelect꞉XGB-RFE-CV꞉X_train_best_features.csv'), index=False)
        log(f'Optimal features found saved to: --------------------------- {PROJECT_ROOT/'out'/(utils.get_file_name(data_dict)+'_FeatureSelect꞉XGB-RFE-CV꞉X_train_best_features.csv')}')
    except Exception as e:
        # Handle error instead of crashing
        log(f'Error: {e}')

    # Save results to file
    try:
        data_dict['date'] = start_time
        # Attempt to save CV results
        if hasattr(rfecv, 'cv_results_'):
            cv_results = rfecv.cv_results_
            cv_results_df = pd.DataFrame(cv_results)
            if VERBOSE > 1:
                log(f'CV Results: \n{cv_results_df}')
            cv_results_df.to_csv(PROJECT_ROOT/'out'/(utils.get_file_name(data_dict)+'_FeatureSelect꞉XGB-RFE-CV꞉cv_results_.csv'), index=False)
            # Try generating plot of optimal number of features
            # Maybe even better looking plot: https://www.scikit-yb.org/en/latest/api/model_selection/rfecv.html
            import matplotlib.pyplot as plt
            n_scores = len(rfecv.cv_results_["mean_test_score"])
            plt.figure()
            plt.title("Optimal Number of Features")
            plt.xlabel("Number of features selected")
            plt.ylabel("Mean test accuracy")
            plt.errorbar(
                range(min_features_to_select, n_scores + min_features_to_select),
                rfecv.cv_results_["mean_test_score"],
                yerr=rfecv.cv_results_["std_test_score"],
            )
            plt.savefig(PROJECT_ROOT/'out'/(utils.get_file_name(data_dict)+'_FeatureSelect꞉XGB-RFE-CV꞉cv_results_.png'))
            log(f'Plot of feature selection CV results saved to: ------------- {PROJECT_ROOT/"out"/(utils.get_file_name(data_dict)+"_FeatureSelect:XGB-RFE-CV:cv_results_.png")}')
    except Exception as e:
        # Handle error instead of crashing
        log(f'Error: {e}')
    # Save the classifier and RFECV object to file
    try:
        joblib.dump(deepcopy(xgbclf), PROJECT_ROOT / 'out' / (config['START_TIME'].strftime('%Y-%m-%d %H:%M:%S') + '__FeatureSelect__XGB.pkl'))
        log(f"XGB feature selection classifier object saved to file at: -- {PROJECT_ROOT/'out'/(config['START_TIME'] + '__FeatureSelect__XGB.pkl')}")

        joblib.dump(deepcopy(rfecv), PROJECT_ROOT / 'out' / (config['START_TIME'].strftime('%Y-%m-%d %H:%M:%S') + '__FeatureSelect__RFECV.pkl'))
        log(f"XGB feature selection RFECV object saved to file at: ------- {PROJECT_ROOT/'out'/(config['START_TIME'].strftime('%Y-%m-%d %H:%M:%S') + '__FeatureSelect__RFECV.pkl')}\n")
    except Exception as e:
        log(f'Error: {e}')

    # Update the dataset dictionary with the selected features
    data_dict['feature_selection_rfecv'] = rfecv
    data_dict['feature_selection_xgb'] = xgbclf
    del data_dict['X_training']
    del data_dict['X_testing']
    data_dict['X_training'] = X_train[best_features]
    data_dict['X_testing'] = X_test[best_features]

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


# %%
if __name__ == '__main__':
    main()


