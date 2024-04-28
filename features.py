"""
## Feature Selection Techniques

Authors:
Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Imports

# External imports
import pandas as pd

# Local imports
import utils
verbose = utils.verbosity_level  # get verbosity level
seed = utils.random_seed  # get random seed


# %% Data Splitting

def split_data(data, test_size=0.2, random_state=42, col=11):
    """
    Split the data into input and target variables.
 
    :param data: A pandas DataFrame or a dictionary with a 'data' key.
    :param test_size: The proportion of the dataset to include in the test split.
    :param random_state: The seed used by the random number generator.
    :param col: Column index to start from. Will split the data [cols:] into
                 input and target variables.
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
    
    y_data = df['FT5']  # Vector for the target variable
    X_data = df.iloc[:, col:]  # Matrix with variable input

    # Splitting the dataset into training and testing sets (80% - 20%)
    X_training, X_testing, y_training, y_testing = train_test_split(
        X_data, y_data, test_size=test_size, random_state=random_state
    )
    
    if type(data) is dict:
        data['X_training'] = X_training
        data['X_testing'] = X_testing
        data['y_training'] = y_training
        data['y_testing'] = y_testing
        return data
    else:
        return X_training, X_testing, y_training, y_testing


# %% 1. Filter Methods for Feature Selection
## 1.1 Univariate Feature Selection
"""
Univariate feature selection works by selecting the best features based on univariate statistical tests.
It can be seen as a preprocessing step to an estimator. There are three options.

 - SelectKBest removes all but the highest scoring features.

 - SelectPercentile removes all but a user-specified highest scoring percentage of features using
   common univariate statistical tests for each feature: false positive rate SelectFpr,
   false discovery rate SelectFdr, or family wise error SelectFwe.

 - GenericUnivariateSelect allows to perform univariate feature selection with a configurable
   strategy. This allows to select the best univariate selection strategy with hyper-parameter
   search estimator.
"""
def select_KBest(dataset_dict, k=100): # northstar_cutoff=0.5
    """
    Does KBest feature selection on given test data.
    
    This method selects the best features based on univariate statistical tests. It can be seen as a preprocessing step to an estimator.
    
    :param dataset_dict: A dictionary containing the dataset, its type and other metadata.
    :param k: Number of features to select.
    :param northstar_cutoff: The y-variable (the Northstar score) will be made binary. A Northstar
        score below the cutoff is converted into 1, a score above the cutoff is converted into 0.
    :return: dataset_dict
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    
    if dataset_dict['type'] == 'no_imputation':
        return dataset_dict
    x_train = dataset_dict['X_training']
    x_test = dataset_dict['X_testing']
    y_train = dataset_dict['y_training']
     
    # Make the Northstar score binary
    """
    if northstar_cutoff:
        from utils import make_binary
        y_train = make_binary(y_train, column='FT5', cutoff=northstar_cutoff)
    """
    # Configure the SelectKBest selector (default: f_classif ANOVA F-test)
    k_best_selector = SelectKBest(score_func=f_classif, k=k)
    
    # Apply score function to data and store results in k_best_selector SelectKBest class instance
    k_best_selector.fit(x_train, y_train)
    
    # Use evaluated scores to select the best k features
    x_train_selected_features = k_best_selector.transform(x_train)
    x_test_selected_features = k_best_selector.transform(x_test)
    
    # Assert that the number of selected features is equal to k
    if x_train_selected_features.shape[1] != k:
        raise ValueError(f"Selected Features Shape {x_train_selected_features.shape[1]} "
                         f"is not equal to k ({k})!")
    
    # Update the dataset dictionary with the selected features
    dataset_dict['feature_scores'] = k_best_selector
    del dataset_dict['X_training']
    del dataset_dict['X_testing']
    dataset_dict['X_training'] = x_train_selected_features
    dataset_dict['X_testing'] = x_test_selected_features
    
    return dataset_dict


# %% Main
def main():
    
    # %% Load Data
    
    df_imputed = pd.read_csv('imputed_data.csv')
    if verbose:
        print("Data loaded successfully.")
    if verbose > 1:
        df_imputed.head()
    global seed
    
    
    # %% Split Data
    
    # Splitting the dataset into training and testing sets (80% - 20%)
    X_train, X_test, y_train, y_test = split_data(df_imputed, test_size=0.2, random_state=seed)
    
    
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
    
    if verbose:
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
    
    if verbose:
        print("RFE Selected Features Shape:", X_train_rfe.shape)
    
    
    # %% Feature Selection using SelectKBest
    
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Select the top 100 features based on ANOVA F-value
    select_k_best = SelectKBest(f_classif, k=100)
    X_train_selected = select_k_best.fit_transform(X_train, y_train)
    X_test_selected = select_k_best.transform(X_test)  # noqa
    
    if verbose:
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
    
    if verbose:
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
    
    if verbose:
        print("RFE Selected Features Shape:", X_train_rfe.shape)


# %%
if __name__ == '__main__':
    main()
    