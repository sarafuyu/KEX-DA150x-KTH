"""
Data imputation

Authors:
Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Imports

## External imports
import pandas as pd
import numpy as np

# Local imports
import utils
verbose = utils.verbosity_level  # get verbosity level
seed = utils.random_seed  # get random seed


# %% Option 1: Simple Imputer

def create_simple_imputers():
    """
    Impute missing values (e.g., with simple statistical values of each column) using SimpleImputer.
    
    Required imports:
    from sklearn.impute import SimpleImputer
    import numpy as np
    
    :return: A list of dictionaries, each containing an imputer object and its configuration.
    """
    from sklearn.impute import SimpleImputer
    
    # Configuration
    add_indicator = False  # interesting for later, TODO: explore
    copy = True            # so we can reuse dataframe with other imputers
    strategy = ["mean"]  # ["mean", "median", "most_frequent"]
    
    simple_imputers = []
    for strat in strategy:
        imputer = SimpleImputer(
            missing_values=np.nan,
            strategy=strat,
            copy=copy,
            add_indicator=add_indicator,  # interesting for later, TODO: explore
            keep_empty_features=False,  # no effect: we have removed empty features in cleanup alrdy
        )
        imputer_dict = {
            "type": "SimpleImputer",
            "imputer": imputer,
            "strategy": strat,
            "add_indicator": add_indicator,
        }
        simple_imputers.append(imputer_dict)
        
    return simple_imputers


# %% Option 2: Iterative Imputer

def create_iterative_imputers(df):
    """
    Impute missing values using IterativeImputer (experimental feature).
    
    Required imports:
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer  # noqa
    from sklearn.linear_model import BayesianRidge
    
    :param df: The dataset to impute. Used to determine the min and max values for imputation.
    :return: A list of dictionaries, each containing an imputer object and its configuration.
    """
    # Imports
    
    # Explicitly require experimental feature
    from sklearn.experimental import enable_iterative_imputer  # noqa
    # Now we can import normally from sklearn.impute
    from sklearn.impute import IterativeImputer  # noqa
    from sklearn.linear_model import BayesianRidge
    
    
    # Configuration
    
    estimator = BayesianRidge()  # can probably be customized, but leave default for now
    max_iter = 100  # try low number of iterations first, see if converges, then try higher numbers
    tol = 1e-3  # might need to adjust
    initial_strategy = ["mean"]  # ["mean", "median", "most_frequent", "constant"]
    n_nearest_features = [100, 500, None]  # [10, 100, 500, None]  # try low numbers first, None means all
    # features
    imputation_order = ["ascending"]  # ["ascending", "descending" "random"]
    # ascending: From the features with the fewest missing values to those with the most
    min_value = 0  # no features have negative values, adjust tighter for prot intensities?
    max_value = 1.1*utils.summary_statistics(df, range(11, df.shape[1]))[1].max()
    # We set max_value to 1.1*max value of the dataset, to avoid imputing values significantly
    # higher than the original data.
    
    
    ## Create imputers with different configurations
    iterative_imputers = []
    for strat in initial_strategy:
        for order in imputation_order:
            imputer = IterativeImputer(
                estimator=estimator,
                missing_values=np.nan,
                sample_posterior=False,
                max_iter=max_iter,
                tol=tol,
                n_nearest_features=n_nearest_features[1],
                initial_strategy=strat,
                imputation_order=order,
                skip_complete=False,
                min_value=min_value,
                max_value=max_value,
                verbose=verbose,
                random_state=seed,
                add_indicator=False,        # interesting for later, TODO: explore
                keep_empty_features=False,  # no effect: we have removed empty features in cleanup
            )
            imputer_dict = {
                "type": "IterativeImputer",
                "imputer": imputer,
                "max_iter": max_iter,
                "tol": tol,
                "n_nearest_features": n_nearest_features[1],
                "initial_strategy": strat,
                "imputation_order": order,
                "random_state": seed,
            }
            iterative_imputers.append(imputer_dict)
            
    return iterative_imputers


# %% Option 3: KNN Imputer

def create_KNN_imputers():
    """
    Impute missing values using K-Nearest Neighbour Imputer.
    
    Required imports:
    from sklearn.impute import KNNImputer
    
    :return: A list of dictionaries, each containing an imputer object and its configuration.
    """
    from sklearn.impute import KNNImputer
    
    # Configuration
    n_neighbours = [5, 10, 20, 30, 40, 50]  # initial span of neighbours considering dataset size
    weights = ['uniform', 'distance']  # default='uniform', callable has potential for later fitting
 
    KNN_imputers = []
    for num in n_neighbours:
        for weight in weights:
            imputer = KNNImputer(
                missing_values=np.nan,  # default
                n_neighbors=num,  # default = 5
                weights=weight,
                metric='nan_euclidean',  # default, callable has potential for later fitting
                copy=True,  # default, best option for reuse of dataframe dataset
                add_indicator=False,  # default, interesting for later, TODO: explore
                keep_empty_features=False  # default, we have removed empty features in cleanup
            )
            imputer_dict = {
                'type': 'KNNImputer',
                'imputer': imputer,
                'n_neighbors': num,
                'weights': weight,
            }
            KNN_imputers.append(imputer_dict)
            
    return KNN_imputers


# %% Option 4: NaN Elimination

def eliminate_nan(df):
    """
    Drop rows with any NaN values in the dataset.
    
    :return: A list of dictionaries, each containing the type of imputation, the imputed dataset, and the date of imputation.
    """
    df_dropped = df.copy().dropna()
    
    return [{'type': 'nan_elimination', 'dataset': df_dropped, 'date': pd.Timestamp.now()}]


# %% Option 5: No Imputation

def no_imputer(df):
    """
    Drop rows with any NaN values in the dataset.
    """
    df = df.copy()  # TODO: do we need to copy?
    return [{'type': 'no_imputation', 'dataset': df, 'date': pd.Timestamp.now()}]


def impute_data(imputer_dict, df, cols=11):
    """
    Impute missing values in the dataset using the specified imputer.
    
    :param imputer_dict: A dictionary containing the imputer object and its configuration.
    :param df: The dataset to impute.
    :param cols: The start index of the columns to impute.
    :return: A dictionary containing the type of imputation, the imputed dataset, and the date of imputation.
    """
    # Isolate relevant data
    d = df.iloc[:, cols:]
    df_imputed = df.copy()
    df_imputed.iloc[:, cols:] = pd.DataFrame(imputer_dict['imputer'].fit_transform(d), columns=d.columns)
    
    # Add content to imputer dictionary
    imputer_dict['dataset'] = df_imputed
    imputer_dict['date'] = pd.Timestamp.now()
    
    return imputer_dict


# %% Export Imputed Data

def export_imputed_data(imputer_dict, filename=None):
    """
    Export imputed data to a CSV file.
    
    :param imputer_dict: A dictionary containing the imputer object, the imputed dataset, 
                         and the date of imputation.
    :param filename: The name of the file to which the imputed data will be saved.
    """
    if filename is None:
        imputer_string = ''
        if imputer_dict['type'] == 'SimpleImputer':
            imputer_string = (
                imputer_dict['type'] + '_' +
                imputer_dict['strategy'] + '_' +
                imputer_dict['add_indicator'] + '_' +
                imputer_dict['date'].strftime('%Y%m%d-%H%M%S')
            )
        elif imputer_dict['type'] == 'IterativeImputer':
            imputer_string = (
                imputer_dict['type'] + '_' +
                imputer_dict['max_iter'] + '_' +
                imputer_dict['tol'] + '_' +
                imputer_dict['n_nearest_features'] + '_' +
                imputer_dict['initial_strategy'] + '_' +
                imputer_dict['imputation_order'] + '_' +
                imputer_dict['random_state'] + '_' +
                imputer_dict['date'].strftime('%Y%m%d-%H%M%S')
            )
        elif imputer_dict['type'] == 'KNNImputer':
            imputer_string = (
                imputer_dict['type'] + '_' +
                imputer_dict['n_neighbors'] + '_' +
                imputer_dict['weights'] + '_' +
                imputer_dict['date'].strftime('%Y%m%d-%H%M%S')
            )
        elif imputer_dict['type'] == 'nan_elimination':
            imputer_string = (
                imputer_dict['type'] + '_' +
                imputer_dict['date'].strftime('%Y%m%d-%H%M%S')
            )
        elif imputer_dict['type'] == 'no_imputation':
            imputer_string = (
                imputer_dict['type'] + '_' +
                imputer_dict['date'].strftime('%Y%m%d-%H%M%S')
            )
        imputer_dict['dataset'].to_csv(imputer_string, index=False)
    else:
        imputer_dict['dataset'].to_csv(filename, index=False)


# %% Main

def main():
    dataset = pd.read_csv('normalized_data.csv')
    dataset.head()
    
    
if __name__ == '__main__':
    main()
