"""
Northstar Prediction Estimation

:Date: 2024-05-01
:Authors: Sara Rydell, Noah Hopkins

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Imports

# External imports
import joblib
from sklearn import svm
from sklearn.model_selection import GridSearchCV

# Local imports
import utils
verbose = utils.verbosity_level  # get verbosity level
seed = utils.random_seed         # get random seed


# %% SVM Classifier Model
# TODO: later, try probability=True

def find_best_svm_model(pipeline_config,
                        dataset_dict,
                        C_params=(1.0,),
                        kernels=('poly',),
                        degree_params=(3,),
                        gamma_params=('scale',),
                        coef0_params=(0.0,),
                        shrinking=True,
                        probability=False,
                        tol_params=(0.001,),
                        cache_size_params=(200,),
                        class_weight=None,
                        verb=verbose,
                        max_iter_params=(-1,),
                        decision_function_shape_params=('ovr',),
                        break_ties=False,
                        random_state=seed,
                        verbose_grid_search=0,
                        logger=print,
                        return_train_score=False):
    """
    Create a list of Support Vector Machine models for classification and regression with
    different configurations.

    Required imports:

    - ``sklearn.svm.SVC``
    - ``sklearn.model_selection.GridSearchCV``

    :param pipeline_config: A dictionary containing the pipeline configuration.
    :param dataset_dict: A dictionary containing the dataset and other relevant information.
    :param C_params: A list of regularization parameters.
    :param kernels: A list of kernel types.
    :param degree_params: A list of polynomial degrees.
    :param gamma_params: A list of gamma values.
    :param coef0_params: A list of coef0 values.
    :param shrinking: Whether to use the shrinking heuristic.
    :param probability: Whether to enable probability estimates.
    :param tol_params: A list of tolerance values.
    :param cache_size_params: A list of cache sizes.
    :param class_weight: A dictionary with class weights.
    :param verb: Whether to print verbose output.
    :param max_iter_params: A list of maximum iterations.
    :param decision_function_shape_params: A list of decision function shapes.
    :param break_ties: Whether to break ties.
    :param random_state: The seed used by the random number generator.
    :param logger: A logging function.
    :param return_train_score: Whether to return the training score.
    :return: dataset_dict with the SVM models added to dataset_dict['svm'].
    """
    # If no imputation has been done, return the dataset_dict as is
    if dataset_dict['type'] == 'no_imputation':
        # TODO: Test run to see if SVC can handle NaN values. Probably we need to convert into a sparse matrix first.
        return dataset_dict
    if dataset_dict['type'] == 'sparse':
        # TODO(Sara): do we need to change any SVR/grid search params to make it work with sparse data?
        #             If so, we can do that here.
        pass

    # Extract the training and testing data
    X_training = dataset_dict['X_training']
    X_testing = dataset_dict['X_testing']
    y_training = dataset_dict['y_training']
    y_testing = dataset_dict['y_testing']

    # Construct parameter grid
    param_grid = [
        {
            'C': C_params,
            'kernel': kernels,
            'degree': degree_params,
            'gamma': gamma_params,
            'coef0': coef0_params,
            'shrinking': shrinking,
            'probability': probability,
            'tol': tol_params,
            'cache_size': cache_size_params,
            'verbose': [verb],
            'max_iter': max_iter_params,
            'decision_function_shape': decision_function_shape_params,
            'break_ties': break_ties,
            'random_state': [random_state],
        }
    ]
    if class_weight is not None:
        param_grid[0]['svc__class_weight'] = [class_weight]

    # Perform grid search
    clf = GridSearchCV(
        estimator=svm.SVC(),
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        verbose=verbose_grid_search,
        return_train_score=return_train_score
    )
    clf = clf.fit(X_training, y_training)

    # Calculate and log best score (accuracy)
    if hasattr(clf, 'score'):
        test_accuracy = clf.score(X_testing, y_testing)
    else:
        Warning("The classifier does not have a 'score' attribute.")
        test_accuracy = None
    if verb:
        utils.log_grid_search_results(pipeline_config, dataset_dict, protein_start_col=11, clf=clf, accuracy=test_accuracy, logger=logger)

    dataset_dict['svm'] = {'clf': clf, 'test_accuracy': test_accuracy}

    joblib.dump(clf, utils.get_file_name(dataset_dict) + '.pkl')
    
    return dataset_dict


# %% SVR Model

def find_best_svr_model(pipeline_config,
                        dataset_dict,
                        kernels='rbf',
                        degree_params=3,
                        gamma_params='scale',
                        coef0_params=0.0,
                        tol_params=0.001,
                        C_params=1.0,
                        epsilon_params=0.1,
                        shrinking_params=True,
                        cache_size_params=200,
                        verb=verbose,
                        max_iter_params=-1,
                        verbose_grid_search=0,
                        logger=print,
                        return_train_score=False):
    """
    Create a list of Support Vector Machine models for classification and regression with
    different configurations.

    Required imports:

    - ``sklearn.svm.SVR``
    - ``sklearn.model_selection.GridSearchCV``

    """
    # If no imputation has been done, return the dataset_dict as is
    if dataset_dict['type'] == 'no_imputation':
        # TODO: Test run to see if SVR can handle NaN values. Probably we need to convert into a sparse matrix first.
        return dataset_dict
    if dataset_dict['type'] == 'sparse':

        # TODO: do we need to change any SVR/grid search params to make it work with sparse data?
        #             If so, we can do that here.
        pass

    # Extract the training and testing data
    X_training = dataset_dict['X_training']
    X_testing = dataset_dict['X_testing']
    y_training = dataset_dict['y_training']
    y_testing = dataset_dict['y_testing']

    # Convert to float
    y_testing = y_testing.astype(float)
    y_training = y_training.astype(float)

    # Construct parameter grid
    param_grid = [
        {
            'kernel': kernels,
            'degree': degree_params,
            'gamma': gamma_params,
            'coef0': coef0_params,
            'tol': tol_params,
            'C': C_params,
            'epsilon': epsilon_params,
            'shrinking': shrinking_params,
            'cache_size': cache_size_params,
            'verbose': [verb],
            'max_iter': max_iter_params,
        }
    ]

    # Perform grid search
    clf = GridSearchCV(
        estimator=svm.SVR(),
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=verbose_grid_search,
        return_train_score=return_train_score
    )
    clf = clf.fit(X_training, y_training)


    # Log best parameters
    if verb and hasattr(clf, 'best_params_'):
        logger("Best parameters combination found:")
        best_parameters = clf.best_params_
        for param_name in sorted(best_parameters.keys()):
            logger(f"{param_name}: {best_parameters[param_name]}")

    # Calculate and log best score (neg mean squared error)
    if hasattr(clf, 'score'):
        test_accuracy = clf.score(X_testing, y_testing)
    else:
        Warning("The classifier does not have a 'score' attribute.")
        test_accuracy = None
    if verb:
        logger(f"Test accuracy of best SVR classifier: {test_accuracy}")

    dataset_dict['svr'] = {'clf': clf, 'test_accuracy': test_accuracy}

    joblib.dump(clf, utils.get_file_name(dataset_dict) + '.pkl')

    return dataset_dict


# %% LinearSVC Model

linear_model = svm.LinearSVR(
    epsilon=0.0,
    tol=0.0001,
    C=1.0,
    loss='epsilon_insensitive',
    fit_intercept=True,
    intercept_scaling=1.0,
    dual='warn',
    verbose=0,
    random_state=seed,
    max_iter=1000
)


# %%

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

"""
# Ridge Regression
ridge_model = Ridge(alpha=1.0)

# Lasso Regression
lasso_model = Lasso(alpha=0.1) # adjust for different sparsity levels

ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)

# Making predictions
ridge_predictions = ridge_model.predict(X_test)
lasso_predictions = lasso_model.predict(X_test)

# Calculating Mean Squared Error
ridge_mse = mean_squared_error(y_test, ridge_predictions)
lasso_mse = mean_squared_error(y_test, lasso_predictions)

print("Ridge MSE:", ridge_mse)
print("Lasso MSE:", lasso_mse)
"""

# %% Main

def main():
    pass


if __name__ == '__main__':
    main()
