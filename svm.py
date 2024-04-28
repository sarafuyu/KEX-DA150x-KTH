"""
Northstar Prediction Estimation

Authors:
Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Imports

# External imports
from sklearn.svm import SVC, SVR, LinearSVR

# Local imports
import utils
verbose = utils.verbosity_level  # get verbosity level
seed = utils.random_seed  # get random seed


# %% SVM Classifier Model
# TODO: later, try probability=True

def create_svm_models(dataset_dict):
    """
    Create a list of Support Vector Machine models for classification and regression with
    different configurations.
    
    Required imports:
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    
    :return:
    """
    from sklearn.metrics import accuracy_score
    
    # Configuration
    # for poly specification for degree will be needed
    kernel_types = ['poly']  # ['linear', 'sigmoid', 'precomputed']

    # If no imputation has been done, return the dataset_dict as is
    if dataset_dict['type'] == 'no_imputation':
        # TODO: Look into if SVC can handle NaN values.
        return dataset_dict
        
    X_training = dataset_dict['X_training']
    X_testing = dataset_dict['X_testing']
    y_training = dataset_dict['y_training']
    y_testing = dataset_dict['y_testing']

    svm_model_list = []
    for kernel in kernel_types:
        # Create the SVM model
        svm_model = SVC(
            C=1.0,  # default, Regularization parameter
            kernel=kernel,
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
        svm_model_list.append({'model': svm_model, 'accuracy': accuracy})
        
    dataset_dict['svm'] = svm_model_list
    
    return dataset_dict


# %% SVR Model

svr_model = SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0,
                epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)


# %% LinearSVC Model

linear_model = LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive',
                         fit_intercept=True, intercept_scaling=1.0, dual='warn', verbose=0,
                         random_state=seed, max_iter=1000)


# %% Main

def main():
    pass


if __name__ == '__main__':
    main()
