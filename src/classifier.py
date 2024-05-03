#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from pathlib import Path
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, ParameterGrid, check_cv
from sklearn.metrics import accuracy_score
from sklearn.model_selection._validation import _fit_and_score, _warn_or_raise_about_fit_failures, _insert_error_scores
from sklearn.utils.validation import _check_method_params, check_is_fitted, indexable
from sklearn.base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
from sklearn.utils.parallel import Parallel, delayed
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import time
import numbers
import operator
import time
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict



# Local imports
import utils


# %% Setup

VERBOSE = utils.VERBOSITY_LEVEL  # get verbosity level
SEED = utils.RANDOM_SEED         # get random seed
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# %% Classes and Functions

class CustomCvGridSearch(GridSearchCV):
    def __init__(self,
                 estimator,
                 param_grid,
                 *,
                 scoring=None,
                 n_jobs=None,
                 refit=True,
                 cv=None,
                 verbose=0,
                 pre_dispatch="2*n_jobs",
                 error_score=np.nan,
                 return_train_score=False):
        self.best_params_ = None
        self.feature_names_in_ = None
        self.n_splits_ = None
        self.cv_results_ = None
        self.scorer_ = None
        self.refit_time_ = None
        self.best_estimator_ = None
        self.best_score_ = None
        self.best_index_ = None
        self.multimetric_ = None
        # Call the parent class constructor
        super().__init__(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score
        )
        # Initialize the final accuracy calculation variables
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.calculate_final_accuracy = False


    def fit_calc_final_scores(self, X, y, X_testing, y_testing, groups=None, **params):
        """
        Fit the estimator using the grid search parameters.

        :param X: Training data. Will be used for CV.
        :param y: Target values.
        :param X_testing:Testing data.
        :param y_testing: Testing target values.
        :param groups: Group labels for the samples used while splitting the dataset into train/test set.
        :param fit_params: Parameters to pass to the fit method of the estimator.
        :return: Fitted grid search object.
        """
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        **params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator, the scorer,
            and the CV splitter.

            If a fit parameter is an array-like whose length is equal to
            `num_samples` then it will be split across CV groups along with `X`
            and `y`. For example, the :term:`sample_weight` parameter is split
            because `len(sample_weights) = len(X)`.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        from itertools import product

        self.X_train = X
        self.y_train = y
        self.X_test = X_testing
        self.y_test = y_testing
        if self.X_test is not None:
            self.calculate_final_accuracy = True
        else:
            self.calculate_final_accuracy = False
        if not self.calculate_final_accuracy:
            # Fit the estimator using the training data
            super().fit(X=X, y=y, groups=groups, **params)
            return self
        # Fit the estimator using the provided testing data as well

        estimator = self.estimator
        # Here we keep a dict of scorers as is, and only convert to a
        # _MultimetricScorer at a later stage. Issue:
        # https://github.com/scikit-learn/scikit-learn/issues/27001
        scorers, refit_metric = self._get_scorers(convert_multimetric=False)

        X, y = indexable(X, y)
        params = _check_method_params(X, params=params)

        routed_params = self._get_routed_params_for_fit(params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, **routed_params.splitter.split)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=routed_params.estimator.fit,
            score_params=routed_params.scorer.score,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                        " totalling {2} fits".format(
                            n_splits, n_candidates, n_candidates * n_splits
                        )
                    )

                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, parameters), (split_idx, (train, test)) in product(
                        enumerate(candidate_params),
                        enumerate(cv.split(X, y, **routed_params.splitter.split)),
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(n_splits, len(out) // n_candidates)
                    )

                _warn_or_raise_about_fit_failures(out, self.error_score)

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]["test_scores"]
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(
                self.refit, refit_metric, results
            )
            if not callable(self.refit):
                # With a non-custom callable, we can select the best score
                # based on the best index
                self.best_score_ = results[f"mean_test_{refit_metric}"][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # here we clone the estimator as well as the parameters, since
            # sometimes the parameters themselves might be estimators, e.g.
            # when we search over different estimators in a pipeline.
            # ref: https://github.com/scikit-learn/scikit-learn/pull/26786
            self.best_estimator_ = clone(base_estimator).set_params(
                **clone(self.best_params_, safe=False)
            )

            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **routed_params.estimator.fit)
            else:
                self.best_estimator_.fit(X, **routed_params.estimator.fit)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        # Calculate final accuracy with separate test data for each grid configuration
        final_accuracies = []
        if self.cv_results_ is not None:
            for params in self.cv_results_['params']:
                estimator = self.estimator.set_params(**params)
                estimator.fit(self.X_train, self.y_train)
                final_accuracy = accuracy_score(self.y_test, estimator.predict(self.X_test))
                final_accuracies.append(final_accuracy)

        # Add final accuracy scores to cv_results
        self.cv_results_['final_accuracy'] = np.array(final_accuracies)

        return self


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
                        verb=VERBOSE,
                        max_iter_params=(-1,),
                        decision_function_params=('ovr',),
                        break_ties=False,
                        random_state=SEED,
                        verbose_grid_search=0,
                        log=print,
                        grid_search_verbosity=0,
                        return_train_score=False,
                        grid_search_scoring=0,
                        k_cv_folds=5,
                        calc_final_scores=False):
    """
    Create a list of Support Vector Machine models for classification and regression with
    different configurations.

    Required imports:

    - ``sklearn.svm.SVC``
    - ``sklearn.model_selection.GridSearchCV``

    """
    # If no imputation has been done, return the dataset_dict as is
    if dataset_dict['type'] == 'NO_IMPUTATION':
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
            'decision_function_shape': decision_function_params,
            'break_ties': break_ties,
            'random_state': [random_state],
        }
    ]
    if class_weight is not None:
        param_grid[0]['svc__class_weight'] = [class_weight]

    # Perform grid search
    clf = CustomCvGridSearch(
        estimator=svm.SVC(verbose=grid_search_verbosity),
        param_grid=param_grid,
        scoring=grid_search_scoring,
        cv=k_cv_folds,
        verbose=verbose_grid_search,
        return_train_score=return_train_score,
    )
    if calc_final_scores:
        clf = clf.fit_calc_final_scores(X_training, y_training, X_testing, y_testing)
    else:
        clf = clf.fit(X_training, y_training)

    # Calculate and log best score (accuracy)
    if hasattr(clf, 'score'):
        test_accuracy = clf.score(X_testing, y_testing)
    else:
        Warning("The classifier does not have a 'score' attribute.")
        test_accuracy = None
    if verb:
        utils.log_grid_search_results(
            pipeline_config, dataset_dict, protein_start_col=11, clf=clf, accuracy=test_accuracy, log=log
            )

    dataset_dict['svm'] = {'clf': clf, 'test_accuracy': test_accuracy}

    joblib.dump(clf, PROJECT_ROOT/'out'/Path(utils.get_file_name(dataset_dict, pipeline_config) + '.pkl'))

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
                        verb=VERBOSE,
                        max_iter_params=-1,
                        log=print,
                        grid_search_verbosity=0,
                        return_train_score=False,
                        grid_search_scoring=0,
                        k_cv_folds=5,
                        calc_final_scores=False):
    """
    Create a list of Support Vector Machine models for classification and regression with
    different configurations.

    Required imports:

    - ``sklearn.svm.SVR``
    - ``sklearn.model_selection.GridSearchCV``

    """
    # If no imputation has been done, return the dataset_dict as is
    if dataset_dict['type'] == 'NO_IMPUTATION':
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
    clf = CustomCvGridSearch(
        estimator=svm.SVR(verbose=grid_search_verbosity),
        param_grid=param_grid,
        scoring=grid_search_scoring,
        cv=k_cv_folds,
        verbose=grid_search_verbosity,
        return_train_score=return_train_score,
    )
    if calc_final_scores:
        clf = clf.fit_calc_final_scores(X_training, y_training, X_testing, y_testing)
    else:
        clf = clf.fit(X_training, y_training)

    # Log best parameters
    if verb and hasattr(clf, 'best_params_'):
        log("Best parameters combination found:")
        best_parameters = clf.best_params_
        for param_name in sorted(best_parameters.keys()):
            log(f"{param_name}: {best_parameters[param_name]}")

    # Calculate and log best score (neg mean squared error)
    if hasattr(clf, 'score'):
        test_accuracy = clf.score(X_testing, y_testing)
    else:
        Warning("The classifier does not have a 'score' attribute.")
        test_accuracy = None
    if verb:
        log(f"Test accuracy of best SVR classifier: {test_accuracy}")

    dataset_dict['svr'] = {'clf': clf, 'test_accuracy': test_accuracy}

    joblib.dump(clf, PROJECT_ROOT/'out'/Path(utils.get_file_name(dataset_dict, pipeline_config) + '.pkl'))

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
    random_state=SEED,
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
