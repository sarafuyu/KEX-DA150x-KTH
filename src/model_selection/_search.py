#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modifications to the `GridSearchCV` class to allow for final accuracy calculation.

This file also contains the relevant private functions from the sklearn package that
are needed to implement needed modifications to the GridSearchCV parent base class
method `fit()`.

Adapted from source code of scikit-learn version 1.4.1.post1.

License
-------

BSD 3-Clause License

Copyright (c) 2007-2024 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

:Date: 2024-05-01
:Authors: Sara Rydell, Noah Hopkins

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""

# %% Imports

# Standard library imports
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

# External imports
import numpy as np
from sklearn.model_selection import GridSearchCV, check_cv
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import indexable
from sklearn.base import clone, is_classifier
from sklearn.utils.parallel import Parallel, delayed

# Private imports from sklearn (might be removed in future versions)
from sklearn.model_selection._validation import _fit_and_score, _warn_or_raise_about_fit_failures, _insert_error_scores
from sklearn.utils.validation import _check_method_params

# TODO: Copy the relevant code from the sklearn package
#       and include it in this file to avoid future compatibility issues.

# Local imports
import utils


# %% Setup

VERBOSE = utils.VERBOSITY_LEVEL  # get verbosity level
SEED = utils.RANDOM_SEED         # get random seed
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# %% Custom GridSearchCV class

class CustomCvGridSearch(GridSearchCV):
    def __init__(self,
                 estimator,
                 param_grid,
                 *,
                 scoring=None,
                 n_jobs=None,
                 refit=True,
                 cv=None,
                 verbose=VERBOSE,
                 pre_dispatch="2*n_jobs",
                 error_score=np.nan,
                 return_train_score=False):

        # Initialize variables for custom fit method:
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
        # Initialize final accuracy calculation variables:
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.calculate_final_accuracy = False


    def fit_calc_final_scores(self, X, y, X_testing=None, y_testing=None, groups=None, **params):
        """
        Fit the estimator using all sets of grid search parameters.

        If X_testing and y_testing are provided, calculate final accuracy scores for each grid configuration.
        Otherwise, only fit the estimator using the training data (equivalent to the parent class fit() method).

        Documentation below adapted mostly from the override method:
            sklearn.model_selection.BaseSearchCV.fit()

        :param X: array-like of shape (n_samples, n_features). Training vector, where `n_samples`
            is the number of samples and `n_features` is the number of features. Used for CV.
        :param y: array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.
        :param X_testing: array-like of shape (m_samples, n_features). Testing data.
            Used to calculate final accuracy scores.
        :param y_testing: array-like of shape (m_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X_testing for classification or regression;
            None for unsupervised learning. Used to calculate final accuracy scores.
        :param groups: Group labels for the samples used while splitting the dataset into train/test set.
        :param params: Parameters to pass to the fit method of the estimator.
        :return: Instance of fitted estimator.
        """
        # Store the training and testing data
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
        # else: Fit the estimator using the provided testing data as well


        # -------------------------| From the parent class fit() method | ------------------------ #

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
                    delayed(_fit_and_score)(  # noqa
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
            self.best_estimator_ = clone(base_estimator).set_params(  # noqa
                **clone(self.best_params_, safe=False)                # noqa
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

        # --------------------------| End of parent class fit() method | ------------------------- #


        # Calculate final accuracy with the separate test data for each grid cell parameter set
        final_accuracies = []
        if self.cv_results_ is not None:
            # TODO: parallelize this loop
            for params in self.cv_results_['params']:
                estimator = self.estimator.set_params(**params)
                estimator.fit(self.X_train, self.y_train)
                final_accuracy = accuracy_score(self.y_test, estimator.predict(self.X_test))
                final_accuracies.append(final_accuracy)

        # Add final accuracy scores to cv_results
        self.cv_results_['final_accuracy'] = np.array(final_accuracies)

        return self
