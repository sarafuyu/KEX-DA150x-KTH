#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Plotting

This script is run separately from the main script to generate plots.

:Date: 2024-05-03
:Authors: Sara Rydell, Noah Hopkins

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Imports

# Standard library imports
import warnings
import pickle
from pathlib import Path
from collections.abc import Sequence
from copy import deepcopy

# External imports
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns  # For heatmap or advanced plotting
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator, SymmetricalLogLocator, FuncFormatter
from joypy import joyplot


# Local imports
import utils


# %% Setup

SEED = utils.RANDOM_SEED  # get random seed
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# RESULTS_SETS = {
#     'accuracy': {  # Original run (89 features), dataset dict corrupted, has refit with 'roc_auc' metric
#         'results_directory': 'IterativeImputer-RFR-tol-00175-iter-98-cutoff-17-Age-GridSearch-tol-0001-FINAL-poly-detailed-accuracy',
#         'cv_results_': '2024-05-27-073726__GridSearchCV.pkl',
#     },
# }

FEATURE_SECTION_SETS = {
    'accuracy': {
        'results_directory': 'IterativeImputer-RFR-tol-00175-iter-98-cutoff-17-Age-GridSearch-tol-0001-FINAL-poly-detailed-accuracy',
        'dataset_dict': '2024-05-27-073726__FeatureSelect꞉XGB-RFE-CV_dataset_dict.pkl',
    },
}

# %% Configuration

FEATURE_SELECTION_METRIC = 'accuracy'  # 'accuracy', 'roc_auc', 'f1'
GRIDSEARCH_METRIC = 'roc_auc'  # 'accuracy', 'roc_auc', 'f1'
PLOT_METRIC: str = 'roc_auc'

FEATURE_RESULTS_DIR = PROJECT_ROOT/'data'/'results'/FEATURE_SECTION_SETS[FEATURE_SELECTION_METRIC]['results_directory']
# Path to the cv_results_ csv file:
FEATURE_RESULTS_PATH: Path = FEATURE_RESULTS_DIR/FEATURE_SECTION_SETS[FEATURE_SELECTION_METRIC]['dataset_dict']

# Load data dict
feature_results = joblib.load(FEATURE_RESULTS_PATH)
rfecv = feature_results['feature_selection_rfecv']

with plt.rc_context(rc={'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 14,
                        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 14}):
    n_scores = len(rfecv.cv_results_["mean_test_score"])
    min_features_to_select = 1
    x = range(min_features_to_select, 200+min_features_to_select)
    n = len(x)

    # Find the index of the highest y-value
    max_index = rfecv.cv_results_["mean_test_score"][0:n].argmax()
    max_x = x[max_index]
    max_y = rfecv.cv_results_["mean_test_score"][max_index]

    # Create a figure with an elongated aspect ratio
    plt.figure(figsize=(9, 5))
    plt.title("Optimal Number of Features")
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.errorbar(
        x,
        rfecv.cv_results_["mean_test_score"][0:n],
        yerr=rfecv.cv_results_["std_test_score"][0:n],
        fmt='-o',  # Add markers to the error bars
        ecolor='gray',  # Error bar color
        elinewidth=1,  # Error bar line width
        capsize=2,  # Error bar cap size
        color='blue'  # Line color
    )

    # Add a vertical line at the highest y-value
    plt.axvline(x=max_x, color='red', linestyle='--', linewidth=2)  # Line color and width

    plt.legend()
    plt.tight_layout()

    plt.savefig(FEATURE_RESULTS_DIR/'FeatureSelect_optimal_features.png', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()
