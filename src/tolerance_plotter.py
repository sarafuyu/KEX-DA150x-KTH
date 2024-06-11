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
import itertools
from pathlib import Path
from collections.abc import Sequence
from copy import deepcopy

# External imports
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import seaborn as sns  # For heatmap or advanced plotting
from matplotlib import ticker
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator, SymmetricalLogLocator, FuncFormatter
from joypy import joyplot

# Local imports
import utils


# %% Setup

SEED = utils.RANDOM_SEED  # get random seed
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_SETS = {
    'final-prestudy': {
        'results_directory': 'final-prestudy',
        'cv_results_':       '2024-06-02-223758__GridSearchCV.pkl',
    },
    'final-prestudy-tol-normalizers': {
        'results_directory': 'final-prestudy-tol-normalizers',
        'cv_results_':       '2024-06-05-000027__GridSearchCV.pkl',
    },
    'final-prestudy-tolerances': {
        'results_directory': 'final-prestudy-tolerances',
        'cv_results_':       '2024-06-10-165351__GridSearchCV.pkl',
    },
    'final-prestudy-tolerances-all-kernels': {
        'results_directory': 'final-prestudy-tolerances-all-kernels',
        'cv_results_':       '2024-06-11-021024__GridSearchCV.pkl',
    },
}


# %% Configuration

FEATURE_SELECTION_METRIC = 'final-prestudy-tolerances-all-kernels'  # 'accuracy', 'roc_auc', 'f1'
GRIDSEARCH_METRIC = 'roc_auc'  # 'accuracy', 'roc_auc', 'f1'  # noqa
PLOT_METRIC: str = 'accuracy'

FIXED_META_PARAMS = {
    'param_normalizer': ['StandardScaler'],  # ['StandardScaler', 'MinMaxScaler']
    'param_kernel': ['poly', 'rbf', 'sigmoid'],
    # 'param_tol': np.float64(0.001),
    'param_gamma': ['scale'],
    'param_class_weight': ['balanced'],
}

CV_RESULTS_DIR = PROJECT_ROOT/'data'/'results'/RESULTS_SETS[FEATURE_SELECTION_METRIC]['results_directory']  # noqa
GRID_SEARCH_RESULTS_PATH: Path = CV_RESULTS_DIR/RESULTS_SETS[FEATURE_SELECTION_METRIC]['cv_results_']

# Verbosity level:
VERBOSE: int = 2  # Unused at the moment

# Specify parameters of interest
PARAMS_OF_INTEREST = ['C', 'degree', 'coef0', 'kernel', 'gamma', 'class_weight']
PARAM_PREFIX = 'param_'  # Prefix for the parameter columns in the cv_results_ DataFrame

# Plot x-axis scale
SVC_DEFAULT_PARAMS = {'param_C': 1.0, 'param_coef0': 0.0, 'param_gamma': 'scale', 'param_degree': 3}

# # Load the cross-validation results
gridsearch_cv = joblib.load(GRID_SEARCH_RESULTS_PATH)
cv_results = pd.DataFrame(gridsearch_cv.cv_results_)

# Make all param prefixes the same
cv_results = utils.replace_column_prefix(cv_results, ['param_classifier__'],  'param_')

# Add prefix to the parameter names
PARAMS_OF_INTEREST = [f'{PARAM_PREFIX}{param}' for param in PARAMS_OF_INTEREST]

# Sanitize the normalizer repr
cv_results['param_normalizer'] = cv_results['param_normalizer'].astype(str).str.split('(').str[0]

# Set dtype for the parameters of interest
cv_results['param_kernel'] = cv_results['param_kernel'].astype(str)
cv_results['param_normalizer'] = cv_results['param_normalizer'].astype(str)
cv_results['param_C'] = cv_results['param_C'].astype(np.float64)
cv_results['param_degree'] = cv_results['param_degree'].astype(int)
cv_results['param_coef0'] = cv_results['param_coef0'].astype(np.float64)
cv_results['param_tol'] = cv_results['param_tol'].astype(np.float64)
cv_results['param_gamma'] = cv_results['param_gamma'].astype(str)
cv_results['param_class_weight'] = cv_results['param_class_weight'].astype(str)

# # Filter the cv_results_ DataFrame to only include data fixed by the FIXED_META_PARAMS
# mask = pd.DataFrame(cv_results[FIXED_META_PARAMS.keys()] == pd.Series(FIXED_META_PARAMS)).all(axis=1)
# cv_results = cv_results[mask]
cv_results = cv_results.copy()[
        (cv_results.loc[:, 'param_tol'] >= 1e-2) &
        (cv_results.loc[:, 'param_tol'] <= 1e2)
    ]
metrics_label = f'FS-{FEATURE_SELECTION_METRIC}_GS-{GRIDSEARCH_METRIC}_PLOT-{PLOT_METRIC}'

alpha = 0  # 0.7
beta = 0   # 74
gamma = 0  # 0.1

# best_cv_results_rows = utils.get_best_params(cv_results, GRIDSEARCH_METRIC, PARAMS_OF_INTEREST, return_all=True, alpha=alpha, beta=beta, default_params=SVC_DEFAULT_PARAMS)
# best_cv_results_row = best_cv_results_rows if type(best_cv_results_rows) is pd.Series else best_cv_results_rows.iloc[0]
# best_params = best_cv_results_row.to_dict() if type(best_cv_results_row) is pd.Series else best_cv_results_row
#
# # Penalize score for higher degrees according to:
# #   Adjusted Score = Score − α×ln(degree) − β×ln(std(Score)+1) − γ×ln(gamma+1)
# if alpha > 0 or beta > 0 or gamma > 0:
#     # Create a mask for the polynomial kernel
#     poly_kernel_mask = cv_results['param_kernel'] == 'poly'
#     rbf_sigmoid_kernel_mask = cv_results['param_kernel'].isin(['rbf', 'sigmoid']) & (cv_results['param_gamma'].isin(['scale', 'auto']) == False)
#     # Adjust the score based on the conditions
#     cv_results[f'mean_test_{GRIDSEARCH_METRIC}'] = (
#         cv_results[f'mean_test_{GRIDSEARCH_METRIC}']
#         - alpha * (np.log(cv_results['param_degree']) * poly_kernel_mask)
#         - beta * (np.log(cv_results[f'mean_test_{GRIDSEARCH_METRIC}'] + 1))
#         - gamma * (np.log(cv_results['param_gamma'] + 1) * rbf_sigmoid_kernel_mask)
#     )
#
# # Find row of the best parameters in cv_results
# mask = pd.DataFrame(cv_results['param_C'] == best_params['param_C']).all(axis=1)
# mask &= cv_results['param_degree'] == best_params['param_degree']
# mask &= cv_results['param_coef0'] == best_params['param_coef0']
# mask &= cv_results['param_kernel'] == best_params['param_kernel']
# mask &= cv_results['param_gamma'] == best_params['param_gamma']
# mask &= cv_results['param_class_weight'] == best_params['param_class_weight']
#
# best_cv_results_row = cv_results[mask]
#
# best_params['param_kernel'] = best_cv_results_row['param_kernel'].values[0]
# best_params = utils.replace_column_prefix(best_params, ['classifier__'],  'param_')



# %% Plot functions


def plot_tol_ridge(cv_results_, save_path, test_metric):
    # Filter out tolerance parameter values greater or equal to 1e-3
    # cv_results_ = cv_results_[cv_results_['param_tol'] >= 1e-4]

    # Ensure the save path is a Path object
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    normalizers = ['StandardScaler']
    kernels = ['poly', 'sigmoid', 'rbf']
    class_weights = ['balanced']

    # Iterate over each combination of normalizer and kernel
    for class_weight in class_weights:
        for normalizer in normalizers:
            for kernel in kernels:
                # Filter the dataframe for the current normalizer and kernel
                df_filtered = cv_results_.copy()[(cv_results_['param_normalizer'] == normalizer) &
                                                 (cv_results_['param_kernel'] == kernel) &
                                                 (cv_results_['param_class_weight'] == class_weight)]
                if df_filtered.empty:
                    print(f"No data for {normalizer} {kernel} class weight {class_weight}!\n")
                    continue
                if kernel != 'poly':
                    pass
                    # Drop duplicate rows that differ only by 'degree'
                    df_filtered = df_filtered.drop_duplicates(subset=['param_C', 'param_tol', 'param_coef0'])
                    plot_ridge(df_filtered, save_path / f"ridgeplot-param-tol_fixed-{normalizer}_{kernel}_clsweight{class_weight}.png", test_metric, kernel, normalizer, class_weight)
                    print(f"Plotted {normalizer} {kernel} class weight {class_weight} saved at: \n{save_path}\n"
                          f"ridgeplot-param-tol_fixed-{normalizer}_{kernel}_clsweight{class_weight}.png")
                else:
                    for degree in df_filtered['param_degree'].unique():
                        # Filter the dataframe for the current degree
                        df_degree = df_filtered[df_filtered['param_degree'] == degree]
                        plot_ridge(df_degree, save_path / f"ridgeplot-param-tol_fixed-{normalizer}_fixed-{kernel}-deg{degree}-clsweight{class_weight}.png", test_metric, kernel, normalizer, class_weight)
                        print(f"Plotted {normalizer} {kernel} degree {degree} class weight {class_weight} saved at: \n{save_path}\n"
                              f"ridgeplot-param-tol_fixed-{normalizer}_fixed-{kernel}-deg{degree}-clsweight{class_weight}.png")

def plot_ridge(df, save_file, test_metric, kernel, normalizer, class_weight):
    test_metric_labels = {
        'accuracy':  'Accuracy',
        'roc_auc':   'ROC AUC',
        'f1':        'F1 Score',
        'recall':    'Recall',
        'precision': 'Precision',
    }
    # Group by the combination of hyperparameters excluding 'tol'
    hyperparams = ['param_C', 'param_coef0']
    degree = df['param_degree'].iloc[0] if kernel == 'poly' else None
    df = df.copy()  # Ensure we are working on a copy
    df['hyperparams'] = df[hyperparams].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    # Sort the dataframe by 'tol'
    df_sorted = df.sort_values(by='param_tol')

    # Create a new dataframe for ridgeline plot
    df_ridge = df_sorted.pivot(index='param_tol', columns='hyperparams', values=f'mean_test_{test_metric}')

    # Create the ridgeline plot manually using Matplotlib
    fig, ax = plt.subplots(figsize=(3, 12))

    # Reduce the vertical offset to allow for more overlap
    vertical_offset = 0.2

    # Colors for lines
    colors = sns.color_palette(palette='husl', n_colors=len(df_ridge.columns))

    # Plot each line
    for i, col in enumerate(df_ridge.columns):
        y = df_ridge[col].values
        x = df_ridge.index.values
        y_offset = y + i * vertical_offset
        ax.plot(x, y_offset, color=colors[i], alpha=1, linewidth=1)

    # Set axis labels and title
    test_metric_label = test_metric_labels[test_metric]
    yaxis = ax.yaxis
    yaxis.set_label_text(f'Mean CV {test_metric_label}', fontsize=13)
    yaxis.set_label_position('left')
    yaxis.set_label(f'Mean CV {test_metric_label}')
    yaxis_sec = ax.secondary_yaxis('right')
    yaxis_sec.set_ylabel('Hyperparameter Combinations', rotation=-90, labelpad=20, fontsize=13)
    yaxis_sec.set_yticks([])

    ax.set_xlabel('Tolerance', fontsize=13)
    ax.set_title(f'Effect of Tolerance on CV {test_metric_labels[test_metric]}\n'
                 'for Different Hyperparameter Combinations\n'
                 f'using {f'a Deg{degree}-' if degree else ''}{kernel} Kernel, {normalizer} Normalizer\n'
                 f'and Class Weight {class_weight}',
                 fontsize=16)

    # Set x-axis to log scale
    ax.set_xscale('log')

    # Set x-ticks
    ax.set_xticks([1e-2, 1e0, 1e2])
    ax.get_xaxis().set_major_formatter(ticker.LogFormatterMathtext())
    ax.get_xaxis().set_minor_formatter(ticker.NullFormatter())
    ax.get_xaxis().set_tick_params(which='both', width=1)
    ax.get_xaxis().set_tick_params(which='major', length=7)
    ax.get_xaxis().set_tick_params(which='minor', length=4, color='black')


    # Remove y-axis ticks and labels
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_formatter(plt.NullFormatter())

    # Create a small scale bar with a bracket to indicate the y-axis values
    yaxis.set_ticks([0, 1])
    yaxis.set_ticklabels([0, 1])
    x_axis_start, x_axis_end = ax.get_xlim()
    bracket_x = -0.145 + x_axis_start  # x position for the indicator bracket
    bracket = utils.add_label_band(ax, 0, 1, "Accuracy", spine_pos=bracket_x, tip_pos=bracket_x + 0.02)
    yaxis.set_label_coords(x=-0.05, y=0.5)

    # Adjust margins
    ax.margins(x=0.1)  # Add 10% margin to the left and right
    plt.subplots_adjust(left=0.2, right=0.8)  # Adjust the left and right margins

    plt.show()

    # Save the plot
    plt.tight_layout()
    fig.savefig(save_file, dpi=300, transparent=True, bbox_inches='tight')
    plt.pause(0.5)
    plt.close(fig)

# TODO: removed add_label_band duplicate, find original in utils


# Get the parameter names and values
meta_param_names = list(FIXED_META_PARAMS.keys())
meta_param_values = list(FIXED_META_PARAMS.values())

# Generate all possible combinations of parameters
combinations = list(itertools.product(*meta_param_values))

# Create a list of dictionaries from the combinations
meta_param_combinations = [dict(zip(meta_param_names, combination)) for combination in combinations]


for meta_combination in meta_param_combinations:
    # if not (meta_combination['param_gamma'] != 'auto' or meta_combination['param_class_weight'] != 'None'):  # TODO: Remove this line later
    #     continue
    # Filter the cv_results_ DataFrame to only include data fixed by the FIXED_META_PARAMS
    mask = pd.DataFrame(cv_results[list(meta_combination.keys())] == pd.Series(meta_combination)).all(axis=1)
    cv_results_filtered = cv_results[mask]

    # Plot the tolerance ridge plot
    plot_tol_ridge(cv_results_filtered, PROJECT_ROOT/'figures'/metrics_label, PLOT_METRIC)

    # print(f'Plotted {meta_combination} saved at: \n{PROJECT_ROOT/"figures"/metrics_label_}')

