#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Missingness Plotting

This script is run separately from the main script to generate missingness plots.

:Date: 2024-06-03
:Authors: Sara Rydell, Noah Hopkins

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from matplotlib.ticker import FixedLocator, FixedFormatter


# %% Data Setup

PROJECT_ROOT = Path(__file__).parents[1]
DATA_DIR = 'final-prestudy'

# Load feature related ranking and importances
importance_csv = PROJECT_ROOT / 'data' / 'results' / DATA_DIR / 'feature_ranking_info.csv'
feature_importances = pd.read_csv(importance_csv)

# Load the missingness data
missing_csv = PROJECT_ROOT / 'data' / 'results' / DATA_DIR / '2024-06-02-223758__X_missingness_after_select.csv'
stats_missing = pd.read_csv(missing_csv)

# Load the dataset
DATA_FILE = '2024-06-02-223758__FeatureSelect꞉XGB-RFE-CV_dataset_dict.pkl'
data_path = PROJECT_ROOT / 'data' / 'results' / DATA_DIR
dataset_dict = joblib.load(data_path / DATA_FILE)
X_training = dataset_dict['X_training']
X_testing = dataset_dict['X_testing']
X = pd.concat([X_training, X_testing], axis=0)

# %% Merged data on feature name
stats_importance = pd.merge(stats_missing, feature_importances, on='Feature_Name', how='inner')
summary = stats_importance.sort_values(by=["Feature Importance", "Feature Rank"], ascending=[False, True])


# %% Plotting missingness

plot_bar_missing = False
plot_box_stats_old = False
plot_violin_stats = False
plot_violin_stats_overview = False
plot_violin_stats_overview_with_age = True


# %% Bar Plot for missing count and feature importance
if plot_bar_missing:

    fig, ax1 = plt.subplots(figsize=(15, 5), dpi=300)

    bar_width = 0.7  # Set a bar width
    x = np.arange(len(summary))  # X axis
    bar_color = plt.cm.Paired(0)  # Use colorblind-friendly palette
    ax1.set_xlabel('Feature (Name)', fontsize=13)
    ax1.set_ylabel('Missingness (Frequency)', color='k', fontsize=13)
    bars = ax1.bar(x, summary['Num_Missing'], color=bar_color, width=bar_width, label='Missingness Count')

    ax2 = ax1.twinx()
    dot_color = plt.cm.Paired(1)
    ax2.set_ylabel('Feature Importance (Accuracy)', color='k', fontsize=13, rotation=270, labelpad=20)
    dots = ax2.scatter(x, summary['Feature Importance'], color=dot_color, label='Feature Importance')

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [name.split('_')[0] for name in summary['Feature_Name']], rotation=90, ha='center'
    )  # Rotate x-axis labels by 90 degrees and align center

    # Adjust margins between y-axes and first/last bars
    margins = 0
    ax1.set_xlim(-(1 + margins), len(x) + margins)

    plt.title('Feature Importances and Missingness per Selected Feature', fontsize=16)
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes, fontsize=12)
    plt.tight_layout()  # Adjust layout to fit everything

    plt.savefig(PROJECT_ROOT / 'data' / 'results' / DATA_DIR / 'missingness_feature_importance.png')
    plt.show()


# %% Box Plot for data distribution (mean, variance, min, max)

# Filter out the 'Age' feature
box_data = summary[summary['Feature_Name'] != 'Age']

if plot_box_stats_old:

    num_hpa_per_plot = 44
    num_plots = (len(box_data) // num_hpa_per_plot) # Split into multiple plots if necessary

    for i in range(num_plots):
        subset = box_data.iloc[i * num_hpa_per_plot:(i + 1) * num_hpa_per_plot]
        plt.figure(figsize=(20, 7))  # Reduced height
        width = 0.4  # Width of the box and mean line

        # Create custom box plot
        for j, row in subset.iterrows():
            idx = subset.index.get_loc(j)  # Get the integer position of the index
            hpa_id = row['Feature_Name']
            mean_val = row['Mean']
            variance = row['Variance']
            min_val = row['Min']
            max_val = row['Max']

            plt.plot([idx, idx], [min_val, max_val], color='black', zorder=1)  # Min-Max line
            plt.plot([idx - width / 2, idx + width / 2], [min_val, min_val], color='black', zorder=1)  # Min tick
            plt.plot([idx - width / 2, idx + width / 2], [max_val, max_val], color='black', zorder=1)  # Max tick
            plt.fill_between(
                [idx - width / 2, idx + width / 2], [mean_val - variance, mean_val - variance],
                [mean_val + variance, mean_val + variance], color='red', zorder=2
                )  # Variance box
            plt.plot(
                [idx - width / 2, idx + width / 2], [mean_val, mean_val], color='blue', linewidth=1, zorder=3
                )  # Mean line

        plt.xticks(ticks=np.arange(len(subset)), labels=[name.split('_')[0] for name in subset['Feature_Name']], rotation=90, ha='center')
        plt.subplots_adjust(bottom=0.35)  # Adjust the bottom margin to fit the labels
        plt.title(f'Data Distribution per Feature (Mean, Variance, Min, Max) - Plot {i + 1}/2')
        plt.xlabel('Features')
        plt.ylabel('Values (Mean, Variance, Min, Max)')
        plt.tight_layout()  # Adjust layout to fit everything
        plt.show()


# %% Violin plots for data distribution (mean, variance, min, max)

# Filter out the 'Age' feature
box_data = summary[summary['Feature_Name'] != 'Age']

if plot_violin_stats:
    # sns.set_style("whitegrid")
    num_hpa_per_plot = 44
    num_plots = (len(box_data) // num_hpa_per_plot)  # Split into multiple plots if necessary

    for i in range(num_plots):
        subset = box_data.iloc[i * num_hpa_per_plot:(i + 1) * num_hpa_per_plot]
        fig, ax = plt.subplots(figsize=(15, 7), sharey=True, dpi=300)
        width = 1.75  # Width of the box and mean line

        # Create custom violin plot
        data = []
        feature_names = []
        values = None
        for j, row in subset.iterrows():
            feature_names.append(row['Feature_Name'])
            values = X.iloc[:, j]
            data.append(values)

        # Find the global min and max values for the y-axis
        global_min = X[feature_names].min().min()
        global_max = X[feature_names].max().max()

        # # Add a small margin
        margin = 0  # 0.2 + 0.05
        y_min = 10.5  # global_min - margin
        y_max = 14  # global_max + margin

        sns.violinplot(data=data, ax=ax, color='skyblue', width=width, split=False,
                       linewidth=0.6, inner='quart',
                       inner_kws=dict(linewidth=1.25, color='k'))
        for l in ax.lines:
            l.set_linestyle('--')
            l.set_linewidth(0.6)
            l.set_color('black')
            l.set_alpha(0.8)
        for l in ax.lines[1::3]:
            l.set_linestyle('-')
            l.set_linewidth(1)
            l.set_color('black')
            l.set_alpha(0.8)

        ax.set_ylabel('$\\log_2$-Normalized Intensity', fontsize=13)
        ax.set_xlabel('Feature (HPA ID)', fontsize=13)
        ax.tick_params(axis='y')

        # Show y-axis grid lines
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)

        # Clean and rotate x-axis labels 90 degrees
        feature_names = [name.split('_')[0] for name in feature_names]
        ax.xaxis.set_major_locator(FixedLocator(range(len(feature_names))))
        ax.xaxis.set_major_formatter(FixedFormatter(feature_names))
        ax.set_xticklabels(feature_names, rotation=90)

        # Set y-axis limits
        ax.set_ylim(y_min, y_max)

        # Adjust margins between y-axes and first/last violins
        margins = 0
        ax.set_xlim(-(1 + margins), len(data) + margins)

        # Add overall title
        plt.suptitle('Data Distribution per Selected Feature', fontsize=16)

        plt.tight_layout()  # Adjust layout to fit everything and leave space for labels

        plt.savefig(PROJECT_ROOT / 'data' / 'results' / DATA_DIR / f'after_select_violin_plot_plot_{i + 1}.png')
        plt.show()


# %% Violin plots for data distribution (mean, variance, min, max)

# Filter out the 'Age' feature
box_data = summary[summary['Feature_Name'] != 'Age']

if plot_violin_stats_overview:
    num_plots = 1
    for i in range(num_plots):
        subset = box_data
        fig, ax = plt.subplots(figsize=(15, 5), sharey=True, dpi=300)
        width = 1.75  # Width of the box and mean line

        # Create custom violin plot
        data = []
        feature_names = []
        values = None
        for j, row in subset.iterrows():
            feature_names.append(row['Feature_Name'])
            values = X.iloc[:, j]
            data.append(values)

        # Find the global min and max values for the y-axis
        global_min = X[feature_names].min().min()
        global_max = X[feature_names].max().max()

        # # Add a small margin
        margin = 0.5
        y_min = global_min - margin
        y_max = global_max + margin

        sns.violinplot(data=data, ax=ax, color='skyblue', width=width, split=False,
                       linewidth=0.6, inner='quart',
                       inner_kws=dict(linewidth=1.25, color='k'))
        for l in ax.lines:
            l.set_linestyle('--')
            l.set_linewidth(0.6)
            l.set_color('black')
            l.set_alpha(0)
        for l in ax.lines[1::3]:
            l.set_linestyle('-')
            l.set_linewidth(1)
            l.set_color('black')
            l.set_alpha(0.8)

        ax.set_ylabel('$\\log_2$-Normalized Intensity', fontsize=13)
        ax.set_xlabel('Feature (HPA ID)', fontsize=13)
        ax.tick_params(axis='y')

        # Show y-axis grid lines
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)

        # Clean and rotate x-axis labels 90 degrees
        feature_names = [name.split('_')[0] for name in feature_names]
        ax.xaxis.set_major_locator(FixedLocator(range(len(feature_names))))
        ax.xaxis.set_major_formatter(FixedFormatter(feature_names))
        ax.set_xticklabels(feature_names, rotation=90)

        # Set y-axis limits
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-1, len(feature_names))

        # Add overall title
        plt.suptitle('Data Distribution per Selected Feature', fontsize=16)

        plt.tight_layout()  # Adjust layout to fit everything and leave space for labels

        plt.savefig(PROJECT_ROOT / 'data' / 'results' / DATA_DIR / f'after_select_violin_plot_overview.png')
        plt.show()


# %% Violin plots for data distribution (mean, variance, min, max)

# Filter out the 'Age' feature
box_data = summary

if plot_violin_stats_overview_with_age:
    subset = box_data
    fig, ax = plt.subplots(figsize=(15, 5), sharey=True, dpi=300)
    width = 1.75  # Width of the box and mean line
    show_only_outside_percentiles = True
    num_outliers_to_show = np.inf  # 4  # can be set to np.inf to show all outliers only if show_only_outside_percentiles is True
    first_percentile = 100 - 95.4  # 25
    last_percentile = 95.4  # 75

    # Create custom violin plot
    data = []
    feature_names = []
    values = None
    outliers = []
    outlier_indexes = []
    for i, (j, row) in enumerate(subset.iterrows()):
        feature_names.append(row['Feature_Name'])
        values = X.iloc[:, j]
        data.append(values)

        # Find outliers
        if show_only_outside_percentiles:
            p1, p3 = np.nanpercentile(values, [first_percentile, last_percentile])
            whisker_low = p1 - (p3 - p1) * 1.5
            whisker_high = p3 + (p3 - p1) * 1.5
            outliers.append(values[(values > whisker_high) | (values < whisker_low)])
        else:
            values = sorted(values.dropna())
            outliers.append(list(values[:num_outliers_to_show])+list(values[-num_outliers_to_show:]))
        num_outliers = len(outliers[i])
        outlier_indexes.append([j] * num_outliers)

    # Find the global min and max values for the y-axis
    # global_min = X[feature_names].min().min()
    global_min = X[[name for name in feature_names if name != 'Age']].min().min()
    global_max = X[feature_names].max().max()

    # # Add a small margin
    margin = 0.5
    y_min = global_min - margin
    y_max = global_max + margin

    inner = 'quart'  # 'box', 'quartiles', 'point', 'stick', None
    sns.violinplot(data=data, ax=ax, color='skyblue', width=width, split=False,
                   linewidth=0.6, inner=inner,  cut=0, # bw_method=0.2,
                   inner_kws=dict(linewidth=1.25, color='k'))
    if inner == 'quart':
        for l in ax.lines:
            l.set_linestyle('--')
            l.set_linewidth(0.6)
            l.set_color('black')
            l.set_alpha(0)
        for l in ax.lines[1::3]:
            l.set_linestyle('-')
            l.set_linewidth(1)
            l.set_color('black')
            l.set_alpha(0.8)

    # Add outliers
    for i, (points, plot_index) in enumerate(zip(outliers, outlier_indexes)):
        # Select only the two first and last outliers only if they exist
        if len(points) > 2*num_outliers_to_show:
            points = list(points)
            points = points[:num_outliers_to_show] + points[-num_outliers_to_show:]
        ax.scatter(x=(len(points)*[i]), y=points, color='k', s=2, alpha=0.5, zorder=3)


    ax.set_ylabel('$\\log_2$-Normalized Intensity', fontsize=13)
    ax.set_xlabel('Feature (HPA ID)', fontsize=13)
    ax.tick_params(axis='y')

    # Show y-axis grid lines
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)

    # Clean and rotate x-axis labels 90 degrees
    feature_names = [name.split('_')[0] for name in feature_names]
    ax.xaxis.set_major_locator(FixedLocator(range(len(feature_names))))
    ax.xaxis.set_major_formatter(FixedFormatter(feature_names))
    ax.set_xticklabels(feature_names, rotation=90)

    # Set y-axis limits
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-1, len(feature_names))

    # Add overall title
    plt.suptitle('Data Distribution per Selected Feature', fontsize=16)

    plt.tight_layout()  # Adjust layout to fit everything and leave space for labels

    plt.savefig(PROJECT_ROOT / 'data' / 'results' / DATA_DIR / f'after_select_violin_plot_overview.png')
    plt.show()