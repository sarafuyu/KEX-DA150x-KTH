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

print(stats_importance.head())
print(stats_importance.shape)
print(stats_importance.columns)

# %% Plotting missingness

plot_bar_missing = True
plot_box_stats = False
new_plot_box_stats = False


# %% Bar Plot for missing count and feature importance
if plot_bar_missing:

    fig, ax1 = plt.subplots(figsize=(15, 4))

    bar_width = 0.4  # Set a bar width
    x = np.arange(len(summary))  # X axis

    ax1.set_xlabel('Features', fontsize=12)
    ax1.set_ylabel('Number of Missing Data Instances', color='k', fontsize=12)
    bars = ax1.bar(x - bar_width / 2, summary['Num_Missing'], color='tab:blue', width=bar_width, label='Missingness Count')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Feature Importance (accuracy)', color='k', fontsize=12)
    dots = ax2.scatter(x + bar_width / 2, summary['Feature Importance'], color=color, label='Feature Importance')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [name.split('_')[0] for name in summary['Feature_Name']], rotation=90, ha='center'
        )  # Rotate x-axis labels by 90 degrees and align center

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.3, top=0.9)  # Adjust the bottom and top margins
    plt.title('Number of Missing Data Instances and Feature Importance per Feature', fontsize=15)
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes, fontsize=12)
    plt.tight_layout()  # Adjust layout to fit everything
    plt.savefig(PROJECT_ROOT / 'data' / 'results' / DATA_DIR / 'missingness_feature_importance.png')
    plt.show()


# %% Box Plot for data distribution (mean, variance, min, max)

# Filter out the 'Age' feature
box_data = summary[summary['Feature_Name'] != 'Age']

if plot_box_stats:

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


# %% Box Plot (through matplotlibs native) for data distribution (mean, variance, min, max)

# Filter out the 'Age' feature
box_data = summary[summary['Feature_Name'] != 'Age']

if new_plot_box_stats:
    num_hpa_per_plot = 44
    num_plots = (len(box_data) // num_hpa_per_plot)  # Split into multiple plots if necessary
    for i in range(num_plots):
        subset = box_data.iloc[i * num_hpa_per_plot:(i + 1) * num_hpa_per_plot]
        fig, axes = plt.subplots(1, 44, figsize=(20, 7), sharey=True)
        # ax = fig.add_subplot(num_hpa_per_plot)
        width = 0.4  # Width of the box and mean line

        # Create custom box plot
        for (j, row), ax in zip(subset.iterrows(), axes):
            idx = subset.index.get_loc(j)  # Get the integer position of the index
            values = X.iloc[:, j]
            hpa_id = row['Feature_Name']
            mean_val = row['Mean']
            median_val = row['Median']
            variance = row['Variance']
            min_val = row['Min']
            max_val = row['Max']
            sns.violinplot(data=values, ax=ax)

            ax.set_title(hpa_id, fontsize=8)
            ax.get_yaxis().set_visible(False)  # Hide y-axis labels to reduce clutter
            ax.set_xticks([])  # Hide x-axis ticks
            ax.set_xlabel(hpa_id, fontsize=8, rotation=90)  # Set x-axis labels
            ax.xaxis.set_label_position('bottom')  # Set labels position to bottom

            # Remove outer box (spines)
            for spine in ax.spines.values():
                spine.set_visible(False)

            # ax.set_title(hpa_id, fontsize=8)
            # ax.get_yaxis().set_visible(False)  # Hide y-axis labels to reduce clutter
            # if j != 0:
            #     ax.get_xaxis().set_visible(False)  # Hide x-axis labels for all but the first


            # ax.violin(vpstats=vpstats, positions=None, vert=True, widths=0.5, showmeans=False, showextrema=True, showmedians=False)
        # plt.set_title('Violin plot')

        #plt.xticks(ticks=np.arange(len(subset)), labels=[name.split('_')[0] for name in subset['Feature_Name']], rotation=90, ha='center')
        # Adjust the bottom margin to fit the labels
        plt.subplots_adjust(bottom=0.35)
        # Add overall title
        plt.suptitle('Data Distribution per Feature (Mean, Variance, Min, Max)', y=1.02)
        # Add global x-axis label
        fig.text(0.5, 0.02, 'Features', ha='center', va='center', fontsize=12)
        # Add global y-axis label
        fig.text(
            0.02, 0.5, 'Values (Mean, Variance, Min, Max)', ha='center', va='center', rotation='vertical', fontsize=12
            )
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])  # Adjust layout to fit everything and leave space for labels
        plt.show()

