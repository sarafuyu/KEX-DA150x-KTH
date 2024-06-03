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

# %% Merged data on feature name
stats_importance = pd.merge(stats_missing, feature_importances, on='Feature_Name', how='inner')
summary = stats_importance.sort_values(by=["Feature Importance", "Feature Rank"], ascending=[False, True])

print(stats_importance.head())
print(stats_importance.shape)
print(stats_importance.columns)

# %% Plotting missingness

plot_bar_missing = True
plot_box_stats = True

# %% Bar Plot for missing count and feature importance
if plot_bar_missing:

    fig, ax1 = plt.subplots(figsize=(15, 7))

    bar_width = 0.4  # Set a bar width
    x = np.arange(len(summary))  # X axis

    ax1.set_xlabel('Feature')
    ax1.set_ylabel('Number of Missing Instances', color='tab:blue')
    bars = ax1.bar(x - bar_width / 2, summary['Num_Missing'], color='tab:blue', width=bar_width, label='Missing Count')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Feature Importance (accuracy)', color='tab:green')
    dots = ax2.scatter(x + bar_width / 2, summary['Feature Importance'], color=color, label='Feature Importance')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        summary['Feature_Name'], rotation=90, ha='center'
        )  # Rotate x-axis labels by 90 degrees and align center

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.3, top=0.9)  # Adjust the bottom and top margins
    plt.title('Missing Data: Count and Feature Importance per Feature')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.show()

# %% Box Plot for data distribution (mean, variance, min, max)

# Filter out the 'Age' feature
box_data = summary[summary['Feature_Name'] != 'Age']

if plot_box_stats:

    num_hpa_per_plot = 30
    num_plots = (len(box_data) // num_hpa_per_plot) + 1  # Split into multiple plots if necessary

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
                [mean_val + variance, mean_val + variance], color='lightblue', zorder=2
                )  # Variance box
            plt.plot(
                [idx - width / 2, idx + width / 2], [mean_val, mean_val], color='blue', linewidth=1, zorder=3
                )  # Mean line

        plt.xticks(ticks=np.arange(len(subset)), labels=subset['Feature_Name'], rotation=90, ha='center')
        plt.subplots_adjust(bottom=0.35)  # Adjust the bottom margin to fit the labels
        plt.title(f'Distribution of Data per Feature (Mean, Variance, Min, Max) - Plot {i + 1}')
        plt.xlabel('Feature')
        plt.ylabel('Values')
        plt.tight_layout()  # Adjust layout to fit everything
        plt.show()
