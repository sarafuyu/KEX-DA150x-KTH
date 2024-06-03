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

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load the misingness data
missing_csv = PROJECT_ROOT/'data'/'results'/'final-prestudy'/'2024-06-02-223758__X_missingness_after_select.csv'
summary = pd.read_csv(missing_csv)

# Set the index to HPA ID if it's not already set
# summary.set_index('HPA ID', inplace=True)

# %% Plotting missingness

plot_bar_missing = False
plot_box_stats = True
pair_plot = False

# %% Bar Plot for missing count and percentage
if plot_bar_missing:

    fig, ax1 = plt.subplots(figsize=(15, 7))

    # TODO: order df after feature importance
    color = 'tab:blue'
    ax1.set_xlabel('Feature importance (ascending order)')
    ax1.set_ylabel('Number of Missing Instances', color=color)
    ax1.bar(summary.index, summary['Num_Missing'], color=color, label='Missing Count')
    ax1.tick_params(axis='y', labelcolor=color) 

    #ax2 = ax1.twinx()
    #color = 'tab:green'
    #ax2.set_ylabel('Percentage of Missing Instances', color=color)
    #ax2.plot(summary.index, summary['Percentage_Missing'], color=color, marker='o', label='Missing Percentage')
    #ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Missing Data: Count and Percentage per HPA ID')
    plt.xticks(rotation=90)
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.show()

# %% Box Plot for data distribution (median, min, max)

# Filter out the 'Age' feature
summary = summary[summary['Feature_Name'] != 'Age']

if plot_box_stats:

    num_hpa_per_plot = 30
    num_plots = (len(summary) // num_hpa_per_plot) + 1  # Split into multiple plots if necessary

    for i in range(num_plots):
        subset = summary.iloc[i * num_hpa_per_plot:(i + 1) * num_hpa_per_plot]
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

        plt.xticks(ticks=np.arange(len(subset)), labels=subset['Feature_Name'], rotation=90)
        plt.title(f'Distribution of Data per Feature (Mean, Variance, Min, Max) - Plot {i + 1}')
        plt.xlabel('Feature')
        plt.ylabel('Values')
        plt.tight_layout()  # Adjust layout to fit everything
        plt.show()

if pair_plot:
    sns.pairplot(summary[['mean', 'variance', 'median', 'min', 'max']])
    plt.suptitle('Pairwise Relationships of Summary Statistics', y=1.02)
    plt.show()