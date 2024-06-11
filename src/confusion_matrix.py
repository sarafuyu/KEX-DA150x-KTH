#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting confusion matrices

This script is run separately from the main script to generate plots.

:Date: 2024-06-10
:Authors: Sara Rydell, Noah Hopkins

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Imports

# External imports
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# %% Function for confusion martix plot.
def plot_matrix(matrix1, matrix2):
    # Updating the titles with "Final ROC-AUC" instead of "Final ROC"
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Class Weight None
    sns.heatmap(matrix1, annot=True, fmt='d', cmap='Greens', cbar=False, linewidths=.5, square=True, ax=ax[0], annot_kws={"size": 16})
    ax[0].set_title('Class Weight None\nFinal ROC-AUC: 0.38', pad=20)
    ax[0].set_ylabel('Predicted Class')
    ax[0].set_xlabel('True Class')

    # Class Weight Balanced
    sns.heatmap(matrix2, annot=True, fmt='d', cmap='Greens', cbar=False, linewidths=.5, square=True, ax=ax[1], annot_kws={"size": 16})
    ax[1].set_title('Class Weight Balanced\nFinal ROC-AUC: 0.582', pad=20)
    ax[1].set_ylabel('')
    ax[1].set_xlabel('True Class')

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()

# %% Data for the confusion matrices
data_none = {
    'Predicted Class': ['Positive', 'Negative'],
    'Positive': [0, 0],
    'Negative': [52, 9]
}

data_balanced = {
    'Predicted Class': ['Positive', 'Negative'],
    'Positive': [1, 0],
    'Negative': [51, 9]
}

df_none = pd.DataFrame(data_none).set_index('Predicted Class')
df_balanced = pd.DataFrame(data_balanced).set_index('Predicted Class')

# %% Run function
plot_matrix(df_none, df_balanced)