# %% md
# Data Normalization
# %%
## Imports & Data Initialization
import pandas as pd

dataset = pd.read_csv('cleaned_data.csv')
dataset.head()
# %%
## Summary statistics for whole data set

columns_to_normalize = list(range(4, dataset.shape[1]))
max_values = dataset.iloc[:, columns_to_normalize].max()
min_values = dataset.iloc[:, columns_to_normalize].min()
med_values = dataset.iloc[:, columns_to_normalize].median()
# %%
print(max_values)
# %%
print(min_values)
# %%
print(med_values)

# %%
## Normalize dataset

def normalize(df, columns, min_vals, max_vals):
    """
    Normalize given columns in the given DataFrame using the provided sequences of minimum and maximum values.

    Formula: X_normalized = (X - X_min) / (X_max - X_min)

    :param df: A pandas DataFrame to normalize.
    :param columns: A list of column indices to normalize.
    :param min_vals: A pandas Series of minimum values for each column.
    :param max_vals: A pandas Series of maximum values for each column.
    :return: A normalized pandas DataFrame.
    """
    # Create a copy of the DataFrame
    df_normalized = df.copy()
    
    # Normalize the specified columns
    for col in columns:
        col_range = max_vals[col - 4] - min_vals[col - 4]
        # Normalize the current column using scaling formula
        df_normalized.iloc[:, col] = (df.iloc[:, col] - min_vals[
            col - 4]) / col_range  
    
    return df_normalized


# %%
# Execute Normalization
dataset_normalized = normalize(dataset, columns_to_normalize, min_values, max_values)
# %%
# Export normalized data to a new CSV file
dataset_normalized.to_csv('normalized_data.csv', index=False)
dataset_normalized