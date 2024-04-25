"""
Data Normalization

Authors:
Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
# %% Imports

# External imports
import pandas as pd

# Local imports
import utils
verbose = utils.verbosity_level  # get verbosity level
seed = utils.random_seed  # get random seed


# %% Dataset Normalization

def normalize(data, columns, statistics):
    """
    Normalize given columns in the given DataFrame using the provided sequences of minimum and maximum values.

    Formula: X_normalized = (X - X_min) / (X_max - X_min)

    :param data: A dataset dict or pandas DataFrame to normalize.
    :param columns: A list of column indices to normalize.
    :param statistics: A tuple of pandas Series of minimum and maximum values for each column.
    :return: A normalized pandas DataFrame.
    """
    df = None
    if type(data) is dict:
        df = data['dataset']
    elif type(data) is pd.DataFrame:
        df = data
    
    min_vals, max_vals = statistics
    # Create a copy of the DataFrame
    df_normalized = df.copy()
    
    # Normalize the specified columns
    index = df.shape[1] - len(columns)
    for col in columns:
        col_range = max_vals.iloc[col - index] - min_vals.iloc[col - index]
        # Normalize the current column using scaling formula
        df_normalized.iloc[:, col] = (df.iloc[:, col] - min_vals.iloc[col - index]) / col_range
    
    if type(data) is dict:
        data['dataset'] = df_normalized
        return data
    else:
        return df_normalized


# %% Main
def main():
    
    
    # %% Configuration
    cleaned_data_path = 'cleaned_data.csv'
    output_data_path = 'normalized_data.csv'
    start_col = 11
    
    
    # %% Data Initialization
    
    # Load the data
    dataset = pd.read_csv(cleaned_data_path)
    if verbose:
        print("Data loaded successfully.")
    if verbose > 1:
        dataset.head()
        
    
    # %% Data Normalization
    
    columns_to_normalize = list(range(start_col, dataset.shape[1]))
    
    ## Run statistics part for normalization
    stats = utils.summary_statistics(dataset, columns_to_normalize)
    min_values, max_values = stats
    if verbose > 2:
        print(min_values)
        print(max_values)
    
    # Perform Normalization
    dataset_normalized = normalize(dataset, columns_to_normalize, stats)


    # %% Export Normalized Data
    
    # Save the normalized data to a CSV file
    dataset_normalized.to_csv(output_data_path, index=False)
    if verbose > 0:
        dataset_normalized  # noqa


# %%
if __name__ == '__main__':
    main()
