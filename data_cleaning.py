def clean_data(dataset=None):
    """
    # Data Cleaning
    
    Data visualization, cleaning, division, and normalization.
    
    * Execute small adjustments/renaming of columns
    * Disease parameter: DMD/Cnt -> 1/0
    * Remove rows if:
      * Sample.ID's value is "BLANK", "POOL 1" or "POOL 2"
      * Disease or Sample.ID value for row is missing
    * If there are multiple rows with same Sample.ID, drop the duplicate rows with fewer value entires
      * Might need to evalueate manually or reevaluate the heuristiics (which proteins to prioritize over others)
    * Remove columns of antibody consentrations if there are no data entries
    
    * TODO: debugg, continue with later points
    * Potentionally normalize intensities: [-1, 1] or [0, 1]
    * Create new column for LoA based on FT1-5
    
    * Flytta om ordningen på kolumnerna så att varje rad blir en vektor på formen <<METADATA>,<Y>,<X>> där <X> är [LoA], <Y> är intensiteterna (och eventuellt ålder senare) och <METADATA> är allt annat.
    
    * Run a SVM with our input data
    """
    # %%
    ## Imports
    import pandas as pd
    
    # %%
    # Load CSV file through pandas dataframe
    if 'dataset' not in locals() or dataset is None:
        dataset = pd.read_csv('normalised_data_all_w_clinical_kex_20240321.csv')
    dataset.head()  # Pre-view first five rows
    
    # %%
    ## First Data Visualization & Processing
    
    # Dictionary for value conversion
    token_to_val = {"DMD": 1, "Cnt": 0}
    # Rename columns
    dataset.rename(columns={dataset.columns[0]: 'ID'}, inplace=True)
    dataset.rename(columns={'Sample.ID': 'Sample_ID'}, inplace=True)
    dataset.rename(columns={'Participant.ID': 'Participant_ID'}, inplace=True)
    dataset.rename(columns={'dataset': 'Dataset'}, inplace=True)
    dataset.rename(columns={'patregag': 'Age'}, inplace=True)
    
    # Replace the string values in the column using the mapping in token_to_val
    dataset['Disease'] = dataset['Disease'].replace(token_to_val)
    
    # Verify the change
    print(dataset.columns)
    dataset.head()
    
    # %%
    # Give control group (non DMD) default value of 34 (top score) on FT5
    in_control = dataset['Disease'] == 0.0
    control_index = in_control[in_control == True].index
    dataset.loc[control_index, 'FT5'] = 34
    
    # Verify change
    print(dataset.iloc[:15, 7:12])
    
    # %%
    ## Column Based Data Clean-up

    def calculate_column_value_percentage(df, start_column=1):  # TODO: add end_column param
        """
        Calculates the percentage of actual (non-NA) data points for each column in a pandas DataFrame
        within a specified column interval.
        
        :param df: A pandas DataFrame with potential NA values.
        :param start_column: The starting column index for the interval (1-based index).
        :param end_column: The ending column index for the interval. If None, calculates up to
        the last column.
        :return: A pandas Series with the percentage of non-NA values for each column in the interval.
        """
        # Adjust for 0-based indexing
        start_index = max(0, start_column - 1)
        
        # Select only the columns within the specified interval
        interval_df = df.iloc[:, start_index:]
        
        # Calculate the total number of non-NA values for each column
        value_counts = interval_df.count()
        
        # Calculate the total number of rows (to handle potential NA rows)
        total_rows = len(df)
        
        # Calculate the percentage of non-NA values for each column
        val_percentage = (value_counts / total_rows) * 100
        
        return val_percentage
    
    
    # %%
    # Calculate column statistics for low content columns
    value_percentage = calculate_column_value_percentage(dataset, 15)
    limit = 50
    low_percentage_columns = value_percentage[value_percentage < limit]
    
    # Visualize status
    num = 0
    for column, percentage in low_percentage_columns.items():
        print(f"Column {column} has {percentage:.2f}% non-NA values")
        num += 1
    
    print(f"We have {num} proteins with less than {limit}% datapoints")
    
    # %%
    # Remove empty columns
    columns_to_drop = low_percentage_columns.index
    print("Columns to drop:", columns_to_drop)
    
    # Check changes
    print("Before drop:", dataset.shape)
    dataset.drop(labels=columns_to_drop, axis="columns", inplace=True)
    print("After drop:", dataset.shape)
    
    # %%
    # Remove abundant data and calibration columns
    print("Before drop:", dataset.shape)
    dataset.drop(labels=['TREAT', 'Plate', 'Location', 'Empty_SBA1_rep1', 'Rabbit.IgG_SBA1_rep1'],
                 axis='columns', inplace=True)
    print("After drop:", dataset.shape)
    
    # %%
    ## Row Based Data Clean-up
    
    def remove_wrong_value_rows(df, column_name, wrong_val):
        """
      Removes rows from the DataFrame where the specified column has the specified wrong value.
    
      :param df: A pandas DataFrame from which rows will be removed.
      :param column_name: The name of the column to check for the wrong value.
      :param wrong_val: The value considered wrong in the specified column.
      :return: A pandas DataFrame with rows containing the wrong value in the specified column removed.
      """
        if isinstance(wrong_val, str):
            wrong_val = list([wrong_val])
        
        for val in wrong_val:
            # Find indices of rows with the wrong value
            incorrect = dataset[column_name] == val
            idxs_to_drop = incorrect[incorrect == True].index
            # Drop these rows
            df.drop(idxs_to_drop, inplace=True)
        return df
    
    # %%
    # Drop rows with invalid sample data
    print("Before drop:", dataset.shape)
    dataset = remove_wrong_value_rows(dataset, 'Sample_ID', ['BLANK', 'POOL 1', 'POOL 2'])
    print("After drop:", dataset.shape)
    
    # %%
    # Drop rows with NaN in the row's key values
    print("Before drop:", dataset.shape)
    dataset.dropna(subset=['Sample_ID', 'Disease'], inplace=True)
    print("After drop:", dataset.shape)
    
    # %%
    ## Handle sample duplicates
    def get_duplicate_indices(df, cols):
        """
      Find indices of rows with the wrong value in the specified column.
      """
        duplicate = df.duplicated(subset=cols, keep=False)
        duplicate_idxs = duplicate[duplicate == True].index
        return duplicate_idxs
    
    # %%
    def calculate_row_value_percentage(df, start_column=0):
        """
      Calculates the percentage of actual (non-NA) data points for each row in a pandas DataFrame.
    
      :param start_column:
      :param df: A pandas DataFrame with potential NA values.
      :return: A pandas Series with the percentage of non-NA values for each row.
      """
        # Adjust for 0-based indexing
        start_index = max(0, start_column - 1)
        
        # Select only the columns within the specified interval
        interval_df = df.iloc[:, start_index:]
        
        # Calculate the number of non-NA values per row
        value_counts_per_row = df.notna().sum(axis=1)
        
        # Calculate the total number of columns (to handle potential NA values)
        total_columns = interval_df.shape[1]
        
        # Calculate the percentage of non-NA values for each row
        value_percentage_per_row = (value_counts_per_row / total_columns) * 100
        
        return value_percentage_per_row
    
    # %%
    def remove_duplicate_rows(df, duplicate_idxs, row_val_perc):
        for i in duplicate_idxs:
            # For each duplicate find the duplicate sample.ID value using the index
            sample_ID = df.iloc[i]['Sample_ID']
            
            # Find all row indices of occurrences of the value
            duplicate_sample_ID_indices = df.index[df['Sample_ID'] == sample_ID]
            
            # Find which of these rows have the highest percentage in row_val_percentages
            best_index = -1
            best_val = -1
            for duplicate_idx in duplicate_sample_ID_indices:
                val = row_val_perc.loc[duplicate_idx]
                
                if val > best_val:
                    best_val = val
                    best_index = duplicate_idx
            
            # Remove best from list of duplicates
            duplicate_sample_ID_indices = duplicate_sample_ID_indices.drop(best_index)
            
            # Drop the rest of the duplicates
            df.drop(index=duplicate_sample_ID_indices, inplace=True)
    
    # %%
    # Remove duplicate rows for same Sample_ID
    duplicate_indexes = get_duplicate_indices(dataset, 'Sample_ID')
    row_val_percentages = calculate_row_value_percentage(dataset, start_column=15)
    
    # Check changes
    print("Before drop:", dataset.shape)
    remove_duplicate_rows(dataset, duplicate_indexes, row_val_percentages)
    print("After drop:", dataset.shape)
    
    # %%
    ## Row handling based on FT5 (Should consider data generation based on age/other tests)

    # Drop rows with NaN values in the FT5 column
    not_na = dataset['FT5'].notna()
    indices_to_drop = not_na[not_na == False].index
    
    # Check changes
    print("Before drop:", dataset.shape)
    dataset.drop(indices_to_drop, inplace=True)
    print("After drop:", dataset.shape)
    
    dataset.head(15)
    
    # %%
    ## Cleaned Data Export

    # Export cleaned data to a new CSV file
    dataset.to_csv('cleaned_data.csv', index=False)
    dataset.head()
    
    # %%
    
    return dataset
