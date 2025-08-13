import pandas as pd
import numpy as np
import os

def split_csv(source_filepath, train_filepath, valid_filepath, train_split_ratio=0.8):
    """
    Splits a CSV file into training and validation sets.

    Args:
        source_filepath (str): The full path to the source CSV file.
        train_filepath (str): The full path to save the training CSV file.
        valid_filepath (str): The full path to save the validation CSV file.
        train_split_ratio (float): The proportion of the dataset to allocate to the training set.
        random_seed (int): Seed for the random number generator for reproducibility.
    """
    if not os.path.exists(source_filepath):
        print(f"Error: Source file not found at {source_filepath}")
        return

    print(f"Reading data from {source_filepath}...")
    # Read the source csv
    df = pd.read_csv(source_filepath)

    # --- The Core Splitting Logic ---

    

    # 2. Calculate the split index
    total_rows = len(df)
    split_index = int(total_rows * train_split_ratio)

    # 3. Split the DataFrame into training and validation sets
    train_df = df[:split_index]
    valid_df = df[split_index:]

    # --- Saving the results ---

    print(f"\nOriginal dataset has {total_rows} rows.")
    print(f"Splitting into {len(train_df)} training rows and {len(valid_df)} validation rows.")

    # Save the new dataframes to CSV files
    # index=False prevents pandas from writing the DataFrame index as a column
    train_df.to_csv(train_filepath, index=False)
    print(f"\nSuccessfully created training file: {train_filepath}")

    valid_df.to_csv(valid_filepath, index=False)
    print(f"Successfully created validation file: {valid_filepath}")


# --- Main execution block ---
if __name__ == "__main__":

    # Define your file paths
    SOURCE_CSV = 'csv_augmented/LUNA25_Public_Training_Development_Data_augmented.csv'  # <-- CHANGE THIS to your source file
    TRAIN_CSV = 'csv_augmented/train.csv'
    VALID_CSV = 'csv_augmented/valid.csv'

    # Call the function to perform the split
    split_csv(SOURCE_CSV, TRAIN_CSV, VALID_CSV)