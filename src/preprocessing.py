import numpy as np
import pandas as pd
import src.load_data as load_data

def preprocess_data(file_path):
    try:
        df = load_data.load_data(file_path)

        # Check for nulls
        if df.isnull().values.any():
            print("Null values found. Filling with 0.")
            df.fillna(0, inplace=True)
        else:
            print("No null values found.")

        # Filter out low-abundance species
        df_filtered = df[df.drop(columns='Taxonomy').sum(axis=1) > 50]

        return df_filtered

    except FileNotFoundError as e:
        print(e)
        return None

def assign_group(sample_id):
    if sample_id.startswith('A'):
        return 'ASD'
    elif sample_id.startswith('B'):
        return 'TD'
    else:
        return 'Unknown Group'
    
def get_group_labels(df):
    return [assign_group(sample) for sample in df.columns if sample != "Taxonomy"]

# if __name__ == "__main__":
#     df_processed = preprocess_data("../data/ASD meta abundance.csv")
#     if df_processed is not None:
#         print(df_processed.head())
