import numpy as np
import pandas as pd
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
 
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    return df
    
# if __name__ == "__main__":
#     load_data("../data/ASD meta abundance.csv")