import pandas as pd
from src.preprocessing import preprocess_data, get_group_labels

def test_preprocess_data_valid_file():
    df = preprocess_data("data/ASD_meta_abundance.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_get_group_labels():
    df = preprocess_data("data/ASD_meta_abundance.csv")
    labels = pd.Series(get_group_labels(df))
    assert isinstance(labels, pd.Series)
    assert set(labels.unique()).issubset({'ASD', 'TD', 'Unknown Group'})
