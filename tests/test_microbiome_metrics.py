import pandas as pd
from src.micriobome import (
    compute_alpha_shannon,
    compute_beta_braycurtis_mds,
    differential_abundance
)

def get_sample_df():
    df = pd.read_csv("data/ASD_meta_abundance.csv", index_col=0)
    return df.T, df.columns.str.contains("ASD").map({True: "ASD", False: "TD"})

def test_alpha_diversity_plot():
    df, labels = get_sample_df()
    fig = compute_alpha_shannon(df.T, labels)
    assert fig is not None
    assert "data" in fig.to_dict()

def test_beta_diversity_plot():
    df, labels = get_sample_df()
    fig = compute_beta_braycurtis_mds(df.T, labels)
    assert fig is not None
    assert "data" in fig.to_dict()

def test_differential_abundance():
    df, labels = get_sample_df()
    fig = differential_abundance(df.T, labels)
    assert fig is not None
    assert "data" in fig.to_dict()
