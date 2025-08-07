# src/microbiome.py
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

EPS = 1e-9

def shannon_entropy(counts: np.ndarray) -> float:
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts / (s + EPS)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def compute_alpha_shannon(X_samples_by_taxa: pd.DataFrame) -> pd.Series:
    """
    X: samples x taxa (non-negative counts/abundances)
    returns: pd.Series of Shannon per sample (index = sample ids)
    """
    return X_samples_by_taxa.apply(lambda row: shannon_entropy(row.values), axis=1)

def compute_beta_braycurtis_mds(X_samples_by_taxa: pd.DataFrame, random_state=42) -> pd.DataFrame:
    """
    Returns a DataFrame with columns ['MDS1','MDS2'] indexed by sample ids.
    """
    # Replace negatives and normalize rows (optional but stabilizes distances)
    X = X_samples_by_taxa.clip(lower=0)
    # Bray–Curtis on raw non-negative counts is OK
    D = squareform(pdist(X.values, metric="braycurtis"))
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=random_state)
    coords = mds.fit_transform(D)
    return pd.DataFrame(coords, index=X.index, columns=["MDS1", "MDS2"])

def differential_abundance(
    X_samples_by_taxa: pd.DataFrame,
    y: pd.Series,
    taxa_names: list,
    alpha: float = 0.05
):
    """
    Mann–Whitney U test per taxon (ASD vs TD) + Benjamini–Hochberg FDR.
    Returns sorted results with means, log2 fold-change, p, q.
    """
    asd_mask = (y == "ASD").values
    td_mask  = (y == "TD").values

    asd_vals = X_samples_by_taxa[asd_mask]
    td_vals  = X_samples_by_taxa[td_mask]

    stats = []
    for j, taxon in enumerate(taxa_names):
        a = asd_vals.iloc[:, j].values
        b = td_vals.iloc[:, j].values
        # Mann–Whitney (two-sided)
        try:
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            # If both groups constant, skip gracefully
            stat, p = np.nan, 1.0

        asd_mean = a.mean()
        td_mean  = b.mean()
        log2_fc  = np.log2((asd_mean + EPS) / (td_mean + EPS))
        stats.append((taxon, asd_mean, td_mean, log2_fc, p))

    df = pd.DataFrame(stats, columns=["Taxon", "ASD_mean", "TD_mean", "log2FC_ASD_vs_TD", "pval"])
    # FDR correction
    df["qval"] = multipletests(df["pval"].values, method="fdr_bh")[1]
    # Sort by significance then effect size magnitude
    df = df.sort_values(["qval", df["log2FC_ASD_vs_TD"].abs().name], ascending=[True, False]).reset_index(drop=True)
    # Mark direction
    df["direction"] = np.where(df["log2FC_ASD_vs_TD"] > 0, "↑ ASD", "↑ TD")
    return df
