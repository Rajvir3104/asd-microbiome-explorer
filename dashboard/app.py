import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_training import train_models
from src.preprocessing import preprocess_data, get_group_labels
from src.microbiome import (
    compute_alpha_shannon,
    compute_beta_braycurtis_mds,
    differential_abundance
)


import dash
from dash import dcc, html, dash_table
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

# ---------- Train and get model outputs (in-memory) ----------
DATA_PATH = "/Users/rajvir/Desktop/Personal/asd-microbiome-explorer/data/ASD_meta_abundance.csv"
model, X_test, y_test, y_pred, cv_score, taxa_names = train_models(DATA_PATH)

# Build X_all and y_all for diversity/diff-abundance from the same file
df_raw = preprocess_data(DATA_PATH)                               # rows = taxa, columns = samples (+ 'Taxonomy')
X_all = df_raw.drop(columns=["Taxonomy"]).T                       # samples x taxa
sample_ids = X_all.index
y_all = pd.Series([ "ASD" if s.startswith("A") else ("TD" if s.startswith("B") else "UNK") for s in sample_ids ],
                  index=sample_ids, name="label")
mask = y_all.isin(["ASD", "TD"])
X_all = X_all.loc[mask]
y_all = y_all.loc[mask]

# ---------- (Tab 1) Confusion Matrix ----------
class_labels = ["ASD", "TD"]
cm = confusion_matrix(y_test, y_pred, labels=class_labels)
z = cm.tolist()
z_text = [[str(v) for v in row] for row in z]
conf_matrix_fig = ff.create_annotated_heatmap(z, x=class_labels, y=class_labels, annotation_text=z_text, colorscale="Viridis")
conf_matrix_fig.update_layout(
    title_text=f"<b>Confusion Matrix</b> (CV F1: {cv_score:.2f})",
    xaxis_title="Predicted",
    yaxis_title="True",
)
conf_matrix_fig["layout"]["yaxis"]["autorange"] = "reversed"

# Feature importance (tree models)
feature_fig = {}
if hasattr(model.named_steps["classifier"], "feature_importances_"):
    importances = model.named_steps["classifier"].feature_importances_
    imp_df = pd.DataFrame({"Species": taxa_names, "Importance": importances}).sort_values("Importance", ascending=False).head(20)
    feature_fig = px.bar(imp_df, x="Importance", y="Species", orientation="h", title="Top 20 Microbial Taxa (Feature Importance)", color="Importance", color_continuous_scale="Viridis")
    feature_fig.update_layout(yaxis=dict(autorange="reversed"))

# ---------- (Tab 2) Diversity ----------
# Alpha (Shannon)
shannon = compute_alpha_shannon(X_all)  # Series indexed by sample
alpha_df = pd.DataFrame({"SampleID": shannon.index, "Shannon": shannon.values, "Label": y_all.values})

alpha_box = px.box(alpha_df, x="Label", y="Shannon", points="all", title="Alpha Diversity (Shannon) by Group")

# Beta (Bray–Curtis MDS)
mds_df = compute_beta_braycurtis_mds(X_all)
mds_df["Label"] = y_all.values
beta_scatter = px.scatter(mds_df, x="MDS1", y="MDS2", color="Label", title="Beta Diversity (Bray–Curtis MDS)", hover_name=mds_df.index)

# ---------- (Tab 3) Differential Abundance ----------
diff_df = differential_abundance(X_all, y_all, taxa_names)
top_n = 20
diff_top = diff_df.head(top_n)

diff_bar = px.bar(
    diff_top.assign(sign=np.sign(diff_top["log2FC_ASD_vs_TD"])),
    x="log2FC_ASD_vs_TD", y="Taxon",
    orientation="h",
    title=f"Differential Abundance (Top {top_n}) — log2FC (ASD vs TD)",
    color="direction",
)
diff_bar.update_layout(yaxis=dict(autorange="reversed"))

# Data table of results
table_cols = ["Taxon", "ASD_mean", "TD_mean", "log2FC_ASD_vs_TD", "pval", "qval", "direction"]
diff_table = diff_top[table_cols].copy()
diff_table["ASD_mean"] = diff_table["ASD_mean"].round(4)
diff_table["TD_mean"]  = diff_table["TD_mean"].round(4)
diff_table["log2FC_ASD_vs_TD"] = diff_table["log2FC_ASD_vs_TD"].round(3)
diff_table["pval"] = diff_table["pval"].map(lambda x: f"{x:.2e}")
diff_table["qval"] = diff_table["qval"].map(lambda x: f"{x:.2e}")

# ---------- Dash App ----------
app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.H1("ASD Microbiome Explorer", style={"textAlign": "center"}),

        dcc.Tabs([
            dcc.Tab(label="Model Results", children=[
                html.Br(),
                dcc.Graph(figure=conf_matrix_fig),
                html.Br(),
                html.H3("Feature Importance"),
                dcc.Graph(figure=feature_fig),
            ]),

            dcc.Tab(label="Diversity", children=[
                html.Br(),
                dcc.Graph(figure=alpha_box),
                html.Br(),
                dcc.Graph(figure=beta_scatter),
            ]),

            dcc.Tab(label="Differential Abundance", children=[
                html.Br(),
                dcc.Graph(figure=diff_bar),
                html.Br(),
                html.H3("Top Differential Taxa"),
                dash_table.DataTable(
                    columns=[{"name": c, "id": c} for c in diff_table.columns],
                    data=diff_table.to_dict("records"),
                    sort_action="native",
                    filter_action="native",
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_cell={"fontFamily": "monospace", "padding": "6px"},
                )
            ]),
        ])
    ],
    style={"padding": "12px"}
)

if __name__ == "__main__":
    app.run(debug=True)
