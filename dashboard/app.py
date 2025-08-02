import dash
from dash import html, dcc
import pandas as pd
import joblib
import sys
import os

# Add the src directory to the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import preprocess_data, assign_group
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from evaluation import get_confusion_matrix_figure, get_feature_importance_plot



# Load model + data
model = joblib.load("../models/best_model.pkl")
df = preprocess_data("../data/ASD meta abundance.csv")

# Prepare X and y
X = df.drop(columns=["Taxonomy"]).T
y = [assign_group(sample_id) for sample_id in X.index]
X["label"] = y
X = X[X["label"] != "Unknown Group"]
y = X.pop("label")

# Predict
y_pred = model.predict(X)

# Create figures
cm_fig = get_confusion_matrix_figure(y, y_pred)
fi_fig = get_feature_importance_plot(model, df["Taxonomy"].tolist())

# Dash layout
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("ASD Microbiome Classifier Results"),

    dcc.Graph(figure=cm_fig),
    
    dcc.Graph(figure=fi_fig if fi_fig else px.scatter(title="Feature importance not available"))
])

if __name__ == "__main__":
    app.run_server(debug=True)
