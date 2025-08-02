# src/evaluation.py

import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def get_confusion_matrix_figure(y_true, y_pred, labels=["ASD", "TD"]):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title="Confusion Matrix",
        color_continuous_scale="Blues"
    )
    fig.update_layout(margin=dict(t=40, l=40, b=40, r=40))
    return fig

def get_feature_importance_plot(model, feature_names, top_n=10):
    if hasattr(model.named_steps["classifier"], "feature_importances_"):
        importances = model.named_steps["classifier"].feature_importances_
        top_indices = np.argsort(importances)[-top_n:]
        data = {
            "Species": [feature_names[i] for i in top_indices],
            "Importance": [importances[i] for i in top_indices]
        }
        df = pd.DataFrame(data)
        fig = px.bar(
            df, 
            x="Importance", 
            y="Species", 
            orientation='h',
            title="Top Feature Importances (Microbial Taxa)"
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        return fig
    return None
