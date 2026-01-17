import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def prepare_shap_input(df, model, encoder):
    """
    Prepare input EXACTLY as training pipeline
    """
    model_features = model.get_booster().feature_names
    X = df[model_features].copy()

    # Encode Credit_Mix
    X[['Credit_Mix']] = encoder.transform(X[['Credit_Mix']])

    return X.astype(np.float32)


@st.cache_resource
def init_shap(_model):
    """
    Initialize SHAP explainer (cached, Streamlit-safe)
    """
    explainer = shap.Explainer(
        _model,
        algorithm="tree"
    )
    return explainer


def plot_local_shap_waterfall(
    explainer,
    model,
    X_row,
    class_idx,
    max_display=9
):
    """
    Correct SHAP waterfall for ONE instance (multiclass-safe)
    """
    shap_values = explainer(X_row)

    # Extract correctly for multiclass:
    # shape = (1, n_features, n_classes)
    values = shap_values.values[0, :, class_idx]
    base_value = shap_values.base_values[0, class_idx]

    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=X_row.iloc[0],
        feature_names=X_row.columns
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(
        explanation,
        max_display=max_display,
        show=False
    )

    st.pyplot(fig)
    plt.close(fig)

def generate_human_explanation(shap_row, feature_names, top_k=3):
    """
    Convert SHAP values into a human-readable explanation.
    
    Parameters
    ----------
    shap_row : array-like
        SHAP values for a single instance & class
    feature_names : list
        Feature names in correct order
    top_k : int
        Number of most influential features to explain
    """

    shap_pairs = list(zip(feature_names, shap_row))
    shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    explanations = []

    for feature, value in shap_pairs[:top_k]:
        feature_name = feature.replace("_", " ")

        if value > 0:
            explanations.append(
                f"• **{feature_name}** increases the likelihood of a poorer credit score."
            )
        else:
            explanations.append(
                f"• **{feature_name}** helps improve the credit score."
            )

    return explanations
