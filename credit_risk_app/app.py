import numpy as np
import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from shap_utils import (
    prepare_shap_input,
    init_shap,
    plot_local_shap_waterfall,
    generate_human_explanation
)

from counterfactual_utils import generate_counterfactual_advice


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Credit Score Classification",
    layout="wide"
)

st.markdown(
    "<h2 style='text-align:center; color:#f5a623;'>Credit Score Classification</h2>",
    unsafe_allow_html=True
)

# ===============================
# CENTERED BANNER IMAGE
# ===============================
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image(
        "https://images.moneycontrol.com/static-mcnews/2024/09/20240912124503_Credit-Score-Blog-Banner.png",
        use_container_width=True,
        caption="Credit Score Classification"
    )

# ===============================
# LOAD MODEL & ENCODERS
# ===============================
loaded_model = XGBClassifier()
loaded_model.load_model("credit_score_multi_class_xgboost_model.json")

loaded_le  = pickle.load(open("credit_score_multi_class_le.pkl", "rb"))
loaded_enc = pickle.load(open("credit_score_multi_class_ord_encoder.pkl", "rb"))

# ===============================
# INIT SHAP (CACHED & SAFE)
# ===============================
explainer = init_shap(loaded_model)

# ===============================
# SIDEBAR INPUTS
# ===============================
st.sidebar.header("User Input Parameters")

def user_input_data():
    return pd.DataFrame({
        "Num_of_Delayed_Payment": [st.sidebar.slider("Num_of_Delayed_Payment", 0, 25, 14)],
        "Num_Bank_Accounts":      [st.sidebar.slider("Num_Bank_Accounts", 0, 11, 5)],
        "Total_EMI_per_month":    [st.sidebar.slider("Total_EMI_per_month", 0.0, 1780.0, 107.0)],
        "Delay_from_due_date":    [st.sidebar.slider("Delay_from_due_date", 0, 62, 21)],
        "Changed_Credit_Limit":   [st.sidebar.slider("Changed_Credit_Limit", 0.5, 30.0, 9.4)],
        "Num_Credit_Card":        [st.sidebar.slider("Num_Credit_Card", 0, 11, 5)],
        "Outstanding_Debt":       [st.sidebar.slider("Outstanding_Debt", 0.0, 5000.0, 1426.0)],
        "Interest_Rate":          [st.sidebar.slider("Interest_Rate", 1, 34, 14)],
        "Credit_Mix":             [st.sidebar.selectbox("Credit_Mix", ["Poor", "Standard", "Good"])]
    })

df_input = user_input_data()

# ===============================
# MAIN LAYOUT
# ===============================
left_col, right_col = st.columns([1.1, 1.4])

# ===============================
# USER INPUT SUMMARY (DASHBOARD STYLE)
# ===============================
with left_col:
    st.subheader("User Input Summary")

    # ----- Payment Behaviour -----
    st.markdown(
        "<div style='background:#111827;padding:16px;border-radius:12px;margin-bottom:16px;'>"
        "<h4>💳 Payment Behaviour</h4>",
        unsafe_allow_html=True
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("⏰ Delayed Payments", df_input["Num_of_Delayed_Payment"][0])
    c2.metric("📅 Delay (days)", df_input["Delay_from_due_date"][0])
    c3.metric("📈 Interest Rate (%)", df_input["Interest_Rate"][0])
    st.markdown("</div>", unsafe_allow_html=True)

    # ----- Debt & Credit Usage -----
    st.markdown(
        "<div style='background:#111827;padding:16px;border-radius:12px;margin-bottom:16px;'>"
        "<h4>💰 Debt & Credit Usage</h4>",
        unsafe_allow_html=True
    )
    c4, c5, c6 = st.columns(3)
    c4.metric("💸 Outstanding Debt", f"{df_input['Outstanding_Debt'][0]:,.0f}")
    c5.metric("🧾 Monthly EMI", f"{df_input['Total_EMI_per_month'][0]:,.0f}")
    c6.metric("💳 Credit Cards", df_input["Num_Credit_Card"][0])
    st.markdown("</div>", unsafe_allow_html=True)

    # ----- Credit Profile -----
    st.markdown(
        "<div style='background:#111827;padding:16px;border-radius:12px;'>"
        "<h4>🧠 Credit Profile</h4>",
        unsafe_allow_html=True
    )
    c7, c8 = st.columns(2)
    c7.metric("🏦 Bank Accounts", df_input["Num_Bank_Accounts"][0])

    credit_mix = df_input["Credit_Mix"][0]
    badge_color = (
        "#dc2626" if credit_mix == "Poor"
        else "#f59e0b" if credit_mix == "Standard"
        else "#16a34a"
    )

    c8.markdown(
        f"""
        <div style="padding:12px;border-radius:10px;
                    background:{badge_color};color:white;
                    text-align:center;font-size:18px;">
        Credit Mix<br><b>{credit_mix}</b>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# PREDICTION + SHAP + EXPLANATION
# ===============================
with right_col:
    st.subheader("Prediction")

    if st.button("Make Prediction"):

        # ----- Prediction -----
        df_model = df_input.copy()
        df_model[["Credit_Mix"]] = loaded_enc.transform(df_model[["Credit_Mix"]])

        pred_idx = loaded_model.predict(df_model)[0]
        pred_label = loaded_le.inverse_transform([pred_idx])[0]

        st.success(f"Predicted Credit Score: **{pred_label}**")

        # ----- SHAP -----
        st.markdown("### Why this prediction? (SHAP)")

        X_shap = prepare_shap_input(df_input, loaded_model, loaded_enc)
        class_idx = list(loaded_le.classes_).index(pred_label)

        plot_local_shap_waterfall(
            explainer=explainer,
            model=loaded_model,
            X_row=X_shap,
            class_idx=class_idx,
            max_display=9
        )

        # ----- HUMAN READABLE EXPLANATION -----
        st.markdown("### What does this mean in simple terms?")

        shap_values = explainer(X_shap)
        shap_row = shap_values.values[0, :, class_idx]

        explanations = generate_human_explanation(
            shap_row=shap_row,
            feature_names=X_shap.columns,
            top_k=3
        )

        st.markdown(
            "<div style='background:#0f172a;padding:16px;border-radius:12px;'>",
            unsafe_allow_html=True
        )
        for line in explanations:
            st.markdown(line)
        st.markdown("</div>", unsafe_allow_html=True)

        # ----- COUNTERFACTUAL ADVICE -----
        if pred_label == "Poor":
            st.markdown("### How can you improve your credit score?")

            advice = generate_counterfactual_advice(
                shap_values_row=shap_row,
                feature_names=X_shap.columns,
                top_k=3
            )

            for tip in advice:
                st.warning(tip)
