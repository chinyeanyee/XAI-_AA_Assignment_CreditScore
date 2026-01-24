import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
import dice_ml
from dice_ml import Dice

# ===============================
# PAGE CONFIG & PROFESSIONAL UI CSS
# ===============================
st.set_page_config(page_title="Credit Risk Intelligence", layout="wide")

st.markdown("""
    <style>
    /* Main Background and Professional Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    .main { background-color: #f8fafc; font-family: 'Inter', sans-serif; }
    
    /* Global Font Size Adjustments */
    html, body, [class*="st-"] { font-size: 1.05rem; line-height: 1.6; }
    
    /* Sidebar Styling */
    div[data-testid="stSidebar"] { background-color: #0f172a; color: white; }
    div[data-testid="stSidebar"] .stMarkdown p { font-size: 1.1rem !important; color: #cbd5e1; }

    /* Headers */
    h1 { font-size: 2.8rem !important; font-weight: 700 !important; color: #1e293b; margin-bottom: 0.5rem !important; }
    h2 { font-size: 1.8rem !important; font-weight: 600 !important; color: #334155; margin-top: 2rem !important; }
    h3 { font-size: 1.4rem !important; font-weight: 600 !important; color: #475569; }

    /* Professional Card Styling */
    .bank-card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 25px;
    }
    
    /* Decision Hero Card */
    .result-card {
        padding: 40px;
        border-radius: 16px;
        box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1);
        border-left: 12px solid;
        margin-bottom: 30px;
    }
    
    .result-label { font-size: 1.1rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b; }
    .result-value { font-size: 3.5rem !important; font-weight: 800 !important; margin: 10px 0; }
    .result-desc { font-size: 1.25rem !important; font-weight: 500; }

    /* Actionable Items List */
    .bank-card ul { font-size: 1.15rem !important; list-style-type: none; padding-left: 0; }
    .bank-card li { margin-bottom: 12px; padding-left: 1.5rem; position: relative; }
    .bank-card li::before { content: "→"; position: absolute; left: 0; color: #2563eb; font-weight: bold; }

    /* Modern Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.1rem !important;
        text-transform: uppercase;
        padding: 0.8rem 2rem;
        background-color: #2563eb;
        color: white;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { background-color: #1d4ed8; transform: translateY(-2px); border: none; }
    </style>
    """, unsafe_allow_html=True)

# ===============================
# LOAD ARTIFACTS (Original Logic)
# ===============================
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("xgb_final_model.pkl", "rb"))
    le = pickle.load(open("final_label_encoder.pkl", "rb"))
    feature_names = pickle.load(open("final_feature_names.pkl", "rb"))
    selected_features = pickle.load(open("selected_features.pkl", "rb"))
    baseline = pickle.load(open("baseline_human_encoded.pkl", "rb"))
    dice_data = pickle.load(open("dice_reference_data.pkl", "rb"))
    shap_bg = pickle.load(open("shap_background.pkl", "rb"))
    return model, le, feature_names, selected_features, baseline, dice_data, shap_bg

model, le, feature_names, selected_features, baseline, dice_data, shap_bg = load_artifacts()

# Safety Checks
assert baseline.select_dtypes(include="object").empty, "Baseline has object columns"
assert dice_data.select_dtypes(include="object").empty, "DiCE data has object columns"
assert shap_bg.select_dtypes(include="object").empty, "SHAP background has object columns"

# ===============================
# SIDEBAR — USER INPUT
# ===============================
with st.sidebar:
    st.markdown("## 🏢 Risk Management")
    st.markdown("---")
    st.header("📋 Applicant Profile")

    user_input = {}
    with st.expander("Update Financial Parameters", expanded=True):
        for col in feature_names:
            user_input[col] = st.number_input(
                col.replace('_', ' ').title(), 
                value=float(baseline[col].iloc[0])
            )

    st.markdown("---")
    predict_btn = st.button("🔮 Generate Decision", type="primary")

# ===============================
# MAIN DASHBOARD
# ===============================
st.markdown("# 🧠 Credit Decision Support System")
st.caption("Universiti Malaya Master of AI • Credit Evaluation Framework")

if predict_btn:
    # --- DATA PROCESSING ---
    X_user = pd.DataFrame([user_input])[feature_names]
    X_user = X_user.apply(pd.to_numeric, errors="raise")

    # --- PREDICTION ---
    pred_idx = model.predict(X_user)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    proba = model.predict_proba(X_user)[0]

    # --- HERO RESULT SECTION ---
    st.markdown("---")
    tier_styling = {
        "Poor": {"color": "#dc2626", "bg": "#fef2f2", "icon": "🚫", "desc": "High Risk - Immediate Review Required."},
        "Standard": {"color": "#d97706", "bg": "#fffbeb", "icon": "⚠️", "desc": "Medium Risk - Conditional Approval Possible."},
        "Good": {"color": "#16a34a", "bg": "#f0fdf4", "icon": "✅", "desc": "Low Risk - Automated Approval recommended."}
    }
    style = tier_styling.get(pred_label.title(), tier_styling["Standard"])

    col_res, col_prob = st.columns([1.3, 1], gap="large")

    with col_res:
        st.markdown(f"""
            <div class="result-card" style="background-color: {style['bg']}; border-color: {style['color']};">
                <p class="result-label">Audit Assessment Outcome</p>
                <h1 class="result-value" style="color: {style['color']};">{style['icon']} {pred_label.upper()}</h1>
                <p class="result-desc" style="color: #1e293b;">Confidence Score: <b>{np.max(proba)*100:.1f}%</b></p>
                <p class="result-desc" style="color: #475569; margin-top: 10px;">{style['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

    with col_prob:
        st.markdown("### 📊 Probability Analysis")
        for i, class_name in enumerate(le.classes_):
            st.write(f"**{class_name}** ({proba[i]*100:.1f}%)")
            st.progress(float(proba[i]))

    # --- SHAP ANALYSIS ---
    st.markdown("---")
    st.subheader("🔍 Decision Intelligence (XAI Factors)")
    
    explainer = shap.TreeExplainer(model, shap_bg)
    shap_values = explainer.shap_values(X_user)
    shap_vals = np.asarray(shap_values[pred_idx]).reshape(-1) if isinstance(shap_values, list) else np.asarray(shap_values).reshape(-1)

    base_val = float(explainer.expected_value[pred_idx])
    n = min(len(feature_names), len(shap_vals))
    shap_df = pd.DataFrame({"Feature": feature_names[:n], "SHAP Value": shap_vals[:n]})
    shap_top9 = shap_df.assign(abs=shap_df["SHAP Value"].abs()).sort_values("abs", ascending=False).head(9)

    col_list, col_plot = st.columns([1, 1.6], gap="large")
    with col_list:
        st.write("#### Factor Influence Weights")
        st.dataframe(shap_top9[["Feature", "SHAP Value"]].style.background_gradient(cmap='RdYlGn'), use_container_width=True)
    
    with col_plot:
        st.write("#### SHAP Waterfall (Visual Impact)")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_exp := shap.Explanation(values=shap_top9["SHAP Value"].values, 
                                              base_values=base_val, 
                                              feature_names=shap_top9["Feature"].tolist()), show=False)
        st.pyplot(fig)

    # --- NARRATIVE SECTION ---
    st.markdown("---")
    st.subheader("📝 Executive Summary Narrative")
    risk_increasing = shap_top9[shap_top9["SHAP Value"] > 0]
    risk_reducing   = shap_top9[shap_top9["SHAP Value"] < 0]

    def humanize_feature(name): return name.replace("_", " ").title()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="bank-card" style="border-top: 6px solid #ef4444;">
        <h4 style="color:#ef4444; margin-top:0;">🚩 Primary Risk Drivers</h4>
        {"".join([f"• <b>{humanize_feature(r.Feature)}</b> contributed significantly to risk.<br>" for r in risk_increasing.head(3).itertuples()]) or "No significant risks."}
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="bank-card" style="border-top: 6px solid #22c55e;">
        <h4 style="color:#22c55e; margin-top:0;">✅ Key Mitigating Factors</h4>
        {"".join([f"• <b>{humanize_feature(r.Feature)}</b> helped offset potential risk.<br>" for r in risk_reducing.head(3).itertuples()]) or "No mitigating factors."}
        </div>""", unsafe_allow_html=True)

    # Synthesis
    n_up, n_down = len(risk_increasing), len(risk_reducing)
    synthesis_text = (f"The assessment of **{pred_label.upper()}** reflects an imbalance where risk-increasing factors outweigh existing financial strengths." 
                      if n_up > n_down else f"The assessment of **{pred_label.upper()}** indicates that financial strengths successfully mitigated several risk factors.")

    st.markdown(f'<div class="bank-card" style="border-left: 12px solid #1e293b;">{synthesis_text} Primary drivers listed above should be prioritized for review.</div>', unsafe_allow_html=True)

    # --- DiCE ROADMAP ---
    st.markdown("---")
    st.subheader("🔁 Counterfactual Roadmap (Path to Approval)")

    if pred_idx in [1, 2]:
        st.write("Suggested profile adjustments to transition to a **GOOD** tier.")
        
        # Logically safe directions as per your existing code
        DECREASE_ONLY = {"Delay_from_due_date", "Num_of_Delayed_Payment", "Outstanding_Debt", "Interest_Rate", "Total_EMI_per_month"}
        INCREASE_ONLY = {"Changed_Credit_Limit", "Credit_Mix"}
        
        original = X_user.iloc[0]
        risk_features = shap_top9[shap_top9["SHAP Value"] > 0]["Feature"].tolist()
        permitted_range = {}
        for f in risk_features:
            if f in DECREASE_ONLY: permitted_range[f] = [0, float(original[f])]
            elif f in INCREASE_ONLY: permitted_range[f] = [float(original[f]), dice_data[f].max()]

        with st.spinner("Generating Strategic Counterfactuals..."):
            dice_df = dice_data.copy()
            dice_df["is_Good"] = (model.predict(dice_df) == 0).astype(int)
            dice_data_if = dice_ml.Data(dataframe=dice_df, continuous_features=feature_names, outcome_name="is_Good")
            dice_model_if = dice_ml.Model(model=model, backend="sklearn")
            dice_obj = Dice(dice_data_if, dice_model_if, method="random")
            
            cf = dice_obj.generate_counterfactuals(X_user, total_CFs=3, desired_class=1, 
                                                features_to_vary=list(permitted_range.keys()), 
                                                permitted_range=permitted_range)

        st.write("#### Comparison of Approval Scenarios")
        st.dataframe(cf.cf_examples_list[0].final_cfs_df.style.highlight_max(axis=0, color="#dcfce7"), use_container_width=True)

        st.markdown("### 🧭 Strategic Action Plan")
        actions = set()
        for _, row in cf.cf_examples_list[0].final_cfs_df.iterrows():
            for f in permitted_range.keys():
                if not np.isclose(original[f], row[f], atol=1e-2):
                    direction = "Reduce" if f in DECREASE_ONLY else "Increase"
                    actions.add(f"<b>{direction}</b> {humanize_feature(f)} from {original[f]:.2f} → <b>{row[f]:.2f}</b>")

        st.markdown(f"""<div class="bank-card" style="border-left: 12px solid #2563eb;">
        To reach the <b>GOOD</b> tier, consider these prioritized actions:
        <ul>{"".join(f"<li>{a}</li>" for a in sorted(actions))}</ul>
        </div>""", unsafe_allow_html=True)
    else:
        st.success("🎉 Applicant already meets high-quality credit standards.")

else:
    st.markdown('<div class="bank-card">👈 Use the sidebar to enter applicant data and click <b>Generate Decision</b> to begin audit.</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("⚠️ Universiti Malaya Master of AI research project. Confidential internal use only.")