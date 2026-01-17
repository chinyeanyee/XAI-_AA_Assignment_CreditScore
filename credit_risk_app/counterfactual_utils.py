# counterfactual_utils.py
import numpy as np

# Feature → advice mapping
FEATURE_ADVICE = {
    "Interest_Rate": "Try to reduce your interest rate (e.g., refinancing or negotiating with lenders).",
    "Num_of_Delayed_Payment": "Reducing late payments can significantly improve your credit score.",
    "Delay_from_due_date": "Paying bills earlier and on time will positively affect your credit profile.",
    "Outstanding_Debt": "Lowering your outstanding debt can improve creditworthiness.",
    "Total_EMI_per_month": "Reducing monthly EMI commitments may help improve your score.",
    "Num_Bank_Accounts": "Maintaining fewer, well-managed bank accounts may be beneficial.",
    "Num_Credit_Card": "Avoid opening too many credit cards in a short period.",
    "Changed_Credit_Limit": "Avoid frequent changes to your credit limit.",
    "Credit_Mix": "Maintaining a healthy mix of credit types improves long-term stability."
}


def generate_counterfactual_advice(
    shap_values_row,
    feature_names,
    top_k=3
):
    """
    Generate counterfactual advice for moving
    from Poor → Good based on SHAP values.

    Parameters
    ----------
    shap_values_row : array-like (n_features,)
        SHAP values for ONE instance and ONE class (Poor)
    feature_names : list
        Feature names in correct order
    top_k : int
        Number of recommendations

    Returns
    -------
    List[str]
        Human-readable advice
    """

    shap_values = np.array(shap_values_row)

    # Focus on features pushing prediction toward Poor (positive SHAP)
    harmful_idx = np.argsort(-shap_values)

    advice = []
    for idx in harmful_idx:
        if shap_values[idx] <= 0:
            continue  # skip features not hurting the score

        feature = feature_names[idx]
        if feature in FEATURE_ADVICE:
            advice.append(FEATURE_ADVICE[feature])

        if len(advice) == top_k:
            break

    return advice
