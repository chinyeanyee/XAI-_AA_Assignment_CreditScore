import pickle

X_shap_bg = pickle.load(open("shap_background.pkl", "rb"))

print("Type:", type(X_shap_bg))

try:
    print("Shape:", X_shap_bg.shape)
except Exception as e:
    print("No shape attribute:", e)
