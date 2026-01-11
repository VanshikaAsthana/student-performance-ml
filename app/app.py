import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Load models and scalers
# -----------------------------
model_early = joblib.load("../model_early.pkl")
scaler_early = joblib.load("../scaler_early.pkl")

model_mid = joblib.load("../model_mid.pkl")
scaler_mid = joblib.load("../scaler_mid.pkl")

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Academic Risk Predictor", layout="centered")

st.title("🎓 Academic Risk Assessment System")
st.write(
    "This tool provides an **early-warning academic risk score** to help identify "
    "students who may need additional support."
)

# -----------------------------
# Prediction stage selector
# -----------------------------
stage = st.radio(
    "Select Prediction Stage",
    [
        "Early Semester (behavioral factors only)",
        "Mid Semester (includes internal assessment)"
    ]
)

st.markdown("---")

# -----------------------------
# Input fields
# -----------------------------
studytime = st.selectbox("Study Time (1 = very low, 4 = very high)", [1, 2, 3, 4])
failures = st.number_input("Number of Past Failures", 0, 4, 0)
absences = st.number_input("Number of Absences", 0, 100, 0)

schoolsup = st.selectbox("School Support", ["yes", "no"])
famsup = st.selectbox("Family Support", ["yes", "no"])
paid = st.selectbox("Paid Classes", ["yes", "no"])
activities = st.selectbox("Extracurricular Activities", ["yes", "no"])
internet = st.selectbox("Internet Access at Home", ["yes", "no"])
romantic = st.selectbox("Romantic Relationship", ["yes", "no"])

# Mid-semester academic signal
if stage == "Mid Semester (includes internal assessment)":
    G1 = st.number_input("Mid-Semester Score (G1)", 0, 20, 10)

# -----------------------------
# Helper function
# -----------------------------
def encode(val):
    return 1 if val == "yes" else 0

# -----------------------------
# Prediction
# -----------------------------
if st.button("Assess Academic Risk"):
    base_features = [
        studytime,
        failures,
        absences,
        encode(schoolsup),
        encode(famsup),
        encode(paid),
        encode(activities),
        encode(internet),
        encode(romantic),
    ]

    if stage == "Early Semester (behavioral factors only)":
        features = np.array([base_features])
        features_scaled = scaler_early.transform(features)
        model_used = model_early
    else:
        features = np.array([base_features + [G1]])
        features_scaled = scaler_mid.transform(features)
        model_used = model_mid

    # Probability of PASS
    prob_pass = model_used.predict_proba(features_scaled)[0][1]

    # Risk = probability of FAIL
    risk_score = 1 - prob_pass

    # Risk level
    if risk_score < 0.4:
        risk_level = "Low"
        color = "green"
    elif risk_score < 0.7:
        risk_level = "Moderate"
        color = "orange"
    else:
        risk_level = "High"
        color = "red"

    # -----------------------------
    # Output
    # -----------------------------
    st.markdown("---")
    st.markdown(
        f"""
        ### 🎯 Risk Assessment Result
        - **Risk Level:** <span style="color:{color}; font-weight:bold">{risk_level}</span>
        - **Risk Score:** {risk_score * 100:.1f}%

        📌 *This is a {stage.lower()} prediction.  
        The score represents an **early-warning estimate**, not a final academic outcome.*
        """,
        unsafe_allow_html=True
    )
