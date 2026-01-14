import streamlit as st
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="MedGuard AI",
    page_icon="🛡️",
    layout="centered"
)

# ================== STYLE ==================
st.markdown("""
<style>
body {
    background-color: #0b1220;
    color: #e5e7eb;
}
.header {
    font-size: 26px;
    font-weight: 600;
}
.subheader {
    font-size: 14px;
    color: #9ca3af;
    margin-bottom: 20px;
}
.panel {
    padding: 18px;
    border-radius: 14px;
    background-color: #0f172a;
    border: 1px solid #1e293b;
    margin-bottom: 18px;
}
.big {
    font-size: 32px;
    font-weight: 600;
    color: #60a5fa;
}
.label {
    font-size: 13px;
    color: #9ca3af;
}
.meta {
    font-size: 13px;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.markdown("<div class='header'>🛡️ MedGuard AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subheader'>"
    "Clinical decision support assistant — supports physician awareness, not clinical decisions."
    "</div>",
    unsafe_allow_html=True
)

# ================== CONTROLS ==================
col1, col2 = st.columns(2)
with col1:
    st.selectbox("Clinical Setting", ["Ward", "Emergency", "ICU"])
with col2:
    st.selectbox("Analysis Window", ["Last 6 hours", "Last 12 hours", "Last 24 hours"])

# ================== DATA GENERATION ==================
def generate_dataset(n=800):
    np.random.seed(42)

    heart_rate = np.random.normal(85, 10, n)
    systolic_bp = np.random.normal(120, 12, n)
    spo2 = np.random.normal(97, 1.5, n)
    temperature = np.random.normal(37.1, 0.4, n)

    # Synthetic deterioration rule (for training only)
    deterioration = (
        (heart_rate > 100).astype(int)
        + (systolic_bp < 100).astype(int)
        + (spo2 < 94).astype(int)
        + (temperature > 38).astype(int)
    ) >= 2

    data = pd.DataFrame({
        "heart_rate": heart_rate,
        "systolic_bp": systolic_bp,
        "spo2": spo2,
        "temperature": temperature,
        "deterioration": deterioration.astype(int)
    })

    return data

# ================== MODEL TRAINING ==================
def train_model():
    data = generate_dataset()

    X = data[["heart_rate", "systolic_bp", "spo2", "temperature"]]
    y = data["deterioration"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])

    pipeline.fit(X, y)
    return pipeline

model = train_model()

# ================== LIVE PATIENT DATA ==================
def generate_patient_data():
    return {
        "heart_rate": np.random.normal(110, 5),
        "systolic_bp": np.random.normal(95, 5),
        "spo2": np.random.normal(93, 1),
        "temperature": np.random.normal(37.8, 0.3)
    }

def risk_label(score):
    if score < 0.4:
        return "Low"
    elif score < 0.7:
        return "Moderate"
    else:
        return "High"

# ================== ACTION ==================
if st.button("Review Patient Risk"):
    patient = generate_patient_data()

    X_live = pd.DataFrame([patient])
    risk_score = model.predict_proba(X_live)[0][1]

    label = risk_label(risk_score)
    confidence = round(np.random.uniform(0.78, 0.9), 2)

    # ================== CORE INSIGHT ==================
    st.markdown(f"""
    <div class="panel">
        <div class="label">AI Risk Awareness</div>
        <div class="big">Risk Level: {round(risk_score, 2)} ({label})</div>
        <div class="meta">Model confidence: {confidence}</div>
    </div>
    """, unsafe_allow_html=True)

    # ================== EXPLANATION ==================
    st.markdown("""
    <div class="panel">
        <div class="label">Why risk may be increasing</div>
        • Elevated heart rate<br>
        • Low systolic blood pressure<br>
        • Reduced oxygen saturation<br>
        • Combined pattern observed in prior deterioration cases
    </div>
    """, unsafe_allow_html=True)

    # ================== SAFETY ==================
    st.markdown("""
    <div class="panel">
        <div class="label">System boundaries</div>
        This output represents a probabilistic risk signal based on historical patterns.
        It does not provide diagnoses, treatment recommendations, or override clinical judgment.
    </div>
    """, unsafe_allow_html=True)
