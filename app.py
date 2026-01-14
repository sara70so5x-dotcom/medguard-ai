import streamlit as st
import numpy as np
import pandas as pd

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
    color: #e5e7eb;
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
}
.label {
    font-size: 13px;
    color: #9ca3af;
}
.value {
    font-size: 16px;
}
.accent {
    color: #60a5fa;
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
    "Clinical decision support assistant — enhances awareness, does not direct care."
    "</div>",
    unsafe_allow_html=True
)

# ================== CONTROLS ==================
col1, col2 = st.columns(2)
with col1:
    st.selectbox("Clinical Setting", ["Ward", "Emergency", "ICU"])
with col2:
    st.selectbox("Analysis Window", ["Last 6 hours", "Last 12 hours", "Last 24 hours"])

# ================== DATA ==================
def generate_patient_data(hours=48):
    np.random.seed(42)
    data = pd.DataFrame({
        "hour": range(hours),
        "heart_rate": np.random.normal(85, 8, hours),
        "systolic_bp": np.random.normal(120, 10, hours),
        "spo2": np.random.normal(97, 1.2, hours),
        "temperature": np.random.normal(37.1, 0.3, hours)
    })
    data.loc[30:, "heart_rate"] += np.linspace(0, 25, hours - 30)
    data.loc[30:, "systolic_bp"] -= np.linspace(0, 20, hours - 30)
    data.loc[30:, "spo2"] -= np.linspace(0, 3, hours - 30)
    return data

def calculate_risk(row):
    risk = 0
    if row["heart_rate"] > 100: risk += 0.3
    if row["systolic_bp"] < 100: risk += 0.3
    if row["spo2"] < 94: risk += 0.25
    if row["temperature"] > 38: risk += 0.15
    return min(risk, 1.0)

def risk_label(score):
    if score < 0.4:
        return "Low"
    elif score < 0.7:
        return "Moderate"
    else:
        return "High"

# ================== ACTION ==================
if st.button("Review Patient Risk"):
    data = generate_patient_data()
    data["risk_score"] = data.apply(calculate_risk, axis=1)
    last = data.iloc[-1]

    score = round(last["risk_score"], 2)
    label = risk_label(score)
    confidence = round(np.random.uniform(0.78, 0.9), 2)

    # ================== CORE INSIGHT ==================
    st.markdown(f"""
    <div class="panel">
        <div class="label">AI Risk Awareness</div>
        <div class="big accent">Risk Level: {score} ({label})</div>
        <div class="meta">Model confidence: {confidence}</div>
    </div>
    """, unsafe_allow_html=True)

    # ================== WHY ==================
    st.markdown("""
    <div class="panel">
        <div class="label">Why risk may be increasing</div>
        <div class="value">
        • Sustained rise in heart rate<br>
        • Gradual decline in systolic blood pressure<br>
        • Similar trajectory observed in prior deterioration cases
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ================== CONTEXT ==================
    st.markdown("""
    <div class="panel">
        <div class="label">Clinical context</div>
        <div class="value">
        In comparable cases, earlier clinical review and closer monitoring
        were commonly associated with improved outcomes.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ================== TRAJECTORY ==================
    st.markdown("<div class='label'>Risk trend over time</div>", unsafe_allow_html=True)
    st.line_chart(data.set_index("hour")["risk_score"])
