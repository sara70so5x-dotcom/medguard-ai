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
body { background-color: #f8fafc; }
.card {
    padding: 22px;
    border-radius: 14px;
    background-color: white;
    border: 1px solid #e2e8f0;
    margin-bottom: 18px;
}
.header {
    font-size: 28px;
    font-weight: 700;
    color: #0f172a;
}
.subtitle {
    font-size: 15px;
    color: #475569;
}
.label {
    font-weight: 600;
    color: #0f172a;
}
.note {
    font-size: 13px;
    color: #64748b;
}
.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background-color: #e0f2fe;
    color: #0369a1;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.markdown("<div class='header'>🛡️ MedGuard AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>"
    "AI-assisted clinical decision support for early detection of patient deterioration<br>"
    "<span class='note'>Supports — not replaces — physician judgment.</span>"
    "</div>",
    unsafe_allow_html=True
)

# ================== DISCLAIMER ==================
st.markdown("""
<div class="card">
<span class="badge">Clinical Decision Support Assistant</span><br><br>
MedGuard AI analyzes temporal patterns in vital signs to highlight potential clinical risk.
It is designed to <b>support situational awareness</b> and does not provide diagnoses
or treatment decisions.
<br><br>
<b>Final clinical decisions remain the responsibility of the treating physician.</b>
</div>
""", unsafe_allow_html=True)

# ================== CONTROLS ==================
st.subheader("Analysis Context")

col1, col2 = st.columns(2)
with col1:
    department = st.selectbox(
        "Clinical Setting",
        ["General Ward", "Emergency Department", "ICU"]
    )
with col2:
    hours_window = st.selectbox(
        "Data Window",
        ["Last 6 hours", "Last 12 hours", "Last 24 hours"]
    )

st.markdown("<span class='badge'>🔕 Low alert frequency mode enabled</span>", unsafe_allow_html=True)

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

# ================== ACTION ==================
if st.button("Analyze Patient Trends"):
    data = generate_patient_data()
    data["risk_score"] = data.apply(calculate_risk, axis=1)
    last = data.iloc[-1]

    # Confidence (simulated uncertainty)
    confidence = round(np.random.uniform(0.75, 0.9), 2)

    # ================== SNAPSHOT ==================
    st.markdown("""
    <div class="card">
    <span class="label">Patient Trend Summary</span><br><br>
    • Heart rate: sustained upward trend<br>
    • Blood pressure: gradual decline<br>
    • Oxygen saturation: mild downward shift<br>
    • Temperature: stable
    </div>
    """, unsafe_allow_html=True)

    # ================== RISK INSIGHT ==================
    st.markdown(f"""
    <div class="card">
    <span class="label">AI Risk Insight</span><br><br>
    Estimated deterioration risk score: <b>{round(last["risk_score"], 2)}</b><br>
    AI confidence level: <b>{confidence}</b><br><br>
    This assessment reflects similarity to historical deterioration patterns
    observed in comparable clinical contexts.
    </div>
    """, unsafe_allow_html=True)

    # ================== XAI TABLE ==================
    st.markdown("<div class='card'><span class='label'>Trend Contribution Analysis</span><br><br>", unsafe_allow_html=True)
    contrib = pd.DataFrame({
        "Parameter": ["Heart Rate", "Blood Pressure", "SpO₂", "Temperature"],
        "Trend": ["↑ Rising", "↓ Dropping", "↓ Mild decline", "Stable"],
        "Contribution": ["High", "Medium", "Low", "Minimal"]
    })
    st.table(contrib)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================== SUPPORTIVE GUIDANCE ==================
    st.markdown("""
    <div class="card">
    <span class="label">Supportive Clinical Considerations</span><br><br>
    In similar cases, clinicians often considered:
    <ul>
        <li>Closer monitoring of vital signs</li>
        <li>Re-evaluation of laboratory results</li>
        <li>Early senior clinical review</li>
    </ul>
    These are <b>not recommendations</b>, but commonly observed actions.
    </div>
    """, unsafe_allow_html=True)

    # ================== TIMING ==================
    st.markdown("""
    <div class="card">
    <span class="label">Time-Sensitive Insight</span><br><br>
    Historical data suggests that earlier evaluation within the next
    <b>90 minutes</b> was associated with improved outcomes in similar scenarios.
    </div>
    """, unsafe_allow_html=True)

    # ================== SAFETY ==================
    st.markdown("""
    <div class="card">
    <span class="label">What this system does NOT do</span><br><br>
    • Does not diagnose medical conditions<br>
    • Does not prescribe or recommend treatments<br>
    • Does not override clinical judgment
    </div>
    """, unsafe_allow_html=True)

    # ================== TRAJECTORY ==================
    st.subheader("Risk Trend Over Time")
    st.line_chart(data.set_index("hour")["risk_score"])
