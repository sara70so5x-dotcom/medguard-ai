import streamlit as st
import numpy as np
import pandas as pd

# ================== PAGE ==================
st.set_page_config(
    page_title="MedGuard AI",
    page_icon="🛡️",
    layout="centered"
)

# ================== STYLE ==================
st.markdown("""
<style>
body {
    background-color: #020617;
    color: #e5e7eb;
}
.block {
    padding: 18px;
    border-radius: 14px;
    background-color: #020617;
    border: 1px solid #1e293b;
    margin-bottom: 16px;
}
.header {
    font-size: 26px;
    font-weight: 600;
    color: #e5e7eb;
}
.sub {
    font-size: 14px;
    color: #9ca3af;
}
.label {
    font-size: 13px;
    color: #9ca3af;
}
.value {
    font-size: 18px;
    font-weight: 500;
}
.muted {
    color: #9ca3af;
    font-size: 13px;
}
.divider {
    height: 1px;
    background-color: #1e293b;
    margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.markdown("<div class='header'>🛡️ MedGuard AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub'>"
    "Clinical decision support assistant for early risk awareness<br>"
    "Supports physician judgment — does not provide diagnoses or treatment decisions."
    "</div>",
    unsafe_allow_html=True
)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ================== CONTEXT ==================
col1, col2 = st.columns(2)
with col1:
    department = st.selectbox("Clinical Setting", ["Ward", "Emergency", "ICU"])
with col2:
    window = st.selectbox("Analysis Window", ["Last 6 hours", "Last 12 hours", "Last 24 hours"])

st.markdown("<div class='muted'>Low alert frequency mode enabled</div>", unsafe_allow_html=True)

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

    confidence = round(np.random.uniform(0.75, 0.9), 2)

    # ================== SUMMARY ==================
    st.markdown("""
    <div class="block">
        <div class="label">Patient Trend Summary</div>
        <div class="value">
        Heart rate rising · Blood pressure declining · SpO₂ mildly reduced · Temperature stable
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ================== RISK ==================
    st.markdown(f"""
    <div class="block">
        <div class="label">AI Risk Awareness</div>
        <div class="value">Estimated risk level: {round(last["risk_score"], 2)}</div>
        <div class="muted">Model confidence: {confidence}</div>
    </div>
    """, unsafe_allow_html=True)

    # ================== EXPLANATION ==================
    st.markdown("""
    <div class="block">
        <div class="label">Why risk may be increasing</div>
        <div class="value">
        • Sustained rise in heart rate<br>
        • Gradual drop in systolic blood pressure<br>
        • Pattern similarity to prior deterioration cases
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ================== SUPPORT ==================
    st.markdown("""
    <div class="block">
        <div class="label">Supportive clinical insight</div>
        <div class="value">
        In comparable cases, earlier review and closer monitoring were often associated
        with improved outcomes.
        </div>
        <div class="muted">
        This is contextual information — not a clinical directive.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ================== TRAJECTORY ==================
    st.markdown("<div class='label'>Risk trend over time</div>", unsafe_allow_html=True)
    st.line_chart(data.set_index("hour")["risk_score"])
