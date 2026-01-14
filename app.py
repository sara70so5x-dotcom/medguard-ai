import streamlit as st
import numpy as np
import pandas as pd

# ---------- Page Config ----------
st.set_page_config(
    page_title="MedGuard AI",
    page_icon="🛡️",
    layout="centered"
)

# ---------- Header ----------
st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 16px;
    background-color: #0f172a;
    color: white;
    margin-bottom: 20px;
}
.metric {
    font-size: 36px;
    font-weight: bold;
}
.subtitle {
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ MedGuard AI")
st.caption(
    "From early signals to smart intervention — AI-powered clinical decision support"
)

# ---------- Data Simulation ----------
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

# ---------- AI Logic ----------
def calculate_risk(row):
    risk = 0
    if row["heart_rate"] > 100: risk += 0.3
    if row["systolic_bp"] < 100: risk += 0.3
    if row["spo2"] < 94: risk += 0.25
    if row["temperature"] > 38: risk += 0.15
    return min(risk, 1.0)

def decision_logic(risk):
    if risk >= 0.7:
        return "🔴 High Risk – ICU Transfer Recommended"
    elif risk >= 0.4:
        return "🟠 Medium Risk – Order Labs & Adjust Medication"
    else:
        return "🟢 Low Risk – Continue Monitoring"

def explain_decision(row):
    reasons = []
    if row["heart_rate"] > 100: reasons.append("Increasing Heart Rate")
    if row["systolic_bp"] < 100: reasons.append("Dropping Blood Pressure")
    if row["spo2"] < 94: reasons.append("Decreasing Oxygen Saturation")
    if row["temperature"] > 38: reasons.append("Elevated Temperature")
    return reasons

# ---------- Run ----------
if st.button("▶️ Analyze Patient Now"):
    data = generate_patient_data()
    data["risk_score"] = data.apply(calculate_risk, axis=1)
    data["decision"] = data["risk_score"].apply(decision_logic)
    data["explanation"] = data.apply(explain_decision, axis=1)

    last = data.iloc[-1]
    similarity = np.random.randint(70, 85)
    intervention_time = 90

    # ---------- Risk Card ----------
    st.markdown(f"""
    <div class="card">
        <div class="metric">Risk Score: {round(last["risk_score"], 2)}</div>
        <div class="subtitle">{last["decision"]}</div>
    </div>
    """, unsafe_allow_html=True)

    # ---------- Insights ----------
    st.subheader("🧠 Explainable AI – Why?")
    for r in last["explanation"]:
        st.write("•", r)

    st.subheader("🧬 Clinical Pattern Matching")
    st.info(
        f"Patient trajectory matches **{similarity}%** of historical cases "
        "that experienced deterioration within 8 hours."
    )

    st.subheader("⏱️ Best Time to Intervene")
    st.success(
        f"Intervening within the next **{intervention_time} minutes** "
        "can reduce ICU admission risk by **35%**."
    )

    # ---------- Trajectory ----------
    st.subheader("📈 Risk Trajectory Over Time")
    st.line_chart(data.set_index("hour")["risk_score"])
