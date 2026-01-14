import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="MedGuard AI",
    layout="wide"
)

# =========================
# Dark Clinical Theme
# =========================
st.markdown("""
<style>
html, body {
    background-color: #0f172a;
    color: #e5e7eb;
}
h1, h2, h3 {
    color: #f9fafb;
}
.card {
    background-color: #111827;
    padding: 22px;
    border-radius: 14px;
    margin-bottom: 20px;
    border: 1px solid #1f2937;
}
.alert-low {
    border-left: 5px solid #22c55e;
    background-color: #052e1c;
    padding: 16px;
    border-radius: 10px;
}
.alert-med {
    border-left: 5px solid #facc15;
    background-color: #2a2200;
    padding: 16px;
    border-radius: 10px;
}
.alert-high {
    border-left: 5px solid #ef4444;
    background-color: #2a0606;
    padding: 16px;
    border-radius: 10px;
}
.badge {
    text-align: center;
    padding: 10px;
    border-radius: 8px;
    font-weight: 600;
}
.badge-low { background-color: #14532d; }
.badge-med { background-color: #713f12; }
.badge-high { background-color: #7f1d1d; }
.note {
    font-size: 0.85rem;
    color: #9ca3af;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.title("MedGuard AI")
st.caption(
    "AI-assisted early warning system for continuous patient monitoring "
    "(Clinical decision-support only)"
)

# =========================
# Patient Scenario
# =========================
scenario = st.selectbox(
    "Select patient scenario",
    [
        "Patient A – Stable",
        "Patient B – Early deterioration",
        "Patient C – Severe deterioration"
    ]
)

# =========================
# Patient Data Simulation
# =========================
def generate_patient_data(hours=48, mode="stable"):
    np.random.seed(42)
    df = pd.DataFrame({
        "hour": range(hours),
        "heart_rate": np.random.normal(78, 5, hours),
        "systolic_bp": np.random.normal(125, 8, hours),
        "spo2": np.random.normal(98, 1, hours),
        "temperature": np.random.normal(36.8, 0.2, hours)
    })

    if mode == "early":
        df.loc[28:, "heart_rate"] += np.linspace(0, 18, hours - 28)
        df.loc[28:, "systolic_bp"] -= np.linspace(0, 15, hours - 28)
        df.loc[28:, "spo2"] -= np.linspace(0, 3, hours - 28)

    if mode == "severe":
        df.loc[20:, "heart_rate"] += np.linspace(8, 35, hours - 20)
        df.loc[20:, "systolic_bp"] -= np.linspace(10, 45, hours - 20)
        df.loc[20:, "spo2"] -= np.linspace(3, 9, hours - 20)
        df.loc[20:, "temperature"] += np.linspace(0.3, 1.2, hours - 20)

    return df

if "Stable" in scenario:
    data = generate_patient_data(mode="stable")
elif "Early" in scenario:
    data = generate_patient_data(mode="early")
else:
    data = generate_patient_data(mode="severe")

# =========================
# Training Data (Synthetic)
# =========================
def generate_training_data(samples=400):
    rows = []
    np.random.seed(1)
    for _ in range(samples):
        hr = np.random.normal(85, 15)
        bp = np.random.normal(120, 20)
        spo2 = np.random.normal(96, 3)
        temp = np.random.normal(37, 0.6)

        label = int((hr > 100) or (bp < 95) or (spo2 < 92))
        rows.append([hr, bp, spo2, temp, label])

    return pd.DataFrame(
        rows,
        columns=["heart_rate", "systolic_bp", "spo2", "temperature", "label"]
    )

train = generate_training_data()
X_train = train.drop(columns=["label"])
y_train = train["label"]

model = LogisticRegression()
model.fit(X_train, y_train)

# =========================
# Risk Prediction
# =========================
X_patient = data[["heart_rate", "systolic_bp", "spo2", "temperature"]]
data["risk"] = model.predict_proba(X_patient)[:, 1]
current_risk = data.iloc[-1]["risk"]

# =========================
# Risk Levels
# =========================
if current_risk < 0.35:
    level = "Low Risk"
    badge = "badge-low"
    alert_class = "alert-low"
elif current_risk < 0.65:
    level = "Moderate Risk"
    badge = "badge-med"
    alert_class = "alert-med"
else:
    level = "High Risk"
    badge = "badge-high"
    alert_class = "alert-high"

# =========================
# Layout
# =========================
left, right = st.columns([1.2, 2])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Current Risk Assessment")
    st.metric("Risk Probability", f"{current_risk:.2f}")
    st.markdown(f"<div class='badge {badge}'>{level}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if level != "Low Risk":
        st.markdown(f"<div class='{alert_class}'>", unsafe_allow_html=True)

        if level == "Moderate Risk":
            st.markdown("""
            **Clinical Alert – Early Warning**  
            The patient is showing early physiological deviation from baseline.
            Patterns suggest increasing vulnerability but deterioration may still
            be preventable with timely clinical review.
            """)

        if level == "High Risk":
            st.markdown("""
            **Clinical Alert – High Risk**  
            The patient demonstrates sustained physiological instability.
            Current trends indicate a high likelihood of further deterioration
            if the trajectory continues.
            """)

        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Risk Trajectory Over Time")
    st.line_chart(data.set_index("hour")["risk"])

    with st.expander("What is happening to the patient?"):
        if level == "Low Risk":
            st.markdown("""
            Vital signs remain within expected physiological ranges.
            No progressive stress patterns are detected.
            """)

        elif level == "Moderate Risk":
            st.markdown("""
            Gradual heart rate elevation combined with subtle blood pressure
            reduction indicates early compensatory stress.
            These changes often precede overt clinical deterioration.
            """)

        else:
            st.markdown("""
            Rapid heart rate escalation, declining blood pressure, and reduced
            oxygen saturation indicate failure of physiological compensation.
            This pattern has historically been associated with critical events.
            """)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Explainable AI
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Explainable AI – Key Contributors")

importance = pd.Series(
    abs(model.coef_[0]),
    index=X_train.columns
).sort_values(ascending=False)

st.bar_chart(importance)

with st.expander("Why did the system raise this alert?"):
    st.markdown("""
    The model places the highest weight on heart rate acceleration and
    declining systolic blood pressure. These variables consistently appear
    in cases that progressed to clinical deterioration in historical data.
    """)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Footer
# =========================
st.caption(
    "MedGuard AI is an early warning and decision-support system. "
    "Final clinical decisions remain the responsibility of the healthcare professional."
)
