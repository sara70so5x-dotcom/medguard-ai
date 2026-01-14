import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# =========================
# Page Config
# =========================
st.set_page_config(page_title="MedGuard AI", layout="wide")

# =========================
# Clinical Style
# =========================
st.markdown("""
<style>
html, body {
    background-color: #10161c;
    color: #e2e8f0;
}
h1, h2, h3 {
    color: #f8fafc;
}
.card {
    background-color: #1b2430;
    padding: 22px;
    border-radius: 14px;
    margin-bottom: 20px;
    border: 1px solid #273142;
}
.badge-low {
    background-color: #1f7a55;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
    font-weight: 600;
}
.badge-med {
    background-color: #8f6b1b;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
    font-weight: 600;
}
.badge-high {
    background-color: #374151;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
    font-weight: 600;
}
.alert {
    background-color: #111827;
    border-left: 4px solid #fbbf24;
    padding: 14px;
    border-radius: 10px;
}
.note {
    font-size: 0.85rem;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.title("MedGuard AI")
st.caption("Continuous patient monitoring • AI-assisted early clinical alerts")

# =========================
# Patient Scenario Selector
# =========================
scenario = st.selectbox(
    "Select patient condition",
    [
        "Patient 1 – Stable",
        "Patient 2 – Early deterioration",
        "Patient 3 – Severe deterioration"
    ]
)

# =========================
# Simulated Patient Data
# =========================
def generate_patient_data(hours=48, level="stable"):
    np.random.seed(42)
    df = pd.DataFrame({
        "hour": range(hours),
        "heart_rate": np.random.normal(78, 5, hours),
        "systolic_bp": np.random.normal(125, 8, hours),
        "spo2": np.random.normal(98, 1, hours),
        "temperature": np.random.normal(36.8, 0.2, hours)
    })

    if level == "early":
        df.loc[28:, "heart_rate"] += np.linspace(0, 18, hours - 28)
        df.loc[28:, "systolic_bp"] -= np.linspace(0, 18, hours - 28)
        df.loc[28:, "spo2"] -= np.linspace(0, 3, hours - 28)

    if level == "severe":
        df.loc[20:, "heart_rate"] += np.linspace(5, 35, hours - 20)
        df.loc[20:, "systolic_bp"] -= np.linspace(5, 45, hours - 20)
        df.loc[20:, "spo2"] -= np.linspace(2, 8, hours - 20)
        df.loc[20:, "temperature"] += np.linspace(0.2, 1.2, hours - 20)

    return df

if "Stable" in scenario:
    data = generate_patient_data(level="stable")
elif "Early" in scenario:
    data = generate_patient_data(level="early")
else:
    data = generate_patient_data(level="severe")

# =========================
# Global Training Dataset
# =========================
def generate_training_data(samples=400):
    np.random.seed(1)
    rows = []
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

train_df = generate_training_data()
X_train = train_df[["heart_rate", "systolic_bp", "spo2", "temperature"]]
y_train = train_df["label"]

model = LogisticRegression()
model.fit(X_train, y_train)

# =========================
# Prediction on Patient
# =========================
X_patient = data[["heart_rate", "systolic_bp", "spo2", "temperature"]]
data["risk"] = model.predict_proba(X_patient)[:, 1]
current_risk = data.iloc[-1]["risk"]

# =========================
# Risk Level
# =========================
if current_risk < 0.35:
    level = "Low Risk"
    badge = "badge-low"
elif current_risk < 0.65:
    level = "Moderate Risk"
    badge = "badge-med"
else:
    level = "High Risk"
    badge = "badge-high"

# =========================
# Layout
# =========================
col1, col2 = st.columns([1.2, 2])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Risk Snapshot")
    st.metric("Current Risk Probability", f"{current_risk:.2f}")
    st.markdown(f"<div class='{badge}'>{level}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if level != "Low Risk":
        st.markdown("<div class='alert'>", unsafe_allow_html=True)
        st.markdown("""
        **Clinical Alert**  
        Patient trajectory indicates ongoing physiological deterioration.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Risk Trend")
    st.line_chart(data.set_index("hour")["risk"])

    with st.expander("What is happening to the patient?"):
        if level == "Low Risk":
            st.markdown("""
            The patient’s vital signs remain stable and within expected ranges.
            There is no evidence of physiological stress or instability at this time.
            """)
        elif level == "Moderate Risk":
            st.markdown("""
            The patient is showing **early signs of physiological stress**.
            Gradual increases in heart rate and mild blood pressure reduction
            suggest the beginning of clinical deterioration.
            """)
        else:
            st.markdown("""
            The patient is experiencing **significant physiological instability**.
            Rapid heart rate escalation, declining blood pressure, and reduced
            oxygen saturation indicate a high likelihood of imminent deterioration
            if the current trajectory continues.
            """)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Explainable AI
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Explainable AI")

coeff = pd.Series(model.coef_[0], index=X_train.columns)
coeff = coeff.abs().sort_values(ascending=False)
st.bar_chart(coeff)

with st.expander("Why did the model raise this risk?"):
    st.markdown("""
    The model identified heart rate acceleration and declining systolic
    blood pressure as the strongest contributors to the current risk signal.
    """)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Footer
# =========================
st.caption(
    "MedGuard AI is a clinical decision-support assistant and does not replace medical judgment."
)
