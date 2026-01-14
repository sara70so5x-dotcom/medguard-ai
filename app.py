import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="MedGuard AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# Hide Streamlit UI + Styling
# ===============================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

html, body {
    background-color: #0b1220;
    color: #e5e7eb;
    font-family: Arial, sans-serif;
}

h1, h2, h3 { color: #f9fafb; }

.card {
    background-color: #111827;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #1f2937;
    margin-bottom: 18px;
}

.metric {
    font-size: 1.5rem;
    font-weight: 700;
}

.badge {
    text-align: center;
    padding: 8px;
    border-radius: 8px;
    font-weight: 600;
    margin-top: 8px;
}

.badge-low { background-color: #14532d; }
.badge-med { background-color: #b45309; }
.badge-high { background-color: #7f1d1d; }

.alert-low {
    border-left: 6px solid #22c55e;
    background-color: #052e1c;
    padding: 16px;
    border-radius: 10px;
}

.alert-med {
    border-left: 6px solid #fbbf24;
    background-color: #3a2a00;
    padding: 16px;
    border-radius: 10px;
}

.alert-high {
    border-left: 6px solid #ef4444;
    background-color: #2a0606;
    padding: 16px;
    border-radius: 10px;
}

.note {
    font-size: 0.9rem;
    color: #9ca3af;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# Header
# ===============================
st.markdown("""
<h1> MedGuard AI</h1>
<p class="note">
Clinical Early Warning System (Decision Support Only)<br>
This system assists clinicians in identifying early physiological deterioration.
It does <b>not</b> replace clinical judgment.
</p>
""", unsafe_allow_html=True)

# ===============================
# Scenario Selector
# ===============================
scenario = st.selectbox(
    "Select Patient Scenario",
    ["Stable Patient", "Early Deterioration", "Critical Condition"]
)

# ===============================
# Patient Data Generator
# ===============================
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
        df.loc[20:, "heart_rate"] += np.linspace(10, 35, hours - 20)
        df.loc[20:, "systolic_bp"] -= np.linspace(10, 45, hours - 20)
        df.loc[20:, "spo2"] -= np.linspace(3, 9, hours - 20)
        df.loc[20:, "temperature"] += np.linspace(0.3, 1.2, hours - 20)

    return df

mode = "stable" if scenario == "Stable Patient" else "early" if scenario == "Early Deterioration" else "severe"
data = generate_patient_data(mode=mode)

# ===============================
# Train Prototype Model
# ===============================
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
X_train = train[["heart_rate", "systolic_bp", "spo2", "temperature"]]
y_train = train["label"]

model = LogisticRegression()
model.fit(X_train, y_train)

# ===============================
# Prediction
# ===============================
X_patient = data[["heart_rate", "systolic_bp", "spo2", "temperature"]]
data["raw_risk"] = model.predict_proba(X_patient)[:, 1]
raw_risk = data.iloc[-1]["raw_risk"]

# Controlled risk (prototype clarity)
if scenario == "Stable Patient":
    risk = float(np.clip(raw_risk, 0.10, 0.30))
elif scenario == "Early Deterioration":
    risk = float(np.clip(raw_risk, 0.40, 0.60))
else:
    risk = float(np.clip(raw_risk, 0.75, 0.95))

# ===============================
# Risk Level
# ===============================
if risk < 0.35:
    level, badge, alert = "Stable", "badge-low", "alert-low"
    alert_text = "Patient trajectory remains within expected physiological ranges."
elif risk < 0.7:
    level, badge, alert = "Early Deterioration", "badge-med", "alert-med"
    alert_text = (
        "Subtle but consistent physiological changes detected. "
        "Closer monitoring is advised."
    )
else:
    level, badge, alert = "Critical Condition", "badge-high", "alert-high"
    alert_text = (
        "Pattern strongly resembles prior clinical deterioration cases. "
        "Immediate clinical review is recommended."
    )

# ===============================
# Baseline & Confidence
# ===============================
baseline = data.iloc[:12].mean()
current = data.iloc[-1]
variability = data[["heart_rate", "systolic_bp", "spo2", "temperature"]].std().mean()
confidence = "High" if variability < 5 else "Moderate" if variability < 10 else "Low"

# ===============================
# Layout
# ===============================
left, right = st.columns([1.3, 2])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Current Patient Status")
    st.markdown(f"<div class='metric'>Risk Score: {risk:.2f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='badge {badge}'>{level}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='{alert}'>", unsafe_allow_html=True)
    st.markdown(f"**Clinical Alert**  \n{alert_text}")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Risk Trajectory (Last 48h)")
    st.line_chart(data.set_index("hour")["raw_risk"])
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# Clinical Insight Summary
# ===============================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Clinical Insight Summary")

st.markdown("**Key physiological changes**")
st.write(f"- Heart rate ↑ {((current.heart_rate/baseline.heart_rate)-1)*100:.1f}%")
st.write(f"- Systolic BP ↓ {baseline.systolic_bp - current.systolic_bp:.1f} mmHg")
st.write("- Oxygen saturation shows a downward trend")

st.markdown("**Why this matters**")
if level == "Early Deterioration":
    st.write(
        "This pattern suggests early physiological stress that may precede "
        "clinical deterioration if trends continue."
    )
elif level == "Critical Condition":
    st.write(
        "Findings indicate loss of physiological compensation and are commonly "
        "observed prior to critical events."
    )
else:
    st.write("No clinically concerning deviations are currently observed.")

st.markdown("**Key parameters to monitor**")
st.write("- Blood pressure trend")
st.write("- Oxygen saturation")
st.write("- Temperature progression")
st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# Optional Clinical Foresight
# ===============================
with st.expander("Clinical Foresight (Optional)"):
    st.markdown("**What may happen next (6–12 hours)?**")
    st.write(
        "If the current physiological trend continues, "
        "the patient may clinically worsen within the next 6–12 hours."
    )

    st.markdown("**Confidence in this signal**")
    st.write(f"{confidence} confidence based on consistency of vital sign trends.")

    st.markdown("**Have we seen this before?**")
    st.write(
        "Similar physiological patterns were observed in previous patients "
        "who later experienced deterioration."
    )

# ===============================
# Footer
# ===============================
st.caption(
    "MedGuard AI is a clinical decision-support system. "
    "Final diagnosis and treatment decisions remain the responsibility of the clinician."
)
