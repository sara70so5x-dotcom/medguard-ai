import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="MedGuard AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# Global Styling (Clean / Clinical)
# ===============================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

html, body {
    background-color: #0b1220;
    color: #e5e7eb;
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
}

h1 {
    font-size: 2.1rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

h2, h3 {
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.note {
    font-size: 0.9rem;
    color: #9ca3af;
    margin-bottom: 1.5rem;
}

.card {
    background: linear-gradient(180deg, #111827, #0f172a);
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #1f2937;
    margin-bottom: 20px;
}

.metric {
    font-size: 1.6rem;
    font-weight: 700;
}

.badge {
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    width: fit-content;
    margin-top: 10px;
}

.badge-low { background-color: #166534; }
.badge-med { background-color: #b45309; }
.badge-high { background-color: #7f1d1d; }

.alert {
    padding: 16px;
    border-radius: 12px;
    margin-top: 14px;
    line-height: 1.5;
}

.alert-low {
    background-color: #052e1c;
    border-left: 5px solid #22c55e;
}

.alert-med {
    background-color: #3a2a00;
    border-left: 5px solid #fbbf24;
}

.alert-high {
    background-color: #2a0606;
    border-left: 5px solid #ef4444;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# Header
# ===============================
st.markdown("""
<h1>MedGuard AI</h1>
<p class="note">
Clinical Early Warning System (Decision Support Only)<br>
Supports clinicians in identifying early physiological deterioration. Final decisions remain with the clinician.
</p>
""", unsafe_allow_html=True)

# ===============================
# Scenario Selection
# ===============================
scenario = st.selectbox(
    "Patient Scenario",
    ["Stable Patient", "Early Deterioration", "Critical Condition"]
)

# ===============================
# Patient Data Generator
# ===============================
def generate_patient_data(hours=48, mode="stable"):
    np.random.seed(42)
    data = pd.DataFrame({
        "hour": range(hours),
        "heart_rate": np.random.normal(78, 5, hours),
        "systolic_bp": np.random.normal(125, 8, hours),
        "spo2": np.random.normal(98, 1, hours),
        "temperature": np.random.normal(36.8, 0.2, hours)
    })

    if mode == "early":
        data.loc[28:, "heart_rate"] += np.linspace(0, 18, hours - 28)
        data.loc[28:, "systolic_bp"] -= np.linspace(0, 15, hours - 28)
        data.loc[28:, "spo2"] -= np.linspace(0, 3, hours - 28)

    if mode == "severe":
        data.loc[20:, "heart_rate"] += np.linspace(10, 35, hours - 20)
        data.loc[20:, "systolic_bp"] -= np.linspace(10, 45, hours - 20)
        data.loc[20:, "spo2"] -= np.linspace(3, 9, hours - 20)
        data.loc[20:, "temperature"] += np.linspace(0.3, 1.2, hours - 20)

    return data

mode = (
    "stable" if scenario == "Stable Patient"
    else "early" if scenario == "Early Deterioration"
    else "severe"
)

data = generate_patient_data(mode=mode)

# ===============================
# Prototype ML Model
# ===============================
train = pd.DataFrame({
    "heart_rate": np.random.normal(85, 15, 400),
    "systolic_bp": np.random.normal(120, 20, 400),
    "spo2": np.random.normal(96, 3, 400),
    "temperature": np.random.normal(37, 0.6, 400)
})

train["label"] = (
    (train.heart_rate > 100) |
    (train.systolic_bp < 95) |
    (train.spo2 < 92)
).astype(int)

model = LogisticRegression()
model.fit(
    train[["heart_rate", "systolic_bp", "spo2", "temperature"]],
    train["label"]
)

data["risk_trend"] = model.predict_proba(
    data[["heart_rate", "systolic_bp", "spo2", "temperature"]]
)[:, 1]

# Scenario-based final risk
risk_score = (
    0.25 if scenario == "Stable Patient"
    else 0.55 if scenario == "Early Deterioration"
    else 0.85
)

# ===============================
# Risk Interpretation
# ===============================
if risk_score < 0.35:
    level = "Stable"
    badge = "badge-low"
    alert_class = "alert-low"
    alert_text = (
        "Physiological parameters remain within expected ranges. "
        "No significant deviation from baseline trends is observed at this time."
    )

elif risk_score < 0.7:
    level = "Early Deterioration"
    badge = "badge-med"
    alert_class = "alert-med"
    alert_text = (
        "Gradual changes in vital signs indicate early physiological stress. "
        "The pattern suggests emerging instability that may progress if trends persist."
    )

else:
    level = "Critical Condition"
    badge = "badge-high"
    alert_class = "alert-high"
    alert_text = (
        "Sustained deviations from baseline indicate significant physiological deterioration. "
        "The observed pattern is consistent with high-risk trajectories seen in prior cases."
    )

# ===============================
# Layout
# ===============================
left, right = st.columns([1.2, 2])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Current Risk Assessment")
    st.markdown(f"<div class='metric'>Risk Score: {risk_score:.2f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='badge {badge}'>{level}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='alert {alert_class}'>{alert_text}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Risk Trajectory Over Time")
    st.line_chart(data.set_index("hour")["risk_trend"])
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# Clinical Insight Summary
# ===============================
baseline = data.iloc[:12].mean()
current = data.iloc[-1]

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Clinical Interpretation")

st.write(
    "Recent trends show measurable deviations from the patient's baseline physiology. "
    "These changes are driven primarily by shifts in cardiovascular stability and oxygenation, "
    "which together influence the overall risk trajectory."
)

st.write(
    f"Compared to baseline, heart rate has increased by "
    f"{((current.heart_rate / baseline.heart_rate) - 1) * 100:.1f}%, "
    f"while systolic blood pressure has decreased by "
    f"{baseline.systolic_bp - current.systolic_bp:.1f} mmHg."
)

st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# Optional Foresight
# ===============================
with st.expander("Clinical Foresight (Optional)"):
    st.write(
        "If current trends continue, the patient may transition to a higher severity state "
        "within the next 6–12 hours. Continued monitoring and clinical correlation are advised."
    )

# ===============================
# Footer
# ===============================
st.caption(
    "MedGuard AI is a clinical decision-support prototype and does not provide diagnoses or treatment recommendations."
)
