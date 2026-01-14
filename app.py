import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# =================================
# Page Config (App-like)
# =================================
st.set_page_config(
    page_title="MedGuard AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =================================
# Hide Streamlit Branding
# =================================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

html, body {
    background-color: #0b1220;
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
    border-left: 6px solid #22c55e;
    background-color: #052e1c;
    padding: 18px;
    border-radius: 10px;
}
.alert-med {
    border-left: 6px solid #f59e0b;
    background-color: #2a1f05;
    padding: 18px;
    border-radius: 10px;
}
.alert-high {
    border-left: 6px solid #ef4444;
    background-color: #2a0606;
    padding: 18px;
    border-radius: 10px;
}
.badge {
    text-align: center;
    padding: 10px;
    border-radius: 8px;
    font-weight: 600;
    margin-top: 10px;
}
.badge-low { background-color: #14532d; }
.badge-med { background-color: #78350f; }
.badge-high { background-color: #7f1d1d; }
.note {
    font-size: 0.85rem;
    color: #9ca3af;
}
</style>
""", unsafe_allow_html=True)

# =================================
# App Header
# =================================
st.markdown("""
<h1 style="margin-bottom:0;">MedGuard AI</h1>
<p class="note">
AI-assisted early warning system for continuous patient monitoring
</p>
""", unsafe_allow_html=True)

# =================================
# Scenario Selector
# =================================
scenario = st.selectbox(
    "Patient Scenario",
    [
        "Stable Patient",
        "Early Deterioration",
        "Critical Condition"
    ]
)

# =================================
# Patient Data Generator
# =================================
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

if scenario == "Stable Patient":
    data = generate_patient_data(mode="stable")
elif scenario == "Early Deterioration":
    data = generate_patient_data(mode="early")
else:
    data = generate_patient_data(mode="severe")

# =================================
# Global Training Data
# =================================
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

# =================================
# Risk Prediction
# =================================
X_patient = data[["heart_rate", "systolic_bp", "spo2", "temperature"]]
data["risk"] = model.predict_proba(X_patient)[:, 1]
current_risk = data.iloc[-1]["risk"]

# =================================
# Risk Levels
# =================================
if current_risk < 0.35:
    level = "Stable"
    badge = "badge-low"
    alert_class = "alert-low"
elif current_risk < 0.7:
    level = "Early Deterioration"
    badge = "badge-med"
    alert_class = "alert-med"
else:
    level = "Critical Condition"
    badge = "badge-high"
    alert_class = "alert-high"

# =================================
# Layout
# =================================
left, right = st.columns([1.2, 2])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Current Status")
    st.metric("Risk Probability", f"{current_risk:.2f}")
    st.markdown(f"<div class='badge {badge}'>{level}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if level != "Stable":
        st.markdown(f"<div class='{alert_class}'>", unsafe_allow_html=True)
        st.markdown(f"""
**{level} Alert**  
Patient trajectory indicates physiological instability.
""")
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Risk Timeline")
    st.line_chart(data.set_index("hour")["risk"])

    with st.expander("What is happening to the patient?"):
        if level == "Stable":
            st.markdown("""
Vital signs remain within expected physiological ranges.
No progressive stress patterns are currently observed.
""")
        elif level == "Early Deterioration":
            st.markdown("""
Gradual heart rate elevation and subtle blood pressure decline
suggest early physiological stress and increasing vulnerability.
""")
        else:
            st.markdown("""
Rapid heart rate escalation, hypotension, oxygen desaturation,
and rising temperature indicate failure of physiological compensation.
""")

    st.markdown("</div>", unsafe_allow_html=True)
