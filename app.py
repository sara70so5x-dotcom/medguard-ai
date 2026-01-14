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
}
h1, h2, h3 {
    color: #f9fafb;
}

.card {
    background-color: #111827;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #1f2937;
    margin-bottom: 18px;
}

.metric {
    font-size: 1.4rem;
    font-weight: 600;
}

.badge {
    text-align: center;
    padding: 8px;
    border-radius: 8px;
    font-weight: 600;
    margin-top: 8px;
}

.badge-low { background-color: #14532d; }
.badge-med { background-color: #b45309; }   /* ORANGE */
.badge-high { background-color: #7f1d1d; }

.alert-low {
    border-left: 6px solid #22c55e;
    background-color: #052e1c;
    padding: 16px;
    border-radius: 10px;
}

.alert-med {
    border-left: 6px solid #fbbf24;           /* ORANGE */
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
    font-size: 0.85rem;
    color: #9ca3af;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# Header
# ===============================
st.markdown("""
<h1 style="margin-bottom:0;">MedGuard AI</h1>
<p class="note">
AI-assisted early warning system for continuous patient monitoring
</p>
""", unsafe_allow_html=True)

# ===============================
# Scenario Selector
# ===============================
scenario = st.selectbox(
    "Patient Scenario",
    ["Stable Patient", "Early Deterioration", "Critical Condition"]
)

# ===============================
# Patient Data
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
# Train Model (Global)
# ===============================
def generate_training_data(samples=400):
    rows = []
    for _ in range(samples):
        hr = np.random.normal(85, 15)
        bp = np.random.normal(120, 20)
        spo2 = np.random.normal(96, 3)
        temp = np.random.normal(37, 0.6)
        label = int((hr > 100) or (bp < 95) or (spo2 < 92))
        rows.append([hr, bp, spo2, temp, label])
    return pd.DataFrame(rows, columns=["hr","bp","spo2","temp","label"])

train = generate_training_data()
X_train = train[["hr","bp","spo2","temp"]]
y_train = train["label"]

model = LogisticRegression()
model.fit(X_train, y_train)

# ===============================
# Prediction
# ===============================
X_patient = data[["heart_rate","systolic_bp","spo2","temperature"]]
data["risk"] = model.predict_proba(X_patient)[:,1]
risk = data.iloc[-1]["risk"]

# ===============================
# Risk Level
# ===============================
if risk < 0.35:
    level, badge, alert = "Stable", "badge-low", "alert-low"
elif risk < 0.7:
    level, badge, alert = "Early Deterioration", "badge-med", "alert-med"
else:
    level, badge, alert = "Critical Condition", "badge-high", "alert-high"

# ===============================
# Layout
# ===============================
left, right = st.columns([1.3, 2])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Patient Overview")
    st.markdown(f"<div class='metric'>Risk Score: {risk:.2f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='badge {badge}'>{level}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Live Vitals")
    st.write(f"❤️ Heart Rate: **{int(data.iloc[-1]['heart_rate'])} bpm**")
    st.write(f"🩸 Systolic BP: **{int(data.iloc[-1]['systolic_bp'])} mmHg**")
    st.write(f"🫁 SpO₂: **{data.iloc[-1]['spo2']:.1f}%**")
    st.write(f"🌡️ Temperature: **{data.iloc[-1]['temperature']:.1f} °C**")
    st.markdown("</div>", unsafe_allow_html=True)

    if level != "Stable":
        st.markdown(f"<div class='{alert}'>", unsafe_allow_html=True)
        st.markdown(f"""
**{level} Alert**  
Physiological patterns indicate abnormal progression requiring attention.
""")
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Risk Trajectory (Last 48h)")
    st.line_chart(data.set_index("hour")["risk"])
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Clinical Interpretation"):
        if level == "Stable":
            st.write("Patient vitals are stable with no signs of physiological stress.")
        elif level == "Early Deterioration":
            st.write(
                "Early compensatory stress detected. Trends suggest rising workload "
                "on cardiovascular and respiratory systems."
            )
        else:
            st.write(
                "Severe instability detected. Patterns align with high-risk clinical deterioration."
            )
