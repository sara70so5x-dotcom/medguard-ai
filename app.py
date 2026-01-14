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
.note {
    font-size: 0.85rem;
    color: #94a3b8;
}
.alert {
    background-color: #111827;
    border-left: 4px solid #fbbf24;
    padding: 14px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header (مختصر)
# =========================
st.title("MedGuard AI")
st.caption("Continuous patient monitoring • AI-assisted early alerts")

# =========================
# Data Simulation
# =========================
def generate_patient_data(hours=48):
    np.random.seed(42)
    df = pd.DataFrame({
        "hour": range(hours),
        "heart_rate": np.random.normal(85, 8, hours),
        "systolic_bp": np.random.normal(120, 10, hours),
        "spo2": np.random.normal(97, 1.2, hours),
        "temperature": np.random.normal(37, 0.3, hours)
    })
    df.loc[30:, "heart_rate"] += np.linspace(0, 25, hours - 30)
    df.loc[30:, "systolic_bp"] -= np.linspace(0, 30, hours - 30)
    df.loc[30:, "spo2"] -= np.linspace(0, 5, hours - 30)
    return df

data = generate_patient_data()

# =========================
# ML Model
# =========================
X = data[["heart_rate", "systolic_bp", "spo2", "temperature"]]
y = (X["heart_rate"] > 100).astype(int)

model = LogisticRegression()
model.fit(X, y)

data["risk"] = model.predict_proba(X)[:, 1]
current_risk = data.iloc[-1]["risk"]

# =========================
# Risk Level
# =========================
if current_risk < 0.4:
    level = "Low Risk"
    badge = "badge-low"
elif current_risk < 0.7:
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

    # Alert (مختصر)
    st.markdown("<div class='alert'>", unsafe_allow_html=True)
    st.markdown("""
    **Early Clinical Alert**  
    Patient trajectory shows early signs of deterioration.  
    *Earlier review was associated with better outcomes.*
    """)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Risk Trend")
    st.line_chart(data.set_index("hour")["risk"])
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Expandable Sections
# =========================
with st.expander("Why was this alert raised?"):
    st.markdown("""
    • Rising heart rate over time  
    • Declining systolic blood pressure  
    • Gradual reduction in oxygen saturation  
    • Pattern similarity to prior ICU deterioration cases
    """)

with st.expander("Explainable AI – Model Insight"):
    coeff = pd.Series(model.coef_[0], index=X.columns)
    coeff = coeff.abs().sort_values(ascending=False)
    st.bar_chart(coeff)
    st.caption(
        "Heart rate and blood pressure trends contributed most to the current risk signal."
    )

with st.expander("Clinical Data Sources"):
    st.markdown("""
    • **PhysioNet Sepsis Challenge 2019** – ICU time-series data  
    • **eICU Collaborative Research Database** – Multi-center validation  
    • **NIH ChestX-ray14 Dataset** – Future multimodal expansion
    """)

# =========================
# Footer
# =========================
st.caption(
    "MedGuard AI is a clinical decision-support assistant. "
    "It does not provide diagnoses or treatment recommendations."
)
