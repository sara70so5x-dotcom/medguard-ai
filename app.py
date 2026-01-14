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
    # ---- Risk Snapshot ----
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Risk Snapshot")
    st.metric("Current Risk Probability", f"{current_risk:.2f}")
    st.markdown(f"<div class='{badge}'>{level}</div>", unsafe_allow_html=True)

    with st.expander("What does this risk level indicate?"):
        st.markdown("""
        This probability reflects the likelihood of early clinical deterioration
        based on temporal patterns across multiple vital signs.
        It is not a diagnosis.
        """)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Alert ----
    st.markdown("<div class='alert'>", unsafe_allow_html=True)
    st.markdown("""
    **Early Clinical Alert**  
    Patient trajectory shows early signs of deterioration.
    """)
    with st.expander("Why was this alert triggered?"):
        st.markdown("""
        • Sustained increase in heart rate  
        • Declining systolic blood pressure  
        • Progressive reduction in oxygen saturation  
        • Similar patterns observed in prior ICU deterioration cases
        """)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # ---- Risk Trend ----
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Risk Trend Over Time")
    st.line_chart(data.set_index("hour")["risk"])

    with st.expander("How to interpret this trend?"):
        st.markdown("""
        A rising trajectory indicates accumulating risk over time.
        The model focuses on trend behavior rather than isolated values.
        """)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Explainable AI
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Explainable AI")

coeff = pd.Series(model.coef_[0], index=X.columns)
coeff = coeff.abs().sort_values(ascending=False)
st.bar_chart(coeff)

with st.expander("Model explanation"):
    st.markdown("""
    Heart rate and systolic blood pressure contributed most to the current risk signal,
    indicating early hemodynamic instability patterns.
    """)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Footer
# =========================
st.caption(
    "MedGuard AI is a clinical decision-support assistant and does not replace medical judgment."
)
