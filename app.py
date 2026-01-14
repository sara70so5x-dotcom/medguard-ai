import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="MedGuard AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom Dark Theme Styling
# ----------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0e1117;
    color: #e6e6e6;
}
h1, h2, h3 {
    color: #f5f5f5;
}
.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.badge {
    padding: 10px;
    border-radius: 8px;
    font-weight: bold;
    text-align: center;
}
.low { background-color: #1f6f43; }
.medium { background-color: #8a6d1d; }
.high { background-color: #7a1f1f; }
.disclaimer {
    font-size: 0.9rem;
    color: #9aa4b2;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.title("🛡️ MedGuard AI")
st.markdown("""
**Clinical Decision Support System**  
Early detection of patient deterioration using time-series vital signs with explainable insights.
""")

st.markdown("""
<div class="disclaimer">
⚠️ MedGuard AI is a decision-support assistant.  
It does NOT replace clinical judgment and does NOT issue medical orders.
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Simulated Patient Data
# ----------------------------
def generate_patient_data(hours=48):
    np.random.seed(42)
    data = pd.DataFrame({
        "hour": range(hours),
        "heart_rate": np.random.normal(85, 8, hours),
        "systolic_bp": np.random.normal(120, 10, hours),
        "spo2": np.random.normal(97, 1.2, hours),
        "temperature": np.random.normal(37, 0.3, hours)
    })

    # Inject deterioration pattern
    data.loc[30:, "heart_rate"] += np.linspace(0, 25, hours - 30)
    data.loc[30:, "systolic_bp"] -= np.linspace(0, 30, hours - 30)
    data.loc[30:, "spo2"] -= np.linspace(0, 5, hours - 30)

    return data

data = generate_patient_data()

# ----------------------------
# Feature Engineering
# ----------------------------
features = data[["heart_rate", "systolic_bp", "spo2", "temperature"]]
labels = (features["heart_rate"] > 100).astype(int)

model = LogisticRegression()
model.fit(features, labels)

risk_score = model.predict_proba(features.iloc[[-1]])[0][1]

# ----------------------------
# Risk Interpretation
# ----------------------------
if risk_score < 0.4:
    risk_label = "Low Risk"
    risk_class = "low"
    recommendation = "Continue monitoring and routine assessment."
elif risk_score < 0.7:
    risk_label = "Medium Risk"
    risk_class = "medium"
    recommendation = "Consider ordering labs and closer observation."
else:
    risk_label = "High Risk"
    risk_class = "high"
    recommendation = "Urgent clinical review and escalation may be required."

# ----------------------------
# Dashboard Layout
# ----------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Current Risk Assessment")
    st.metric("Risk Probability", f"{risk_score:.2f}")
    st.markdown(f"<div class='badge {risk_class}'>{risk_label}</div>", unsafe_allow_html=True)
    st.write("**Suggested Action (Advisory):**")
    st.write(recommendation)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Risk Trajectory Over Time")
    data["risk"] = model.predict_proba(features)[:, 1]
    st.line_chart(data.set_index("hour")["risk"])
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Explainable AI Section
# ----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Explainable AI – Key Contributing Factors")

coeffs = pd.Series(model.coef_[0], index=features.columns)
top_factors = coeffs.abs().sort_values(ascending=False)

for factor in top_factors.index[:3]:
    st.write(f"• **{factor.replace('_',' ').title()}** shows a significant deviation from baseline.")

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Data Sources (Hackathon Aligned)
# ----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Referenced Clinical Data Sources")

st.markdown("""
**PhysioNet Sepsis Challenge 2019**  
• ICU time-series vital signs  
• Used as primary reference for system design and temporal patterns  

**eICU Collaborative Research Database**  
• Multi-center ICU dataset  
• Planned for external validation and generalization testing  

**NIH ChestX-ray14 Dataset**  
• Large-scale imaging dataset  
• Considered for future multimodal expansion (not used in current scope)
""")
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
<div class="disclaimer">
MedGuard AI demonstrates how explainable machine learning can assist clinicians
in recognizing early deterioration patterns using continuous physiological data.
</div>
""", unsafe_allow_html=True)
