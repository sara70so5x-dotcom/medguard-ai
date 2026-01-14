import streamlit as st
import numpy as np
import pandas as pd

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="MedGuard AI",
    page_icon="🛡️",
    layout="centered"
)

# ================== SESSION STATE ==================
if "lang" not in st.session_state:
    st.session_state.lang = "ar"

# ================== LANGUAGE BUTTONS ==================
col1, col2 = st.columns(2)
with col1:
    if st.button("🇸🇦 عربي"):
        st.session_state.lang = "ar"
with col2:
    if st.button("🇬🇧 English"):
        st.session_state.lang = "en"

lang = st.session_state.lang

# ================== TEXT CONTENT ==================
TEXT = {
    "ar": {
        "title": "🛡️ MedGuard AI",
        "subtitle": "نظام دعم قرار سريري يعتمد على الذكاء الاصطناعي",
        "problem_title": "🧠 المشكلة السريرية",
        "problem": "تدهور حالة المريض يحدث غالبًا بشكل تدريجي وغير ملحوظ.",
        "button": "▶️ تحليل حالة المريض",
        "snapshot": "📊 ملخص حالة المريض",
        "decision": "🟠 القرار السريري",
        "xai": "🧠 لماذا هذا القرار؟",
        "outcome": "🔮 ماذا لو لم يتم التدخل؟",
        "timing": "⏱️ أفضل وقت للتدخل",
        "trajectory": "📈 المسار الزمني للمخاطر",
        "decision_text": "المريض يسير في مسار تدهور محتمل ويُنصح بالتدخل المبكر."
    },
    "en": {
        "title": "🛡️ MedGuard AI",
        "subtitle": "AI-powered clinical decision support system",
        "problem_title": "🧠 Clinical Problem",
        "problem": "Patient deterioration often occurs silently over time.",
        "button": "▶️ Analyze Patient Case",
        "snapshot": "📊 Patient Snapshot",
        "decision": "🟠 Clinical Decision",
        "xai": "🧠 Why this decision?",
        "outcome": "🔮 What if no action is taken?",
        "timing": "⏱️ Best Time to Intervene",
        "trajectory": "📈 Risk Trajectory",
        "decision_text": "The patient is entering a deterioration trajectory. Early intervention is recommended."
    }
}

# ================== HEADER ==================
st.title(TEXT[lang]["title"])
st.caption(TEXT[lang]["subtitle"])

st.subheader(TEXT[lang]["problem_title"])
st.write(TEXT[lang]["problem"])

# ================== DATA ==================
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

def calculate_risk(row):
    risk = 0
    if row["heart_rate"] > 100: risk += 0.3
    if row["systolic_bp"] < 100: risk += 0.3
    if row["spo2"] < 94: risk += 0.25
    if row["temperature"] > 38: risk += 0.15
    return min(risk, 1.0)

# ================== RUN ==================
if st.button(TEXT[lang]["button"]):
    data = generate_patient_data()
    data["risk_score"] = data.apply(calculate_risk, axis=1)
    last = data.iloc[-1]

    st.subheader(TEXT[lang]["snapshot"])
    st.write("HR ↑ | BP ↓ | SpO₂ ↓ | Temp stable")

    st.subheader(TEXT[lang]["decision"])
    st.metric("Risk Score", round(last["risk_score"], 2))
    st.success(TEXT[lang]["decision_text"])

    st.subheader(TEXT[lang]["xai"])
    st.write("• Heart rate increasing")
    st.write("• Blood pressure dropping")

    st.subheader(TEXT[lang]["outcome"])
    st.warning("78% deteriorated within 8 hours | ICU risk +35%")

    st.subheader(TEXT[lang]["timing"])
    st.success("Intervene within the next 90 minutes")

    st.subheader(TEXT[lang]["trajectory"])
    st.line_chart(data.set_index("hour")["risk_score"])
