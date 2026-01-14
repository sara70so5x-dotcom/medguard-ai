st.markdown("""
<style>
/* ===== Global ===== */
html, body, [class*="css"] {
    background-color: #10161c;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

/* ===== Cards ===== */
.card {
    background-color: #1b2430;
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 24px;
    border: 1px solid #273142;
}

/* ===== Headings ===== */
h1, h2, h3 {
    color: #f8fafc;
}

/* ===== Risk Badges ===== */
.badge-low {
    background: linear-gradient(135deg, #1f7a55, #2ea77a);
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    font-weight: 600;
}

.badge-med {
    background: linear-gradient(135deg, #8f6b1b, #c59b2d);
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    font-weight: 600;
}

.badge-high {
    background: linear-gradient(135deg, #7a1f2b, #b9374b);
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    font-weight: 600;
}

/* ===== Notes / Disclaimer ===== */
.note {
    font-size: 0.85rem;
    color: #94a3b8;
}

/* ===== Charts ===== */
svg {
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)
