import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Set wide page layout to use full screen width
st.set_page_config(layout="wide")

# ----- PAGE STYLE -----
custom_css = """
<style>
    body { background: #101010 !important; color: #39FF14 !important; }
    .main, .block-container, .css-18e3th9 {
        max-width: 100vw !important;
        padding-left: 1.5vw !important;
        padding-right: 1.5vw !important;
    }
    .main { background: #18181a !important; }
    .stButton>button {
        background-color: #39FF14;
        color: #18181a;
        font-weight: bold;
        border-radius: 7px;
        border: 2px solid #39FF14;
        font-size: 22px;
    }
    .stMarkdown { color: #39FF14!important; }
    .stSelectbox > div > div {
        background: linear-gradient(90deg, #39FF14, #00c3ff);
        color: #18181a;
        border-radius: 8px;
        padding-left: 10px;
    }
    .stSlider > div > div > input {
        accent-color: #39FF14;
    }
    .stRadio > div > label {
        color: #39FF14 !important;
        font-weight: bold;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ----- APP TITLE -----
st.title("💰 Fraud Detection in Online Transactions (~By Nasreen)")

# ---------------- FORM ----------------
with st.form("fraud_form"):
    amount = st.number_input(
        "💸 Transaction Amount (₹)", min_value=0.0, format="%.2f", value=100.0
    )
    location = st.selectbox(
        "🌍 Transaction Location",
        ["Mumbai", "Delhi", "Bangalore", "Pune", "Hyderabad", "Other"],
    )
    txn_type = st.selectbox(
        "💳 Transaction Type",
        ["E-Commerce", "UPI", "ATM Withdrawal", "POS Swipe", "Bank Transfer"],
    )
    hour = st.slider("🕒 Transaction Hour", 0, 23, 12)
    day = st.selectbox(
        "📆 Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    )
    is_new_location = st.radio(
        "🛰️ Is New Location?", [0, 1], index=0, horizontal=True
    )
    is_high_amount = st.radio(
        "⚡ High Amount?", [0, 1], index=0, horizontal=True
    )

    submit_btn = st.form_submit_button("⚡ Predict Fraud")

# ---------------- PREDICTION ----------------
if submit_btn:
    pipe = joblib.load("fraud_detection_pipeline.pkl")

    input_dict = {
        "Amount": amount,
        "Location": location,
        "Type": txn_type,
        "Hour": hour,
        "Day": day,
        "Is_New_Location": int(is_new_location),
        "Is_High_Amount": int(is_high_amount),
    }

    input_df = pd.DataFrame([input_dict])

    pred = pipe.predict(input_df)[0]
    proba_scalar = float(pipe.predict_proba(input_df)[:, 1][0])

    st.markdown("### 👽 Prediction Hack Result")

    if pred == 1:
        st.markdown(
            f"""
            <div style='padding:1em; background:#1a202c;
                        border:3px solid #ff3131; border-radius:12px;'>
                <span style='font-size:2em;color:#ff3131;
                             font-weight:bold;'>⚠️ FRAUD DETECTED!</span><br>
                <span style='color:#ffea00;font-size:1.4em;'>
                    Probability: {proba_scalar:.2%}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style='padding:1em; background:#102512;
                        border:3px solid #39FF14; border-radius:12px;'>
                <span style='font-size:2em;color:#39FF14;
                             font-weight:bold;'>✔️ Transaction is Legitimate</span><br>
                <span style='color:#00ffea;font-size:1.4em;'>
                    Probability of Fraud: {proba_scalar:.2%}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("##### Explainable Insights:")

    insights = ""
    if is_high_amount:
        insights += "- 💸 Unusually high amount<br>"
    if is_new_location:
        insights += f"- 🛰️ New location used: <b>{location}</b><br>"

    st.markdown(
        insights or "✅ No suspicious attributes detected.",
        unsafe_allow_html=True,
    )

    st.toast("Prediction complete!", icon="⚡")
