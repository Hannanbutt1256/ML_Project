import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go

# -----------------------------
# Load trained model
# -----------------------------

def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Loan Approval Prediction")
st.write("Fill in applicant details to estimate loan approval likelihood.")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    annual_income = st.number_input("Annual Income", min_value=0.0, step=1000.0)
    monthly_income = st.number_input("Monthly Income", min_value=0.0, step=500.0)
    net_worth = st.number_input("Net Worth", min_value=0.0, step=1000.0)
    age = st.number_input("Age", min_value=18, max_value=100)
    experience = st.number_input("Work Experience (Years)", min_value=0, max_value=60)

with col2:
    total_assets = st.number_input("Total Assets", min_value=0.0, step=1000.0)
    credit_score = st.number_input("Credit Score", min_value=300.0, max_value=900.0)
    credit_history = st.number_input("Length of Credit History (Years)", min_value=0, max_value=60)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, step=1000.0)
    dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0)

education_level = st.selectbox(
    "Education Level",
    options=[0, 1, 2, 3],
    help="0: High School | 1: Bachelor | 2: Master | 3: PhD"
)

employment_status = st.selectbox(
    "Employment Status",
    ["Employed", "Self-Employed", "Unemployed"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Widowed"]
)

home_ownership = st.selectbox(
    "Home Ownership Status",
    ["Rent", "Own", "Other"]
)

loan_purpose = st.selectbox(
    "Loan Purpose",
    ["Debt Consolidation", "Education", "Home", "Other"]
)

# -----------------------------
# One-Hot Encoding
# -----------------------------
employment_self = employment_status == "Self-Employed"
employment_unemployed = employment_status == "Unemployed"

married = marital_status == "Married"
single = marital_status == "Single"
widowed = marital_status == "Widowed"

home_other = home_ownership == "Other"
home_own = home_ownership == "Own"
home_rent = home_ownership == "Rent"

purpose_debt = loan_purpose == "Debt Consolidation"
purpose_education = loan_purpose == "Education"
purpose_home = loan_purpose == "Home"
purpose_other = loan_purpose == "Other"

# -----------------------------
# Feature Vector (ORDER MATTERS)
# -----------------------------
features = np.array([[
    annual_income,
    monthly_income,
    net_worth,
    age,
    experience,
    total_assets,
    credit_score,
    credit_history,
    loan_amount,
    dti_ratio,
    education_level,
    employment_self,
    employment_unemployed,
    married,
    single,
    widowed,
    home_other,
    home_own,
    home_rent,
    purpose_debt,
    purpose_education,
    purpose_home,
    purpose_other
]])

# -----------------------------
# Prediction
# -----------------------------
st.divider()

def approval_gauge(probability):
    fig = go.Figure(go.Pie(
        values=[probability, 100 - probability],
        labels=["Approved", "Remaining"],
        hole=0.75,
        marker=dict(colors=["#00C853", "#E0E0E0"]),
        textinfo="none"
    ))

    fig.update_layout(
        showlegend=False,
        annotations=[dict(
            text=f"{probability:.1f}%",
            x=0.5, y=0.5,
            font_size=24,
            showarrow=False
        )],
        margin=dict(t=10, b=10, l=10, r=10)
    )

    return fig


if st.button("🔍 Predict Loan Approval"):
    prediction = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(features)[0][1] * 100
        st.subheader("Approval Probability")

        st.subheader("Approval Probability")
        st.plotly_chart(approval_gauge(probability), use_container_width=True)


    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
