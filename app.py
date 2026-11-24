import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("📊 Customer Churn Predictor")
st.caption("Telco Customer Churn • Machine Learning + Streamlit")

with st.sidebar:
    st.header("About the model")
    st.write("The deployed model is a tuned Logistic Regression pipeline with preprocessing included.")
    st.info("Risk levels: Low < 30% • Medium 30–59.9% • High ≥ 60%")

st.subheader("Enter Customer Information")

with st.form("churn_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    with c2:
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    with c3:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=0.01)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=840.0, step=0.01)

    submitted = st.form_submit_button("🔍 Predict Churn Risk", type="primary", use_container_width=True)

if submitted:
    customer = pd.DataFrame({
        "gender": [gender], "SeniorCitizen": [senior_citizen], "Partner": [partner],
        "Dependents": [dependents], "tenure": [tenure], "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines], "InternetService": [internet_service],
        "OnlineSecurity": [online_security], "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection], "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv], "StreamingMovies": [streaming_movies],
        "Contract": [contract], "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method], "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    pred = model.predict(customer)[0]
    churn_idx = list(model.classes_).index("Yes")
    probability = float(model.predict_proba(customer)[0][churn_idx])
    pct = probability * 100

    if pct >= 60:
        risk = "High"; action = "Prioritize this customer for immediate retention outreach."
    elif pct >= 30:
        risk = "Medium"; action = "Monitor this customer and consider a retention offer."
    else:
        risk = "Low"; action = "Customer currently shows a relatively low churn risk."

    st.divider()
    st.subheader("Prediction Result")
    r1, r2, r3 = st.columns(3)
    r1.metric("Churn Probability", f"{pct:.1f}%")
    r2.metric("Predicted Class", "Churn" if pred == "Yes" else "Stay")
    r3.metric("Risk Level", risk)
    if risk == "High": st.error(f"🚨 High risk — {action}")
    elif risk == "Medium": st.warning(f"⚠️ Medium risk — {action}")
    else: st.success(f"✅ Low risk — {action}")

    with st.expander("Customer data sent to the model"):
        st.dataframe(customer, use_container_width=True)

st.divider()
st.caption("Model trained on the Telco Customer Churn dataset. This prediction is a decision-support tool, not a guarantee of customer behavior.")
