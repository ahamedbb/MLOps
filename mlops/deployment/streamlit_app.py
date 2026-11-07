import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Bank Customer Churn Prediction App")
st.markdown("Predict whether a bank customer is likely to churn based on their profile and account statistics.")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="ahamedb/churn-model",
        filename="best_churn_model.joblib"
    )
    return joblib.load(model_path)

with st.spinner("Loading trained XGBoost model from Hugging Face..."):
    model = load_model()

st.sidebar.header("Customer Input Parameters")

def user_input_features():
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
    geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
    age = st.sidebar.slider("Age", 18, 92, 38)
    tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
    balance = st.sidebar.number_input("Account Balance", 0.0, 250000.0, 60000.0)
    num_of_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = st.sidebar.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.sidebar.selectbox("Is Active Member", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    estimated_salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

    data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary
    }
    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader("Customer Profile Summary")
st.dataframe(input_df)

if st.button("Predict Churn", type="primary"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction Results")
    if prediction[0] == 1:
        st.error(f"⚠️ This customer is **likely to churn** (Probability: {prediction_proba[0][1]:.2%})")
    else:
        st.success(f"✅ This customer is **likely to stay** (Probability of retention: {prediction_proba[0][0]:.2%})")
