import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Page config
st.set_page_config(page_title="CryptoSense AI", layout="centered")
st.title("🛡️ CryptoSense AI - Fraud Detection in Cryptocurrency Transactions")

# -------------------------------
# 📖 Summary Section
# -------------------------------
st.markdown("""
### 🤖 About CryptoSense AI

CryptoSense AI is a machine learning-powered tool that helps users detect **potentially fraudulent cryptocurrency transactions** in real-time.

- 🧠 Uses the **Isolation Forest** algorithm to identify abnormal behavior.
- 🧾 Accepts user input OR CSV uploads for analysis.
- 🚨 Flags suspicious activity like:
  - High-value transfers
  - Zero confirmations
  - Zero gas fees
  - Whale-like behavior

Built with **Python, Streamlit, and Scikit-learn**.
""")

# -------------------------------
# 💻 Train the model
# -------------------------------
@st.cache_data
def train_model():
    n = 1000
    data = pd.DataFrame({
        'amount': np.random.exponential(scale=1000, size=n),
        'confirmations': np.random.randint(0, 10, size=n),
        'gas_fee': np.random.exponential(scale=10, size=n),
        'account_age_days': np.random.randint(0, 3650, size=n),
        'is_whale_transfer': np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
    })

    # Inject synthetic fraud examples
    for i in range(20):
        data.loc[np.random.randint(0, n), ['amount', 'gas_fee', 'confirmations']] = [100000, 0, 0]

    model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    model.fit(data)
    return model

model = train_model()

# -------------------------------
# Sidebar: Mode Switcher
# -------------------------------
mode = st.sidebar.radio("Choose Mode:", ["🔢 Enter Manually", "📤 Upload CSV"])

# -------------------------------
# 🔢 Manual Input Form
# -------------------------------
if mode == "🔢 Enter Manually":
    st.subheader("📝 Enter Transaction Details")

    with st.form(key='form_input'):
        amount = st.number_input("Transaction Amount (USD)", min_value=0.0)
        confirmations = st.slider("Confirmations", 0, 10)
        gas_fee = st.number_input("Gas Fee", min_value=0.0)
        account_age = st.slider("Account Age (in days)", 0, 3650)
        is_whale = st.selectbox("Is Whale Transfer?", [0, 1])

        col1, col2 = st.columns(2)
        submit = col1.form_submit_button("🔍 Predict")
        reset = col2.form_submit_button("🧹 Clear")

    if submit:
        input_data = pd.DataFrame([[amount, confirmations, gas_fee, account_age, is_whale]],
                                  columns=['amount', 'confirmations', 'gas_fee', 'account_age_days', 'is_whale_transfer'])
        prediction = model.predict(input_data)
        if prediction[0] == -1:
            st.error("⚠️ Suspicious Transaction Detected!")
        else:
            st.success("✅ Transaction Looks Normal.")

    if reset:
        st.experimental_rerun()

# -------------------------------
# 📤 Upload CSV or Try Example
# -------------------------------
if mode == "📤 Upload CSV":
    st.subheader("📄 Upload a CSV file")

    st.markdown("""
    📥 [Download sample CSV file](https://raw.githubusercontent.com/your-username/your-repo-name/main/sample_transactions.csv)
    """)

    file = st.file_uploader("Upload CSV with columns: amount, confirmations, gas_fee, account_age_days, is_whale_transfer", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write("📊 Uploaded Data Preview", df.head())

        if st.button("🔍 Predict on Uploaded Data"):
            predictions = model.predict(df)
            df['Prediction'] = ['Fraud' if p == -1 else 'Normal' for p in predictions]
            st.write(df)

            fraud_count = (df['Prediction'] == 'Fraud').sum()
            st.warning(f"🚨 {fraud_count} suspicious transactions detected.")

            # Export
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Prediction CSV", csv, file_name='fraud_predictions.csv', mime='text/csv')

    # Try example test data (no upload)
    st.markdown("📥 Or click below to try with test data:")
    if st.button("✨ Try Example"):
        sample_data = pd.DataFrame({
            'amount': [500, 100000, 80, 50, 250000],
            'confirmations': [6, 0, 2, 9, 0],
            'gas_fee': [3.2, 0.0, 5.5, 1.0, 0.0],
            'account_age_days': [400, 10, 2000, 150, 5],
            'is_whale_transfer': [0, 1, 0, 0, 1]
        })
        predictions = model.predict(sample_data)
        sample_data['Prediction'] = ['Fraud' if p == -1 else 'Normal' for p in predictions]
        st.write("🧪 Sample Prediction Results", sample_data)

        csv = sample_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Sample Prediction CSV", csv, file_name='sample_fraud_results.csv', mime='text/csv')
