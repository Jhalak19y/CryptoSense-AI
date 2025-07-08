import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="CryptoSense AI", layout="wide")
st.title("🛡️ CryptoSense AI – Cryptocurrency Fraud Detection")

# 🔹 Sidebar Help
st.sidebar.markdown("### 📘 How it Works")
st.sidebar.info("""
This app uses **Isolation Forest** to detect fraudulent crypto transactions based on unusual patterns in:
- Amount
- Confirmations
- Gas Fees
- Wallet Age
- Whale Activity
""")

# 🔹 Date-Time Display
st.caption(f"📅 Last checked: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")

# 🔹 ML Model Trainer
@st.cache_data
def train_model():
    data = pd.DataFrame({
        'amount': np.random.exponential(1000, 1000),
        'confirmations': np.random.randint(0, 10, 1000),
        'gas_fee': np.random.exponential(10, 1000),
        'account_age_days': np.random.randint(0, 3650, 1000),
        'is_whale_transfer': np.random.choice([0, 1], size=1000, p=[0.95, 0.05])
    })
    for _ in range(20):
        data.loc[np.random.randint(0, 1000), ['amount', 'gas_fee', 'confirmations']] = [100000, 0, 0]
    model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    model.fit(data)
    return model

model = train_model()

# 🔹 Tabbed Layout
tab1, tab2 = st.tabs(["🧠 Fraud Detection", "📈 Bitcoin Forecast (Coming Soon)"])

# -------------------
# TAB 1 – FRAUD DETECTION
# -------------------
with tab1:

    mode = st.radio("Choose Input Method:", ["🔢 Manual Entry", "📤 Upload CSV"])

    if mode == "🔢 Manual Entry":
        st.subheader("📝 Enter Transaction")

        with st.form(key="form_input"):
            amount = st.number_input("Amount (USD) 💰", min_value=0.0, help="Enter the transaction value in USD")
            confirmations = st.slider("Confirmations 🔄", 0, 10)
            gas_fee = st.number_input("Gas Fee ⛽", min_value=0.0)
            account_age = st.slider("Wallet Age (Days) 📆", 0, 3650)
            is_whale = st.selectbox("Is Whale Transfer? 🐋", [0, 1])

            col1, col2 = st.columns(2)
            submit = col1.form_submit_button("🔍 Predict")
            reset = col2.form_submit_button("🧹 Clear")

        if submit:
            input_df = pd.DataFrame([[amount, confirmations, gas_fee, account_age, is_whale]],
                columns=['amount', 'confirmations', 'gas_fee', 'account_age_days', 'is_whale_transfer'])
            prediction = model.predict(input_df)[0]
            score = model.decision_function(input_df)[0]

            if score < -0.1:
                risk = "🚨 High Risk"
                st.error(risk)
            elif score < 0.1:
                risk = "⚠️ Medium Risk"
                st.warning(risk)
            else:
                risk = "✅ Low Risk"
                st.success(risk)

            st.caption(f"Anomaly Score: {score:.4f}")

        if reset:
            st.experimental_rerun()

    elif mode == "📤 Upload CSV":
        st.subheader("📄 Upload Transaction CSV")
        file = st.file_uploader("Upload a CSV with columns: amount, confirmations, gas_fee, account_age_days, is_whale_transfer", type="csv")

        st.markdown("""
        📥 [Download Sample CSV](https://raw.githubusercontent.com/your-username/your-repo-name/main/sample_transactions.csv)
        """)

        if file:
            df = pd.read_csv(file)
            predictions = model.predict(df)
            scores = model.decision_function(df)

            def get_label(score):
                if score < -0.1:
                    return "🚨 High Risk"
                elif score < 0.1:
                    return "⚠️ Medium Risk"
                else:
                    return "✅ Low Risk"

            df['Score'] = scores
            df['Risk'] = df['Score'].apply(get_label)

            # Filter by amount
            st.markdown("### 🎯 Filter by Transaction Amount")
            min_amt = st.slider("Minimum Amount", 0, int(df['amount'].max()), 100)
            df = df[df['amount'] >= min_amt]

            st.dataframe(df)

            # Metrics
            total = len(df)
            fraud = len(df[df['Risk'] == "🚨 High Risk"])
            normal = len(df[df['Risk'] == "✅ Low Risk"])

            st.metric("📊 Total", total)
            st.metric("🚨 High Risk", fraud)
            st.metric("✅ Low Risk", normal)

            # Pie Chart
            pie_chart = px.pie(df, names='Risk', title='Prediction Distribution')
            st.plotly_chart(pie_chart, use_container_width=True)

            # Download Results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results CSV", csv, file_name="fraud_predictions.csv", mime="text/csv")

        st.markdown("📥 Or click below to try example test data:")

        if st.button("✨ Try Example"):
            sample = pd.DataFrame({
                'amount': [500, 100000, 80, 50, 250000],
                'confirmations': [6, 0, 2, 9, 0],
                'gas_fee': [3.2, 0.0, 5.5, 1.0, 0.0],
                'account_age_days': [400, 10, 2000, 150, 5],
                'is_whale_transfer': [0, 1, 0, 0, 1]
            })
            sample['Score'] = model.decision_function(sample)
            sample['Risk'] = sample['Score'].apply(get_label)
            st.write("🧪 Sample Predictions", sample)

            csv = sample.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Sample CSV", csv, file_name="sample_results.csv", mime="text/csv")

# -------------------
# TAB 2 – BTC Forecast (Placeholder)
# -------------------
with tab2:
    st.header("📈 Bitcoin Price Forecast")
    st.markdown("This section uses Facebook Prophet to forecast future Bitcoin prices based on historical data.")

    n_days = st.slider("🔮 Predict how many days ahead?", 7, 90, 30)

    @st.cache_data
    def load_btc_data():
        import yfinance as yf
        df = yf.download('BTC-USD', start='2020-01-01')
        df = df.reset_index()[['Date', 'Close']]
        df.columns = ['ds', 'y']
        return df

    btc_data = load_btc_data()

    try:
        from prophet import Prophet
    except:
        from fbprophet import Prophet  # fallback if using older version

    model = Prophet(daily_seasonality=True)
    model.fit(btc_data)

    future = model.make_future_dataframe(periods=n_days)
    forecast = model.predict(future)

    st.subheader(f"📅 Forecast for Next {n_days} Days")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.markdown("📉 **Forecast Columns Explained:**")
    st.caption("""
    - `yhat`: Predicted price  
    - `yhat_lower`, `yhat_upper`: Prediction range (uncertainty)
    """)

    # Download forecast option
    csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_days).to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast CSV", csv, file_name="bitcoin_forecast.csv", mime="text/csv")
