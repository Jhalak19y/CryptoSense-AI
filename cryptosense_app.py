import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="CryptoSense AI", layout="wide")
st.title("🛡️ CryptoSense AI – Cryptocurrency Fraud Detection & BTC Forecast")

tab1, tab2 = st.tabs(["🧪 Fraud Detection", "📈 Bitcoin Forecast"])

# ---------------------------- TAB 1: FRAUD DETECTION ----------------------------
with tab1:
    st.subheader("🧪 Detect Suspicious Crypto Transactions")
    st.markdown("Enter data manually or upload a `.csv` file to detect potential fraud using Isolation Forest.")

    sample_data = pd.DataFrame({
        'Amount': [120000, 0.5, 8700],
        'Confirmations': [0, 3, 8],
        'GasFee': [0.0001, 0.001, 0],
        'WalletAgeDays': [1, 400, 12],
        'WhaleTransfer': [1, 0, 1]
    })

    input_mode = st.radio("Choose input mode:", ["Manual Entry", "Upload CSV"])

    if input_mode == "Manual Entry":
        with st.form("manual_form"):
            amount = st.number_input("💰 Amount (in USD)", min_value=0.0, value=1000.0)
            confirmations = st.slider("⛓️ Confirmations", 0, 10, 2)
            gas_fee = st.number_input("⛽ Gas Fee (ETH)", min_value=0.0, value=0.0005)
            wallet_age = st.number_input("📅 Wallet Age (in days)", min_value=0, value=30)
            whale_transfer = st.selectbox("🐋 Whale Transfer?", ["No", "Yes"])

            submit = st.form_submit_button("🔍 Predict")
            clear = st.form_submit_button("🧹 Clear")

        if clear:
            st.experimental_rerun()

        if submit:
            input_data = pd.DataFrame([[
                amount, confirmations, gas_fee, wallet_age, 1 if whale_transfer == "Yes" else 0
            ]], columns=['Amount', 'Confirmations', 'GasFee', 'WalletAgeDays', 'WhaleTransfer'])

            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(sample_data)
            prediction = model.predict(input_data)

            result = "🚨 Suspicious Transaction" if prediction[0] == -1 else "✅ Normal Transaction"
            st.success(f"**Prediction:** {result}")

    else:
        uploaded_file = st.file_uploader("📤 Upload a CSV file", type=["csv"])

        if st.button("📂 Try Example"):
            data = sample_data.copy()
        elif uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                if not all(col in data.columns for col in sample_data.columns):
                    st.error("❌ CSV must contain: Amount, Confirmations, GasFee, WalletAgeDays, WhaleTransfer")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Failed to read CSV: {e}")
                st.stop()
        else:
            data = None

        if data is not None:
            st.write("🔎 Input Data:", data)

            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(sample_data)

            predictions = model.predict(data)
            data['Prediction'] = np.where(predictions == -1, '🚨 Fraud', '✅ Legit')

            st.success("✅ Predictions completed.")
            st.dataframe(data)

            fig = px.histogram(data, x='Prediction', color='Prediction', title="Prediction Summary")
            st.plotly_chart(fig)

            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

# ---------------------------- TAB 2: BITCOIN FORECAST ----------------------------
with tab2:
    st.header("📈 Bitcoin Price Forecast")
    st.markdown("This section uses Facebook Prophet to forecast future Bitcoin prices based on historical data.")

    n_days = st.slider("🔮 Predict how many days ahead?", 7, 90, 30)

    @st.cache_data
    def load_btc_data():
        import pandas as pd
        try:
            import yfinance as yf
            df = yf.download('BTC-USD', start='2020-01-01')
            df = df.reset_index()[['Date', 'Close']]
            df.columns = ['ds', 'y']
            if df.empty or df['y'].isnull().sum() > len(df) - 2:
                raise ValueError("yfinance returned empty")
        except:
            # fallback to static CSV from GitHub
            url = "https://raw.githubusercontent.com/Jhalak19y/cryptosense-ai/main/btc_data.csv"
            df = pd.read_csv(url)
            df.columns = ['ds', 'y']
            df['ds'] = pd.to_datetime(df['ds'])
        return df

    btc_data = load_btc_data()

    if btc_data.empty or btc_data['y'].isnull().sum() > len(btc_data) - 2:
        st.error("❌ Failed to load Bitcoin data. Please try again later.")
    else:
        try:
            from prophet import Prophet
        except:
            from fbprophet import Prophet

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

        csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_days).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Forecast CSV", csv, file_name="bitcoin_forecast.csv", mime="text/csv")
