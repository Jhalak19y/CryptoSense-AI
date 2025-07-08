# 🛡️ CryptoSense AI – Cryptocurrency Fraud Detection & Bitcoin Forecasting App

**CryptoSense AI** is an interactive **machine learning web app** built with **Streamlit** that allows users to:

- 🧠 Detect potentially **fraudulent cryptocurrency transactions** using ML
- 📈 Forecast **Bitcoin prices** with Prophet based on historical data

> 🚨 With rising crypto scams, this app empowers users, exchanges, and students to detect risks and analyze future trends intelligently.

---

## 🧠 Features

### ✅ Real-Time Crypto Fraud Detection
- Manual input or CSV file upload
- Built with **Isolation Forest** model
- Detects:
  - Whale transfers from new wallets
  - Extremely high/low transaction values
  - Zero confirmations or gas fees
- "Try Example" button with test data
- "Clear Input" button to reset form
- Export predictions to CSV

### 📈 Bitcoin Price Forecasting (NEW!)
- Forecasts next 7–90 days using **Facebook Prophet**
- Uses live BTC data from `yfinance`
- Automatically falls back to preloaded data if API fails
- Interactive chart of predicted price trends
- Export forecast as CSV

---

## 📸 Demo Screenshot

> Replace this with your own screenshot after deployment:

![App Screenshot](https://via.placeholder.com/800x400?text=Your+App+Demo+Screenshot+Here)

---

## 🧪 Try It Yourself

### ➤ Option 1: Manual Entry
Enter:
- 💰 Amount
- ⛓️ Confirmations
- ⛽ Gas Fee
- 📅 Wallet Age
- 🐋 Whale Transfer (Yes/No)

### ➤ Option 2: Upload a CSV
Upload `.csv` file with columns:

Amount, Confirmations, GasFee, WalletAgeDays, WhaleTransfer

📥 [Download Sample CSV](https://raw.githubusercontent.com/Jhalak19y/cryptosense-ai/main/sample_transactions.csv)

---

## 📈 Bitcoin Forecast (Prophet)

- Set number of forecast days using slider
- Automatically fetches BTC historical data
- If yfinance fails, loads backup CSV
- Interactive line plot with confidence intervals
- Download forecast as `.csv`

---

## 📤 Export Options

- ✅ Download predictions as CSV
- ✅ Download BTC price forecast
- 🔁 Reset manual input instantly

---

## 🛠️ Run the App Locally

### 1. Clone this repository:
```bash
git clone https://github.com/Jhalak19y/cryptosense-ai.git
cd cryptosense-ai
2. Install requirements:
pip install -r requirements.txt

3. Run the Streamlit app:
streamlit run cryptosense_app.py
🌐 Deploy Online
You can deploy this app easily using Streamlit Cloud:

Push your code to a public GitHub repo

Go to https://streamlit.io/cloud

Click "New app" and connect your repo

Done ✅

📁 Repository Structure

cryptosense-ai/
├── cryptosense_app.py         ← Main Streamlit app
├── requirements.txt           ← Required Python packages
├── btc_data.csv               ← Backup BTC price data
├── sample_transactions.csv    ← Test data for fraud detection
└── README.md                  ← You're here!

⚙️ Tech Stack
Python

Streamlit – Web UI

scikit-learn – Isolation Forest for anomaly detection

Prophet – BTC price forecasting

yfinance – Fetching live BTC price

pandas, numpy, plotly

🙋‍♀️ Author
Jhalak Yadav
BCA in Media & IT | Passionate about Machine Learning, Fintech & App Dev
