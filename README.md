# 🛡️ CryptoSense AI – Cryptocurrency Fraud Detection App

CryptoSense AI is a powerful, interactive machine learning web app built with **Streamlit** that helps detect potentially **fraudulent cryptocurrency transactions** in real time.

> ⚠️ Fraud in crypto is growing rapidly — this tool empowers users, exchanges, and students to **analyze, detect, and understand** suspicious transactions based on behavior patterns.

---

## 🧠 Features

✅ Real-time prediction using ML (Isolation Forest)  
✅ Two input modes: Manual Entry OR CSV Upload  
✅ Built-in **Try Example** feature with test data  
✅ Detects anomalies like:
- Whale transfers from new accounts
- Zero confirmations
- Zero gas fees
- Extremely high-value transactions

✅ Predicts and displays results with clean UI  
✅ Download results as CSV  
✅ Option to reset/clear manual inputs  

---

## 🚀 Demo

![App Screenshot](https://via.placeholder.com/800x400?text=App+Demo+Screenshot)

Or try it out:
(*Replace this with your actual Streamlit Cloud link after deployment*)

---

## 📦 Tech Stack

- **Python**
- **Streamlit** – for the web interface
- **Scikit-learn** – for fraud detection (Isolation Forest)
- **Pandas** + **NumPy** – for data handling

---

## 🧪 Try It Now

### ➤ Option 1: Manual Entry

Enter a transaction directly in the form:
- Amount
- Confirmations
- Gas fee
- Wallet age
- Whale transfer status

### ➤ Option 2: Upload CSV

Upload a `.csv` with columns:

📥 [Download Sample CSV](https://raw.githubusercontent.com/your-username/your-repo-name/main/sample_transactions.csv)

---

## 📤 Download Prediction Results

Once the predictions are complete, you can export results using the **Download CSV** button provided.

---

## 🛠️ Run Locally

### 1. Clone this repo:
```bash
git clone https://github.com/your-username/cryptosense-ai.git
cd cryptosense-ai
