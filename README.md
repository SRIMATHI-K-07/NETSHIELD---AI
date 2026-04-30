# 🔐 NetShield AI - Phishing URL Detection System

NetShield AI is a machine learning-based web application that detects phishing URLs in real time. It uses feature extraction and a Random Forest classifier to identify whether a URL is safe or malicious.

---

## 🚀 Features

- 🔍 Real-time URL analysis  
- 🤖 Machine Learning (Random Forest)  
- 📊 Automatic feature extraction  
- ⚠️ Explainability (reasons for detection)  
- 📈 Confidence score display  
- 🌐 Web interface using Flask  

---

## 🧠 How It Works

1. User enters a URL  
2. System extracts features like:
   - URL length  
   - Number of dots  
   - HTTPS usage  
   - Domain information  
3. Features are passed into a trained ML model  
4. Output is displayed as:
   - ✅ Safe URL  
   - ❌ Phishing URL  
5. System also shows:
   - Confidence score  
   - Reasons for classification  

---

## 🛠️ Tech Stack

- Python  
- Flask  
- HTML/CSS  
- Scikit-learn  
- Pandas & NumPy  

---
cd netshield-ai
pip install -r requirements.txt
python app.py
