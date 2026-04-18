# 🔍 Fake News Detector

A machine learning-based Fake News Detection System built with Python.



## 🛠️ Tech Stack
- Python
- Scikit-learn (TF-IDF + Logistic Regression)
- Gradio (Web Interface)
- NLTK
- Pandas, NumPy

## 📊 Model Performance
- Accuracy: ~91.84%
- ROC-AUC: ~97.18%

## 📁 Dataset
Trained on combined dataset (~83,893 headlines) from:
- ISOT
- FakeNewsNet
- WELFake

## 🚀 How to Run

1. Clone the repository
   git clone https://github.com/YOUR_USERNAME/fake-news-detector.git

2. Create virtual environment
   python -m venv venv
   venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Run the app
   python app.py

## 🎯 Features
- Real-time fake news detection
- Confidence score display
- Rule-based override for conspiracy-style headlines
- Gradio web interface
