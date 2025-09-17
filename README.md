Sentiment Analysis Web App 🚀

An **End-to-End Sentiment Analysis application** built using **FastAPI, scikit-learn, HTML, and CSS**.  
This project predicts whether a given text review is **Positive** or **Negative** using multiple machine learning models.  

## 🔥 Features
- Clean text preprocessing (special characters, stopwords, HTML tags, URLs).
- Supports **Logistic Regression**, **SVM**, and **Multinomial Naive Bayes**.
- Backend built with **FastAPI**.
- Frontend with **HTML + CSS** for user interaction.
- End-to-end structure (train → pickle → serve → predict).
- Ready for deployment.

## 📂 Project Structure
sentiment-analysis-app/
│── backend/
│ ├── main.py # FastAPI backend
│ ├── model.pkl # Trained ML model
│ ├── vectorizer.pkl # TF-IDF vectorizer
│── templates/
│ ├── index.html # Input form
│ ├── result.html # Prediction result page
│── static/
│ ├── style.css # Styling
│── requirements.txt # Dependencies
│── README.md # Documentation
