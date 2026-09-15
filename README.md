# 🛡️ Health Insurance Purchase Prediction

A Machine Learning web app that predicts whether a customer is likely to purchase health insurance, built using **Logistic Regression** and deployed with **Streamlit**.

# AI/ML Workshop


## 🔗 Live Demo
[DEMO](https://insurance-prediction-ml-web-appgit-vyzbopodb6quwmcxgrikrg.streamlit.app/#health-insurance-purchase-prediction)

## 📌 Overview
This project uses customer data (age, previous insurance status, etc.) to train a Logistic Regression classifier that predicts whether a customer will buy insurance. The model is wrapped in an interactive Streamlit web interface where users can input customer details and get real-time predictions.

## ✨ Features
- Interactive web UI built with Streamlit
- Real-time prediction with confidence score
- Sidebar showing model performance (training/testing accuracy)
- Confusion matrix visualization
- Classification report and raw data preview (optional toggles)

## 🧠 Model
- **Algorithm:** Logistic Regression
- **Library:** scikit-learn
- **Data Split:** 70% training / 30% testing
- **Target Variable:** Insurance_Bought (Yes/No)

## 📂 Dataset
The dataset is loaded directly from a public CSV file containing customer details such as age, previous insurance status, and whether they purchased insurance.

## 🛠️ Tech Stack
- Python
- Streamlit
- Pandas & NumPy
- Scikit-learn
- Seaborn & Matplotlib

## 🚀 Run Locally

Clone the repository:
```bash
git clone https://github.com/yoshinoGrg/insurance-prediction-ml-web-app.git
cd insurance-prediction-ml-web-app
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the app:
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## ☁️ Deployment
This app is deployed on [Streamlit Community Cloud](https://share.streamlit.io).

## 📸 Preview
Enter customer details in the form, click **Predict**, and get an instant result on whether the customer is likely to buy insurance.

## 👤 Author
**Suraj Gurung**
- GitHub: [@yoshinoGrg](https://github.com/yoshinoGrg)

