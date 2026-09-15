import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

st.set_page_config(page_title="Insurance Prediction", page_icon="🛡️", layout="centered")

st.title("🛡️ Health Insurance Purchase Prediction")
image_url = "https://i.pinimg.com/736x/b6/ca/86/b6ca86093ea14fc429f1d2eb9d19170b.jpg"
st.image(image_url)

# ---------- Load & Prepare Data ----------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/ankitmisk/UIT-data/refs/heads/main/Insurance.csv"
    df = pd.read_csv(url)
    df.drop("Customer_ID", axis=1, inplace=True)
    df['Previous_Insurance'] = df['Previous_Insurance'].map({'No': 0, 'Yes': 1})
    df['Insurance_Bought'] = df['Insurance_Bought'].map({'No': 0, 'Yes': 1})
    return df

df = load_data()

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.3
)

@st.cache_resource
def train_model():
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

model = train_model()
y_pred = model.predict(X_test)

# ---------- Sidebar: Model Info ----------
with st.sidebar:
    st.header("📊 Model Performance")
    st.metric("Training Accuracy", f"{model.score(X_train, y_train)*100:.2f}%")
    st.metric("Testing Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    if st.checkbox("Show Classification Report"):
        st.text(classification_report(y_test, y_pred))

    if st.checkbox("Show Raw Data Sample"):
        st.dataframe(df.head())

# ---------- Main: User Input Form ----------
st.subheader("🔍 Enter Customer Details")

user_input = []
with st.form("prediction_form"):
    for col in X.columns:
        min_v = float(X[col].min())
        max_v = float(X[col].max())

        # If column only has 0/1 values, show a dropdown instead of a slider
        unique_vals = sorted(X[col].unique())
        if set(unique_vals) <= {0, 1}:
            choice = st.selectbox(f"{col}", options=["No", "Yes"])
            value = 1.0 if choice == "Yes" else 0.0
        else:
            value = st.slider(
                f"{col} (range: {min_v} - {max_v})",
                min_value=min_v, max_value=max_v,
                value=(min_v + max_v) / 2
            )
        user_input.append(value)

    submitted = st.form_submit_button("Predict")

if submitted:
    prediction = model.predict([user_input])[0]
    proba = model.predict_proba([user_input])[0]

    if prediction == 0:
        st.error(f"😟 Prediction: **No Insurance** (Confidence: {proba[0]*100:.1f}%)")
    else:
        st.success(f"😬 Prediction: **Will Buy Insurance** (Confidence: {proba[1]*100:.1f}%)")
