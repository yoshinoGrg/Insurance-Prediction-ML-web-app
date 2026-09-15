import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

st.set_page_config(page_title="Insurance Prediction", page_icon="🛡️", layout="centered")

# ============================================================
#  THEME — custom CSS (Fraunces serif display + Inter body,
#  deep pine-teal hero, parchment card, amber accent)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* Overall app background */
.stApp {
    background: #F7F3EC;
}

/* Hero band */
.hero-band {
    background: linear-gradient(180deg, #0F2A2E 0%, #163B3F 100%);
    margin: -5rem -4rem 2rem -4rem;
    padding: 3.5rem 4rem 3rem 4rem;
    color: #F7F3EC;
    border-bottom: 3px solid #C98A3E;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.4rem;
    line-height: 1.15;
    margin: 0 0 0.6rem 0;
    color: #F7F3EC;
}
.hero-sub {
    font-size: 1rem;
    color: #B9CFC9;
    max-width: 480px;
    margin: 0;
}
.hero-icon {
    font-size: 2.4rem;
    margin-bottom: 0.8rem;
    display: block;
}

/* Section headers */
h3 {
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
    color: #0F2A2E !important;
}

/* Form card */
div[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #E4DDCB;
    border-radius: 4px;
    padding: 2rem 2rem 1.2rem 2rem;
}

/* Buttons */
.stButton button, div[data-testid="stFormSubmitButton"] button {
    background: #C98A3E;
    color: #0F2A2E;
    border: none;
    border-radius: 3px;
    font-weight: 600;
    padding: 0.6rem 1.6rem;
    transition: background 0.15s ease;
}
.stButton button:hover, div[data-testid="stFormSubmitButton"] button:hover {
    background: #B87A2E;
    color: #F7F3EC;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0F2A2E;
}
section[data-testid="stSidebar"] * {
    color: #F7F3EC !important;
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    font-family: 'Fraunces', serif !important;
    color: #F7F3EC !important;
}

/* Badge-style metric cards */
.badge-row { display: flex; gap: 0.7rem; margin-bottom: 1.2rem; }
.badge {
    flex: 1;
    background: #163B3F;
    border: 1px solid #2E6E5E;
    border-radius: 4px;
    padding: 0.8rem 0.6rem;
    text-align: center;
}
.badge-value {
    font-family: 'Fraunces', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: #C98A3E;
    display: block;
}
.badge-label {
    font-size: 0.72rem;
    color: #B9CFC9;
    letter-spacing: 0.02em;
}

/* Result banners */
.result-banner {
    border-radius: 4px;
    padding: 1.1rem 1.3rem;
    margin-top: 1rem;
    font-family: 'Inter', sans-serif;
}
.result-yes {
    background: #EAF3EE;
    border-left: 4px solid #2E6E5E;
    color: #1C4B4F;
}
.result-no {
    background: #F7EAE7;
    border-left: 4px solid #B94A3B;
    color: #7A2E23;
}
.result-title { font-weight: 600; font-size: 1.05rem; margin-bottom: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  HERO
# ============================================================
st.markdown("""
<div class="hero-band">
    <span class="hero-icon">🛡️</span>
    <p class="hero-title">Will this customer buy health insurance?</p>
    <p class="hero-sub">A Logistic Regression model reads a customer's profile and estimates
    how likely they are to purchase coverage — trained on real policy-decision data.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
#  DATA + MODEL
# ============================================================
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

# ============================================================
#  SIDEBAR — "Policy Report Card"
# ============================================================
with st.sidebar:
    st.markdown("### Model Report Card")

    train_acc = model.score(X_train, y_train) * 100
    test_acc = accuracy_score(y_test, y_pred) * 100

    st.markdown(f"""
    <div class="badge-row">
        <div class="badge">
            <span class="badge-value">{train_acc:.1f}%</span>
            <span class="badge-label">TRAINING ACCURACY</span>
        </div>
        <div class="badge">
            <span class="badge-value">{test_acc:.1f}%</span>
            <span class="badge-label">TESTING ACCURACY</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.2))
    fig.patch.set_facecolor('#0F2A2E')
    ax.set_facecolor('#0F2A2E')
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='YlOrBr',
        ax=ax, cbar=False, linewidths=1, linecolor='#0F2A2E',
        annot_kws={"color": "#0F2A2E", "fontweight": "bold"}
    )
    ax.set_xlabel("Predicted", color="#F7F3EC")
    ax.set_ylabel("Actual", color="#F7F3EC")
    ax.tick_params(colors="#F7F3EC")
    st.pyplot(fig)

    if st.checkbox("Show classification report"):
        st.text(classification_report(y_test, y_pred))

    if st.checkbox("Show raw data sample"):
        st.dataframe(df.head())

# ============================================================
#  MAIN — INPUT FORM
# ============================================================
st.markdown("### Enter customer details")

user_input = []
with st.form("prediction_form"):
    for col in X.columns:
        min_v = float(X[col].min())
        max_v = float(X[col].max())
        unique_vals = sorted(X[col].unique())

        if set(unique_vals) <= {0, 1}:
            choice = st.selectbox(f"{col.replace('_', ' ')}", options=["No", "Yes"])
            value = 1.0 if choice == "Yes" else 0.0
        else:
            value = st.slider(
                f"{col.replace('_', ' ')}  ({min_v:.0f} – {max_v:.0f})",
                min_value=min_v, max_value=max_v,
                value=(min_v + max_v) / 2
            )
        user_input.append(value)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([user_input], columns=X.columns)
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    if prediction == 0:
        st.markdown(f"""
        <div class="result-banner result-no">
            <div class="result-title">😟 Unlikely to buy insurance</div>
            Confidence: {proba[0]*100:.1f}%
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-banner result-yes">
            <div class="result-title">😬 Likely to buy insurance</div>
            Confidence: {proba[1]*100:.1f}%
        </div>
        """, unsafe_allow_html=True)
