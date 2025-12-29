import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# App Title
# -----------------------------
st.set_page_config(page_title="EEG Eye State Detection", layout="centered")

st.title("🧠 EEG Eye State Detection")
st.write("Predict whether **Eyes are Open or Closed** using EEG signals")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("EEG Eye State.csv")
    return df

df = load_data()

# -----------------------------
# Prepare Data
# -----------------------------
X = df.drop("eyeDetection", axis=1)
y = df["eyeDetection"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Train Random Forest Model
# -----------------------------
@st.cache_resource
def train_model():
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

model = train_model()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🔧 EEG Channel Inputs")

user_input = {}

for col in X.columns:
    user_input[col] = st.sidebar.number_input(
        f"{col}",
        value=float(X[col].mean())
    )

input_df = pd.DataFrame([user_input])

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Eye State"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    if prediction == 1:
        st.success("👁️ Eye State: **CLOSED**")
    else:
        st.success("👁️ Eye State: **OPEN**")

    st.write("### Prediction Probabilities")
    st.write(f"Open: {probability[0]*100:.2f}%")
    st.write(f"Closed: {probability[1]*100:.2f}%")

# -----------------------------
# Model Accuracy
# -----------------------------
accuracy = model.score(X_test, y_test)
st.write("### 📊 Model Accuracy")
st.write(f"Accuracy: **{accuracy*100:.2f}%**")
