import streamlit as st
import pandas as pd
import bz2
import pickle

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
    page_title="EEG Eye State Detection",
    layout="centered"
)

st.title("🧠 EEG Eye State Detection")
st.write("Prediction using a **pre-trained Random Forest model**")

# ---------------------------------
# Load trained RF model (.pbz2)
# ---------------------------------
@st.cache_resource
def load_model():
    with bz2.open("rf_model.pbz2", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ---------------------------------
# EEG Feature Names
# MUST MATCH training order
# ---------------------------------
feature_names = [
    'AF3','F7','F3','FC5','T7','P7','O1',
    'O2','P8','T8','FC6','F4','F8','AF4'
]

# ---------------------------------
# Sidebar inputs
# ---------------------------------
st.sidebar.header("🔧 EEG Channel Inputs")

user_input = {}

for feature in feature_names:
    user_input[feature] = st.sidebar.number_input(
        label=feature,
        value=0.0,
        format="%.4f"
    )

input_df = pd.DataFrame([user_input])

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("🔍 Predict Eye State"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    if prediction == 1:
        st.success("👁️ Eye State: **CLOSED**")
    else:
        st.success("👁️ Eye State: **OPEN**")

    st.subheader("📊 Prediction Probability")
    st.write(f"Open: **{probability[0]*100:.2f}%**")
    st.write(f"Closed: **{probability[1]*100:.2f}%**")
