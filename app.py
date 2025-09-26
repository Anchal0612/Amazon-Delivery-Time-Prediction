import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load trained model
model = pickle.load(open("rf_model.pkl", "rb"))

# Streamlit Page Config
st.set_page_config(page_title="Amazon Delivery Predictor", page_icon="📦", layout="wide")

# Sidebar Branding
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/679/679922.png", width=120)
st.sidebar.title("Amazon Delivery Time App")
st.sidebar.markdown("🚀 Predict delivery time based on multiple factors.")

st.title("📦 Amazon Delivery Time Prediction")
st.markdown("### Enter details below and get instant predictions ⏱️")

# User inputs in columns
col1, col2 = st.columns(2)

with col1:
    agent_age = st.number_input("Agent Age", min_value=18, max_value=60, value=30)
    agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.5)
    distance = st.number_input("Distance (km)", min_value=0.0, max_value=100.0, value=5.0)
    order_hour = st.slider("Order Hour (0-23)", 0, 23, 12)

with col2:
    weather = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy", "Fog", "Stormy", "Windy"])
    traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam", "Very High"])
    vehicle = st.selectbox("Vehicle", ["motorcycle", "car", "scooter", "bicycle"])
    area = st.selectbox("Area", ["Urban", "Metropolitian", "Semi-Urban", "Rural"])
    category = st.selectbox("Category", ["Electronics", "Clothing", "Food", "Books", "Other"])

# Encoding categorical variables
encoding_dict = {
    "Weather": {"Sunny":0,"Rainy":1,"Cloudy":2,"Fog":3,"Stormy":4,"Windy":5},
    "Traffic": {"Low":0,"Medium":1,"High":2,"Jam":3,"Very High":4},
    "Vehicle": {"motorcycle":0,"car":1,"scooter":2,"bicycle":3},
    "Area": {"Urban":0,"Metropolitian":1,"Semi-Urban":2,"Rural":3},
    "Category": {"Electronics":0,"Clothing":1,"Food":2,"Books":3,"Other":4}
}

input_data = np.array([[agent_age, agent_rating, distance, order_hour,
                        encoding_dict["Weather"][weather],
                        encoding_dict["Traffic"][traffic],
                        encoding_dict["Vehicle"][vehicle],
                        encoding_dict["Area"][area],
                        encoding_dict["Category"][category]]])

# Prediction + History
if "history" not in st.session_state:
    st.session_state["history"] = []

if st.button("🚚 Predict Delivery Time"):
    prediction = model.predict(input_data)[0]
    st.success(f"✅ Estimated Delivery Time: **{round(prediction,2)} hours**")
    st.session_state["history"].append(prediction)

# Show Prediction History Chart
if st.session_state["history"]:
    st.subheader("📊 Prediction History")
    fig, ax = plt.subplots()
    ax.plot(st.session_state["history"], marker='o')
    ax.set_title("Delivery Time Predictions Over Session")
    ax.set_ylabel("Hours")
    ax.set_xlabel("Prediction Count")
    st.pyplot(fig)


