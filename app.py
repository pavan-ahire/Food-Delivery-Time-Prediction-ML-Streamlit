import streamlit as st
import numpy as np
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Delivery Time Prediction",
    page_icon="🚚",
    layout="centered"
)

# --------------------------------------------------
# Load Model & Scaler
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("delivery_time_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

# --------------------------------------------------
# Title & Description
# --------------------------------------------------
st.title("🚚 Food Delivery Time Prediction")
st.markdown(
    """
    Predict the **estimated delivery time**  
    using a trained **Machine Learning model based on Historical data**.
    """
)

st.divider()

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
st.subheader("📥 Enter Order Details")

distance = st.number_input("📍 Distance (km)", 0.1, 100.0, 5.0)
weather = st.selectbox("🌦 Weather Condition", ["Clear", "Rainy", "Foggy", "Stormy"])
traffic = st.selectbox("🚦 Traffic Level", ["Low", "Medium", "High"])
time_of_day = st.selectbox("🕒 Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
prep_time = st.number_input("🍳 Preparation Time (minutes)", 1, 120, 15)
vehicle = st.selectbox("🏍 Delivery Vehicle", ["Bike", "Scooter", "Car"])
experience = st.number_input("🧑‍💼 Delivery Person Experience (years)", 0, 20, 2)

# --------------------------------------------------
# Manual Encoding (MUST match training)
# --------------------------------------------------
weather_map = {"Clear": 0, "Rainy": 1, "Foggy": 2, "Stormy": 3}
traffic_map = {"Low": 0, "Medium": 1, "High": 2}
time_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
vehicle_map = {"Bike": 0, "Scooter": 1, "Car": 2}

input_data = np.array([[  
    distance,
    weather_map[weather],
    traffic_map[traffic],
    time_map[time_of_day],
    prep_time,
    vehicle_map[vehicle],
    experience
]])

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.divider()

if st.button("🔮 Predict Delivery Time"):

    # ✅ SCALE INPUT (THIS WAS MISSING)
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]  # prediction in minutes

    # Convert minutes to hrs/min/sec
    total_seconds = int(prediction * 60)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours > 0:
        st.success(f"🕒 Estimated Delivery Time: **{hours} hr {minutes} min {seconds} sec**")
    else:
        st.success(f"🕒 Estimated Delivery Time: **{minutes} min {seconds} sec**")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption("Developed by Pavan Ahire")
st.caption("📊 Machine Learning Powered | Streamlit Application")
