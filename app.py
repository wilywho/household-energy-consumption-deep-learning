import os
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained model
model = load_model('dnn_energy_consumption.keras')

# Initialize the scaler for inverse scaling
scaler = MinMaxScaler(feature_range=(0, 1))

# Function to calculate lag_1, lag_2, and rolling_avg_24
def calculate_features(data):
    # Assuming 'data' is a Pandas Series
    lag_1 = data.shift(1).iloc[-1]  # Last value from the previous month
    lag_2 = data.shift(2).iloc[-1]  # Value two months ago
    rolling_avg_24 = data.tail(24).mean()  # Rolling average of last 24 months (24 months = 2 years)

    return lag_1, lag_2, rolling_avg_24

# Streamlit UI
st.title("Energy Consumption Prediction")

# User input
monthly_usage = st.number_input("Enter your monthly energy consumption (in kWh):", min_value=0.0)

# When user inputs a value
if monthly_usage > 0:
    # Simulate the past consumption data (for the sake of the example)
    # This should ideally come from a larger dataset, but for this demo we'll generate random data.
    past_consumption_data = pd.Series(np.random.rand(100) * 10)  # Simulated past consumption data (100 months)

    # Calculate lag features and rolling average
    lag_1, lag_2, rolling_avg_24 = calculate_features(past_consumption_data)

    # Scale the input features
    input_data = np.array([[lag_1, lag_2, rolling_avg_24]])
    input_scaled = scaler.fit_transform(input_data)

    # Make a prediction using the trained model
    prediction = model.predict(input_scaled)

    # Rescale the prediction to the original scale
    predicted_value = scaler.inverse_transform(prediction)

    # Calculate the estimated cost (optional)
    cost_per_kWh = st.number_input("Enter the cost per kWh (in your local currency):", min_value=0.01)
    total_cost = predicted_value[0][0] * cost_per_kWh

    # Display results
    st.write(f"Estimated energy consumption for the next month: {predicted_value[0][0]:.2f} kWh")
    st.write(f"Estimated cost for the next month: {total_cost:.2f} (in your local currency)")
else:
    st.write("Please enter a valid monthly energy consumption value.")
