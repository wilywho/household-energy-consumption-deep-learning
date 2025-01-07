import os
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained model
model = load_model('dnn_energy_consumption.keras')

# Initialize scalers
input_scaler = MinMaxScaler(feature_range=(0, 1))
output_scaler = MinMaxScaler(feature_range=(0, 1))

# Simulate historical data for scaler fitting (replace this with your actual data)
historical_data = np.random.rand(100, 3) * 100
input_scaler.fit(historical_data)

# Fit output scaler based on actual energy consumption range
output_scaler.fit(np.array([[historical_data[:, 0].min()], [historical_data[:, 0].max()]]))

# Function to calculate lag_1, lag_2, and rolling_avg_24
def calculate_features(data, new_value):
    data = pd.concat([data, pd.Series([new_value])], ignore_index=True)
    lag_1 = data.iloc[-2] if len(data) >= 2 else new_value  # Last value before the current month
    lag_2 = data.iloc[-3] if len(data) >= 3 else new_value  # Value two months ago
    rolling_avg_24 = data.iloc[-24:].mean() if len(data) >= 24 else data.mean()
    return lag_1, lag_2, rolling_avg_24

# Streamlit UI
st.title("Energy Consumption Prediction")
st.markdown("##### AI Engineer Bootcamp by Skillacademy")
st.markdown("###### Author: Farhan Wily")
st.write("""
This tool predicts your energy consumption for the next month based on historical data.
You will need to input your current monthly energy usage and the cost per kWh. 
Follow the steps below to get started:
""")

# Simulate historical energy consumption data
past_consumption_data = pd.Series(np.random.rand(100) * 100)  # Simulated past consumption data (100 months)

# User input
st.write("Press 'Enter' after you input the data")
monthly_usage = st.number_input("Enter your monthly energy consumption (in kWh):", min_value=0.0)
cost_per_kWh = st.number_input("Enter the cost per kWh (in your local currency):", min_value=0.01)

if st.button("Predict"):
    if monthly_usage > 0:
        # Calculate lag features and rolling average
        lag_1, lag_2, rolling_avg_24 = calculate_features(past_consumption_data, monthly_usage)

        # Scale the input features
        input_data = np.array([[lag_1, lag_2, rolling_avg_24]])
        input_scaled = input_scaler.transform(input_data)

        # Make a prediction using the trained model
        prediction = model.predict(input_scaled)

        # Rescale the prediction to the original scale
        predicted_value = output_scaler.inverse_transform(prediction)

        # Calculate the estimated cost
        total_cost = predicted_value[0][0] * cost_per_kWh

        # Display results
        st.write(f"Estimated energy consumption for the next month: **{predicted_value[0][0]:.2f} kWh**")
        st.write(f"Estimated cost for the next month: **{total_cost:.2f}** (in your local currency)")
    else:
        st.write("Please enter a valid monthly energy consumption value.")


st.write("This application was developed using a **Deep Neural Network (DNN) Model**")
