import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt

# Streamlit Page Setup
st.set_page_config(page_title="🌡️ Temperature Forecast", layout="centered")
st.title("🌡️ Next Day Temperature Predictor")

# Load model and scaler
model = load_model("model.h5", compile=False)
model.compile(optimizer='adam', loss=MeanSquaredError())
scaler = joblib.load("scaler.pkl")

# Load dataset
df = pd.read_csv("daily_minimum_temps.csv", parse_dates=['Date'], index_col='Date')
df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
df = df.dropna()

# Scale data
data_scaled = scaler.transform(df['Temp'].values.reshape(-1, 1))
seq_length = 30

# Create sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, seq_length)

# Train-test split
split_index = int(len(X) * 0.8)
X_test = X[split_index:]
y_test = y[split_index:]

# Make predictions
y_pred_scaled = model.predict(X_test)
y_pred_scaled = np.clip(y_pred_scaled, 0, 1)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_actual = scaler.inverse_transform(y_test)

# Plot results
st.subheader("📉 Actual vs Predicted Temperatures")
fig, ax = plt.subplots()
ax.plot(y_test_actual, label='Actual Temp', color='blue')
ax.plot(y_pred, label='Predicted Temp', color='orange')
ax.set_xlabel('Time Steps')
ax.set_ylabel('Temperature (°C)')
ax.legend()
st.pyplot(fig)

# Predict next day's temperature
last_sequence = data_scaled[-seq_length:].reshape(1, seq_length, 1)
next_temp_scaled = model.predict(last_sequence)
next_temp_scaled = np.clip(next_temp_scaled, 0, 1)
next_temp = scaler.inverse_transform(next_temp_scaled)

st.subheader("🔮 Predicted Temperature for Next Day:")
st.success(f"{next_temp[0][0]:.2f} °C")
