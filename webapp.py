# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime

# Streamlit app title
st.title("Stock Price Predictor App")

# Input for stock ticker
stock = st.text_input("Enter the Stock ID (e.g., GOOG, AAPL, MSFT)", "GOOG")

# Load stock data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)
google_data = yf.download(stock, start=start, end=end)

# Check if data is loaded
if google_data.empty:
    st.error("No data found for the given stock. Please enter a valid stock ticker.")
    st.stop()

# Display stock data
st.subheader("Stock Data")
st.write(google_data)

# Load pre-trained model
try:
    model = load_model(r"C:\Users\HP\PycharmProjects\PythonProject\StockPrice\Latest_stock_price_model.keras")
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'Latest_stock_price_model.keras' is in the working directory.")
    st.stop()

# Add moving averages
google_data['MA_100'] = google_data['Close'].rolling(window=100).mean()
google_data['MA_200'] = google_data['Close'].rolling(window=200).mean()
google_data['MA_250'] = google_data['Close'].rolling(window=250).mean()

# Plot function
def plot_graph(data, ma=None, title="Stock Prices"):
    plt.figure(figsize=(15, 6))
    plt.plot(data['Close'], label="Close Price", color='blue')
    if ma is not None:
        for ma_period, ma_data in ma.items():
            plt.plot(ma_data, label=f"MA {ma_period} Days", linestyle='--')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)

# Plot moving averages
st.subheader("Original Close Price with Moving Averages")
plot_graph(google_data, ma={
    100: google_data['MA_100'],
    200: google_data['MA_200'],
    250: google_data['MA_250']
})

# Prepare data for prediction
google_data = google_data.dropna()  # Remove rows with missing values
features = google_data[['Open', 'High', 'Low', 'Close']].values  # Use 4 features

# Check if features have data
if features.shape[0] == 0:
    st.error("Insufficient data after removing missing values. Unable to proceed with predictions.")
    st.stop()

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features)

# Prepare x_data and y_data
x_data = []
y_data = []
sequence_length = 100

for i in range(sequence_length, len(scaled_data)):
    x_data.append(scaled_data[i - sequence_length:i])  # 100 time steps
    y_data.append(scaled_data[i, -1])  # Target: Last column (Close price)

x_data = np.array(x_data)
y_data = np.array(y_data)

# Check shapes
st.write(f"Shape of input data (x_data): {x_data.shape}")
st.write(f"Shape of target data (y_data): {y_data.shape}")

# Predictions
try:
    predictions = model.predict(x_data)
except ValueError as e:
    st.error(f"Error during model prediction: {e}")
    st.stop()

# Inverse transform predictions and actual values
inv_predictions = scaler.inverse_transform(
    np.concatenate((np.zeros((len(predictions), 3)), predictions), axis=1)
)[:, -1]
inv_y_test = scaler.inverse_transform(
    np.concatenate((np.zeros((len(y_data), 3)), y_data.reshape(-1, 1)), axis=1)
)[:, -1]

# Dataframe of predictions
start_index = len(google_data) - len(inv_predictions)
plotting_data = pd.DataFrame({
    'Original': inv_y_test.flatten(),
    'Predicted': inv_predictions.flatten()
}, index=google_data.index[start_index:])

# Display the dataframe with dates
st.subheader("Original vs Predicted Data Table with Dates")
st.dataframe(plotting_data)

# Plot original vs predicted values with years on the x-axis
st.subheader("Original vs Predicted Values (Years)")
plt.figure(figsize=(15, 6))
plt.plot(
    plotting_data.index, inv_y_test, label="Original Test Data", color='blue'
)
plt.plot(
    plotting_data.index, inv_predictions, label="Predicted Data", color='orange'
)
plt.title("Original vs Predicted Values (Years)")
plt.xlabel("Years")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)  # Rotate the year labels for better readability
st.pyplot(plt)

# Next-Day Prediction
st.subheader("Next-Day Prediction")
last_100_days = scaled_data[-sequence_length:]  # Last 100 days of scaled data
last_100_days = np.expand_dims(last_100_days, axis=0)  # Reshape for model input

# Predict the next day
next_day_prediction = model.predict(last_100_days)
next_day_price = scaler.inverse_transform(
    np.concatenate((np.zeros((1, 3)), next_day_prediction), axis=1)
)[:, -1][0]

# Display the prediction
st.write(f"The predicted stock price for the next day is: **${next_day_price:.2f}**")
