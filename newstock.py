# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# Load data
start = dt.datetime(2015, 1, 1)
end = dt.datetime(2023, 1, 1)

stock = 'TATAMOTORS.NS'
data = yf.download(stock, start=start, end=end)

# Feature Engineering: Add moving averages
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()
data['Volume'] = data['Volume'] / 1e6  # Normalize volume for scaling
data.dropna(inplace=True)  # Remove NaN values

# Define features
features = ['Close', 'MA20', 'MA50', 'Volume']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[features].values)

# Prepare training data
prediction_days = 100
x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x])  # Input features
    y_train.append(scaled_data[x, 0])  # Close price is the target

x_train, y_train = np.array(x_train), np.array(y_train)

# Build the LSTM model
model = Sequential()

# Bidirectional LSTM layers
model.add(Bidirectional(LSTM(units=100, return_sequences=True), input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(units=100, return_sequences=False)))
model.add(Dropout(0.3))
model.add(Dense(units=1))  # Predict the next closing price

model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(x_train, y_train, epochs=20, batch_size=32, callbacks=[early_stopping])

# Save the trained model


# Test the model
test_start = dt.datetime(2023, 1, 1)
test_end = dt.datetime(2024, 1, 1)

test_data = yf.download(stock, start=test_start, end=test_end)
test_data['MA20'] = test_data['Close'].rolling(window=20).mean()
test_data['MA50'] = test_data['Close'].rolling(window=50).mean()
test_data['Volume'] = test_data['Volume'] / 1e6  # Normalize volume for scaling
test_data.dropna(inplace=True)  # Remove NaN values

actual_prices = test_data['Close'].values

# Feature scaling for test data
total_dataset = pd.concat((data[features], test_data[features]), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = scaler.transform(model_inputs)

x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x])

x_test = np.array(x_test)

# Predictions
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(
    np.hstack([predicted_prices, np.zeros((predicted_prices.shape[0], len(features) - 1))])
)[:, 0]

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(actual_prices, color='black', label=f"Actual {stock} Price")
plt.plot(predicted_prices, color='green', label=f"Predicted {stock} Price")
plt.title(f"{stock} Share Price Prediction")
plt.xlabel('Time')
plt.ylabel(f"{stock} Share Price")
plt.legend()
plt.show()

# RMSE for LSTM
lstm_rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))


# Predict the next day
real_data = model_inputs[len(model_inputs) - prediction_days:]
real_data = np.array([real_data])

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(
    np.hstack([prediction, np.zeros((prediction.shape[0], len(features) - 1))])
)[:, 0]
print(f"LSTM RMSE: {lstm_rmse}")
print(f"Prediction for the next day: {prediction[0]}")
model.save('Latest_stock_price_model.keras')