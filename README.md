# Stock Price Predictor using LSTM

## Stock Price Predictor
The **Stock Price Predictor** is a machine learning model designed to forecast stock prices based on historical data. It leverages deep learning techniques to capture complex patterns in stock price movements, aiding in more informed decision-making.

---

## Project Overview
This project develops a **Sequential Neural Network** model to predict stock prices. The system includes hyperparameter tuning using **GridSearchCV** to optimize the model's performance.

The pipeline involves:
- Data preprocessing to handle missing values and scale features.
- Neural network training with various hyperparameters to achieve the best results.
- Evaluation using multiple metrics, including **MAPE**, **MAE**, and **RMSE**.

---

## Data Background
- **Dataset**: Historical stock price data, including features such as opening price, closing price, high, low, and volume.
- **Training and Testing Split**: Data is split into 80% for training and 20% for testing.

---

## Run Locally
To run this project on your local system, ensure you have the following dependencies installed:

```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install tensorflow
pip install matplotlib
