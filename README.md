# Stock-Price-Predictor-using-LSTM
README
Stock Price Predictor
The Stock Price Predictor is a machine learning model designed to forecast stock prices based on historical data. It leverages deep learning techniques to capture complex patterns in stock price movements, aiding in more informed decision-making.

Project Overview
This project develops a Sequential Neural Network model to predict stock prices. The system includes hyperparameter tuning using GridSearchCV to optimize the model's performance.

The pipeline involves:

Data preprocessing to handle missing values and scale features.
Neural network training with various hyperparameters to achieve the best results.
Evaluation using multiple metrics, including MAPE, MAE, and RMSE.
Data Background
Dataset: Historical stock price data, including features such as opening price, closing price, high, low, and volume.
Training and Testing Split: Data is split into 80% for training and 20% for testing.
Run Locally
To run this project on your local system, ensure you have the following dependencies installed:

bash
Copy code
pip install pandas
pip install numpy
pip install scikit-learn
pip install tensorflow
pip install matplotlib
Project Structure
1. Data Preprocessing
Scaling: Used MinMaxScaler from scikit-learn to scale the features between 0 and 1.
Train-Test Split: Split the dataset into training and testing sets (80:20 ratio).
2. Model Architecture
Developed a Sequential Neural Network with:
Input layer matching the feature shape.
Dense layers with ReLU activation.
Output layer for predicting stock price.
Configured the model with the Adam optimizer and Mean Squared Error (MSE) as the loss function.
3. Hyperparameter Tuning
Used GridSearchCV to identify the best combination of:
Number of layers.
Number of neurons per layer.
Batch size.
Learning rate.
4. Performance Metrics
Evaluated the model using:
Mean Absolute Percentage Error (MAPE)
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
5. Visualization
Plotted training and validation loss to observe the model's performance over epochs.
Compared actual vs. predicted stock prices for evaluation.
Technologies and Libraries Used
Data Processing:

pandas: For handling stock price data.
numpy: For numerical computations.
Machine Learning:

scikit-learn: For scaling data, splitting datasets, and GridSearchCV.
TensorFlow/Keras: For building and training the neural network.
Visualization:

matplotlib: For plotting results like loss curves and price predictions.
Project Performance
Accuracy Metrics:

Achieved low MAPE (e.g., 5%), indicating high prediction accuracy.
MAE and RMSE were minimized, confirming model reliability.
Feature Importance: Demonstrated the significance of specific features like opening price and volume in predicting closing prices.

This project serves as a foundational tool for stock price prediction, with the potential for further enhancement using more advanced architectures like LSTMs or additional features.
