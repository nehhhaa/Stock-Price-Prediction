# 📈 Stock Price Prediction Using Machine Learning

A machine learning project that predicts the next-day closing price of a stock using historical stock market data.

## 🚀 Project Overview

This project uses historical TCS stock data and machine learning techniques to predict the next day's closing price.

The project includes:

- Data collection using Yahoo Finance
- Data preprocessing
- Feature engineering
- Linear Regression
- Random Forest comparison
- Model evaluation
- Model saving using Joblib
- Interactive Streamlit dashboard

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- YFinance
- Joblib
- Streamlit

## 🧠 Machine Learning Features

The model uses:

1. Previous Closing Price
2. 7-Day Moving Average
3. 21-Day Moving Average
4. Trading Volume

### Target

The target variable is the **next day's closing price**.

## 🤖 Models Tested

### Linear Regression

MAE: 51.19  
RMSE: 69.61  
R² Score: 0.9903

### Random Forest

MAE: 107.57  
RMSE: 154.10  
R² Score: 0.9525

Based on the evaluation results, **Linear Regression** was selected as the final model.

> Note: R² score is a statistical evaluation metric and should not be interpreted as prediction accuracy.

## 📊 Dashboard

The Streamlit dashboard provides:

- Historical stock price visualization
- Current closing price
- Predicted next-day closing price
- Predicted price change
- Actual vs predicted price graph
- Model performance metrics

## 📁 Project Structure

```text
Stock_Price_Prediction/
│
├── data/
│   └── tcs_stock_data.csv
│
├── models/
│   ├── linear_regression.pkl
│   └── model_info.pkl
│
├── app.py
├── train_model.py
├── download_data.py
├── explore_data.py
├── requirements.txt
├── README.md
└── .gitignore