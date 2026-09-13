import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. DOWNLOAD STOCK DATA
# ==========================================

print("Downloading TCS stock data...")

data = yf.download(
    "TCS.NS",
    start="2015-01-01",
    end="2026-09-01",
    auto_adjust=False
)


# ==========================================
# 2. FIX COLUMN FORMAT
# ==========================================

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data.reset_index()

print("Data downloaded successfully!")
print("Rows:", len(data))


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

data["Previous_Close"] = data["Close"].shift(1)

data["MA7"] = (
    data["Close"]
    .rolling(7)
    .mean()
)

data["MA21"] = (
    data["Close"]
    .rolling(21)
    .mean()
)

# Next day's closing price
data["Target"] = data["Close"].shift(-1)


# Remove missing values

data = data.dropna()


# ==========================================
# 4. SELECT FEATURES
# ==========================================

features = [
    "Previous_Close",
    "MA7",
    "MA21",
    "Volume"
]

X = data[features]

y = data["Target"]


# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 6. TRAIN LINEAR REGRESSION
# ==========================================

print("\nTraining Linear Regression...")

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# ==========================================
# 7. PREDICTIONS
# ==========================================

predictions = model.predict(
    X_test
)


# ==========================================
# 8. MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n================================")
print("       MODEL PERFORMANCE")
print("================================")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 4))


# ==========================================
# 9. CREATE MODELS FOLDER
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)


# ==========================================
# 10. SAVE MODEL
# ==========================================

model_path = "models/linear_regression.pkl"

joblib.dump(
    model,
    model_path
)


print("\nModel saved successfully!")
print("Location:", model_path)


# ==========================================
# 11. SAVE MODEL INFORMATION
# ==========================================

model_info = {
    "features": features,
    "mae": mae,
    "rmse": rmse,
    "r2": r2
}

joblib.dump(
    model_info,
    "models/model_info.pkl"
)


print("Model information saved!")

print("\nTraining completed successfully! 🎉")