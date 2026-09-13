import yfinance as yf
import os

# Stock symbol
ticker = "TCS.NS"

# Download historical data
data = yf.download(
    ticker,
    start="2015-01-01",
    end="2026-09-01",
    auto_adjust=False
)

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save the data
data.to_csv("data/tcs_stock_data.csv")

print("Stock data downloaded successfully!")
print("Number of rows:", len(data))
print(data.head())