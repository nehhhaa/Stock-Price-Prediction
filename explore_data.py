import pandas as pd

# Load the stock data
data = pd.read_csv("data/tcs_stock_data.csv")

# Display first 5 rows
print("\n--- FIRST 5 ROWS ---")
print(data.head())

# Display column names
print("\n--- COLUMNS ---")
print(data.columns.tolist())

# Display dataset information
print("\n--- DATA INFORMATION ---")
print(data.info())

# Check missing values
print("\n--- MISSING VALUES ---")
print(data.isnull().sum())

# Check number of rows and columns
print("\n--- DATASET SHAPE ---")
print(data.shape)

# Basic statistics
print("\n--- STATISTICS ---")
print(data.describe())