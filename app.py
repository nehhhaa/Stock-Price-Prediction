import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import matplotlib.pyplot as plt


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="StockAI Predictor",
    page_icon="📈",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.title {
    font-size: 45px;
    font-weight: 700;
}

.subtitle {
    font-size: 18px;
    color: #666;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# TITLE
# ==========================================

st.markdown(
    '<div class="title">📈 StockAI Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning based Stock Price Prediction</div>',
    unsafe_allow_html=True
)

st.write("")


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("⚙️ Stock Settings")

ticker = st.sidebar.text_input(
    "Stock Symbol",
    "TCS.NS"
)

start_date = st.sidebar.date_input(
    "Start Date",
    pd.to_datetime("2015-01-01")
)

end_date = st.sidebar.date_input(
    "End Date",
    pd.Timestamp.today()
)

analyze = st.sidebar.button(
    "🚀 Analyze Stock"
)


# ==========================================
# LOAD SAVED MODEL
# ==========================================

try:

    model = joblib.load(
        "models/linear_regression.pkl"
    )

    model_info = joblib.load(
        "models/model_info.pkl"
    )

except FileNotFoundError:

    st.error(
        "❌ Saved model not found. Please run train_model.py first."
    )

    st.stop()


# ==========================================
# MAIN ANALYSIS
# ==========================================

if analyze:

    st.info("📡 Downloading stock data...")

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False
    )

    # Fix MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):

        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    if data.empty:

        st.error(
            "❌ No stock data found. Check the stock symbol."
        )

        st.stop()


    # ==========================================
    # HISTORICAL PRICE
    # ==========================================

    st.subheader("📊 Historical Closing Price")

    fig, ax = plt.subplots()

    ax.plot(
        data["Date"],
        data["Close"]
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price")

    st.pyplot(fig)


    # ==========================================
    # FEATURE ENGINEERING
    # ==========================================

    data["Previous_Close"] = (
        data["Close"].shift(1)
    )

    data["MA7"] = (
        data["Close"].rolling(7).mean()
    )

    data["MA21"] = (
        data["Close"].rolling(21).mean()
    )


    # Remove missing values

    clean_data = data.dropna().copy()


    # ==========================================
    # PREPARE FEATURES
    # ==========================================

    features = [
        "Previous_Close",
        "MA7",
        "MA21",
        "Volume"
    ]

    latest_data = clean_data.iloc[-1]

    X_latest = pd.DataFrame(
        [[
            latest_data["Previous_Close"],
            latest_data["MA7"],
            latest_data["MA21"],
            latest_data["Volume"]
        ]],
        columns=features
    )


    # ==========================================
    # PREDICT NEXT DAY
    # ==========================================

    prediction = model.predict(
        X_latest
    )[0]


    current_price = float(
        clean_data["Close"].iloc[-1]
    )

    predicted_change = (
        prediction - current_price
    )

    percentage_change = (
        predicted_change / current_price
    ) * 100


    # ==========================================
    # DISPLAY RESULTS
    # ==========================================

    st.subheader("🔮 Prediction")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Current Price",
            f"₹{current_price:,.2f}"
        )


    with col2:

        st.metric(
            "Predicted Next Close",
            f"₹{prediction:,.2f}"
        )


    with col3:

        st.metric(
            "Predicted Change",
            f"₹{predicted_change:,.2f}",
            f"{percentage_change:.2f}%"
        )


    # ==========================================
    # ACTUAL VS PREDICTED
    # ==========================================

    st.subheader(
        "📈 Actual vs Predicted Prices"
    )


    # Create predictions for test data

    split = int(
        len(clean_data) * 0.8
    )

    test_data = clean_data.iloc[split:].copy()

    X_test = test_data[features]

    y_actual = test_data["Close"]

    y_pred = model.predict(
        X_test
    )


    fig2, ax2 = plt.subplots()

    ax2.plot(
        test_data["Date"],
        y_actual,
        label="Actual Price"
    )

    ax2.plot(
        test_data["Date"],
        y_pred,
        label="Predicted Price"
    )

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")

    ax2.legend()

    st.pyplot(fig2)


    # ==========================================
    # MODEL PERFORMANCE
    # ==========================================

    st.subheader(
        "🤖 Model Performance"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "MAE",
            f"{model_info['mae']:.2f}"
        )


    with col2:

        st.metric(
            "RMSE",
            f"{model_info['rmse']:.2f}"
        )


    with col3:

        st.metric(
            "R² Score",
            f"{model_info['r2']:.4f}"
        )


    # ==========================================
    # MODEL INFORMATION
    # ==========================================

    st.subheader(
        "🧠 Model Details"
    )

    st.write(
        "**Algorithm:** Linear Regression"
    )

    st.write(
        "**Features used:** "
        "Previous Close, MA7, MA21, Volume"
    )

    st.write(
        "**Prediction:** Next Day Closing Price"
    )


    # ==========================================
    # DISCLAIMER
    # ==========================================

    st.warning(
        "⚠️ This project is for educational purposes "
        "only and is not financial advice."
    )


else:

    st.info(
        "👈 Enter a stock symbol and click "
        "**Analyze Stock** to begin."
    )