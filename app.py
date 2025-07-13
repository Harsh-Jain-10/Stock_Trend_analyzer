import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import io

# ============ PAGE CONFIG =============
st.set_page_config(page_title="📊 Stock Trend Analyzer", layout="wide")

# ============ SIDEBAR CONFIG ============
st.sidebar.title("🛠️ Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
forecast_days = st.sidebar.slider("Forecast Days", 1, 60, 30)
time_step = st.sidebar.slider("Time Step (for training)", 10, 100, 60)

# ============ HEADER =============
st.title("📈 Stock Trend Analyzer")
st.markdown(f"Analyze historical data, predict future prices, and visualize trends for **{ticker}**")

# ============ DATA FETCHING ============
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data.dropna()

try:
    stock_data = load_data(ticker, start_date, end_date)

    st.subheader(f"📃 Raw Data: {ticker}")
    st.dataframe(stock_data.tail())

    # ============ CORRELATION HEATMAP ============
    st.subheader("🔍 Correlation Heatmap")
    fig_corr, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(stock_data.corr().iloc[:5, :5], annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig_corr)

    # ============ FEATURE SCALING ============
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_data[['Close']])

    def create_sequences(data, time_step):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i-time_step:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)

    # ============ MODEL TRAINING ============
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1]), y_train)

    y_pred = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1]))
    predicted_prices = scaler.inverse_transform(y_pred.reshape(-1, 1))
    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # ============ METRICS ============
    st.subheader("📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("📉 RMSE", f"{np.sqrt(mean_squared_error(real_prices, predicted_prices)):.4f}")
    col2.metric("📈 R² Score", f"{r2_score(real_prices, predicted_prices):.4f}")
    col3.metric("🔧 MAE", f"{mean_squared_error(real_prices, predicted_prices):.4f}")

    # ============ PRICE PREDICTION CHART ============
    st.subheader("📈 Price Prediction")
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(y=real_prices.flatten(), name='Actual'))
    fig_pred.add_trace(go.Scatter(y=predicted_prices.flatten(), name='Predicted'))
    fig_pred.update_layout(title="Actual vs Predicted Prices", xaxis_title="Time", yaxis_title="Price")
    st.plotly_chart(fig_pred, use_container_width=True)

    # ============ FUTURE FORECAST ============
    last_sequence = list(scaled_data[-time_step:].flatten())
    future_predictions = []

    for _ in range(forecast_days):
        input_seq = np.array(last_sequence[-time_step:]).reshape(1, time_step)
        pred_scaled = model.predict(input_seq)
        pred_price = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        future_predictions.append(pred_price)
        last_sequence.append(pred_scaled[0])

    future_dates = [stock_data.index[-1] + datetime.timedelta(days=i) for i in range(1, forecast_days + 1)]

    st.subheader(f"🔮 {forecast_days}-Day Forecast")
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Close (USD)": future_predictions
    })
    st.dataframe(forecast_df.set_index("Date"))

    csv = forecast_df.to_csv(index=False)
    st.download_button("⬇️ Download Forecast CSV", data=csv, file_name=f"{ticker}_forecast.csv", mime='text/csv')

    # ============ FORECAST PLOT ============
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Historical Close'))
    fig_forecast.add_trace(go.Scatter(x=future_dates, y=future_predictions, name='Future Forecast', line=dict(color='orange')))
    fig_forecast.update_layout(title="30-Day Price Forecast", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # ============ MOVING AVERAGE ============
    st.subheader("📉 100-Day & 200-Day Moving Averages")
    stock_data['100_MA'] = stock_data['Close'].rolling(window=100).mean()
    stock_data['200_MA'] = stock_data['Close'].rolling(window=200).mean()

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Close'))
    fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['100_MA'], name='100-Day MA'))
    fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['200_MA'], name='200-Day MA'))
    fig_ma.update_layout(title="Moving Averages", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_ma, use_container_width=True)

    # ============ SUMMARY ============
    st.success(f"📌 Predicted next day closing price for {ticker}: **${future_predictions[0]:.2f}**")
    st.info(f"📅 Predicted closing price after {forecast_days} days: **${future_predictions[-1]:.2f}**")

except Exception as e:
    st.error(f"⚠️ Error: {e}")
