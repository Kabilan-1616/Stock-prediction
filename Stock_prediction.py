
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from datetime import timedelta, date

st.title("Stock Price Prediction")

ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., NVDA):", value="NVDA")
days = st.sidebar.slider("Days of historical data to use for modeling", 60, 365, 365)

st.sidebar.markdown("### Select Chart Date Range")
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp.today().date())
start_date = st.sidebar.date_input(
    "Start Date",
    value=end_date - timedelta(days=5 * 365),
    max_value=end_date
)

@st.cache_data(show_spinner=False)
def load_long_term_data(ticker):
    data = yf.Ticker(ticker).history(period="5y")
    return data

@st.cache_data(show_spinner=False)
def load_data(ticker, period_days):
    data = yf.Ticker(ticker).history(period=f"{period_days}d")
    data = data.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')
    return data

def add_indicators(df):
    df['SMA_20'] = df['Close'].rolling(20).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    gain_avg = gain.rolling(14).mean()
    loss_avg = loss.rolling(14).mean()
    rs = gain_avg / loss_avg
    df['RSI_14'] = 100 - (100 / (1 + rs))
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_12_26_9'] = ema_12 - ema_26
    df['MACD_S_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    df['MACD_H_12_26_9'] = df['MACD_12_26_9'] - df['MACD_S_12_26_9']
    return df

def prepare_targets(df):
    df['Target'] = df['Close'].shift(-1)
    df['Target_direction'] = (df['Target'] > df['Close']).astype(int)
    df['future_3d_return'] = (df['Close'].shift(-3) - df['Close']) / df['Close']
    df['Target_3d_up'] = (df['future_3d_return'] > 0).astype(int)
    df.dropna(inplace=True)
    return df

def train_models(X_train, y1, y2, y3):
    model1 = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model1.fit(X_train, y1)

    model2 = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    model2.fit(X_train, y2)

    model3 = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    model3.fit(X_train, y3)

    return model1, model2, model3

if ticker:
    long_term_data = load_long_term_data(ticker)

    if long_term_data.empty:
        st.error("No data found. Try another symbol.")
    else:
        filtered_data = long_term_data[
            (long_term_data.index.date >= start_date) & (long_term_data.index.date <= end_date)
        ]

        st.subheader(f"{ticker} Price from {start_date.strftime('%d/%m/%y')} to {end_date.strftime('%d/%m/%y')}")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(filtered_data.index, filtered_data['Close'], label='Close Price')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        st.divider()

        df = load_data(ticker, days)
        df = add_indicators(df)
        df = prepare_targets(df)

        features = ['SMA_20', 'RSI_14', 'MACD_12_26_9', 'MACD_H_12_26_9']
        X = df[features]

        split = int(len(df) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y1_train, y1_test = df['Target'][:split], df['Target'][split:]
        y2_train, y2_test = df['Target_direction'][:split], df['Target_direction'][split:]
        y3_train, y3_test = df['Target_3d_up'][:split], df['Target_3d_up'][split:]

        m1, m2, m3 = train_models(X_train, y1_train, y2_train, y3_train)

        pred_price = m1.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y1_test, pred_price))
        acc_1d = accuracy_score(y2_test, m2.predict(X_test))
        acc_3d = accuracy_score(y3_test, m3.predict(X_test))

        latest = X.iloc[-1:].copy()
        latest_date = df.index[-1]
        predicted_date = latest_date + timedelta(days=1)

        next_price = m1.predict(latest)[0]
        next_dir = m2.predict(latest)[0]
        next_3d = m3.predict(latest)[0]

        st.subheader(f"Predictions for {ticker} (Based on data up to {latest_date.strftime('%d/%m/%y')})")

        st.markdown(
            f"<h3 style='color:#0a5;'>Predicted next-day price (for {predicted_date.strftime('%d/%m/%y')}): "
            f"<strong>${next_price:.2f}</strong></h3>",
            unsafe_allow_html=True
        )

        st.write(f"Next-day direction: **{'Up' if next_dir == 1 else 'Down'}**")
        st.write(f"3-day direction: **{'Up' if next_3d == 1 else 'Down'}**")

        st.subheader("Model Performance")
        st.write(f"RMSE (price): {rmse:.2f}")
        st.write(f"Accuracy (1-day): {acc_1d:.2%}")
        st.write(f"Accuracy (3-day): {acc_3d:.2%}")

        st.subheader("Actual vs Predicted Prices")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(y1_test.index, y1_test, label='Actual')
        ax2.plot(y1_test.index, pred_price, label='Predicted')
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price")
        ax2.legend()
        st.pyplot(fig2)
