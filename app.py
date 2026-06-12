"""
Tesla Stock Price Prediction - Streamlit App
Run with: streamlit run app.py

Place this file in the same directory as:
 - best_overall_model.keras  (saved from the notebook)
 - scaler.pkl                (saved from the notebook)
 - TSLA.csv                  (optional, for showing recent history)
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

WINDOW_SIZE = 60

st.set_page_config(page_title="Tesla Stock Price Predictor", layout="wide")
st.title("📈 Tesla (TSLA) Stock Price Prediction")
st.markdown("Predict Tesla's next closing price using a trained **LSTM / SimpleRNN** deep learning model.")

@st.cache_resource
def load_artifacts():
    model = load_model("best_overall_model.keras")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv("TSLA.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

try:
    model, scaler = load_artifacts()
    df = load_data()

    st.subheader("Recent Closing Price History")
    st.line_chart(df.set_index('Date')['Adj Close'].tail(200))

    st.subheader("Predict Next Closing Price")
    st.write(f"Using the last **{WINDOW_SIZE} trading days** of Adj Close prices to predict the next closing price.")

    last_window = df['Adj Close'].values[-WINDOW_SIZE:].reshape(-1, 1)
    last_window_scaled = scaler.transform(last_window)
    X_input = last_window_scaled.reshape(1, WINDOW_SIZE, 1)

    if st.button("Predict Next Day Closing Price"):
        pred_scaled = model.predict(X_input, verbose=0)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        last_actual = df['Adj Close'].values[-1]
        change = pred_price - last_actual
        pct_change = (change / last_actual) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Last Close Price", f"${last_actual:.2f}")
        col2.metric("Predicted Next Close", f"${pred_price:.2f}", f"{pct_change:+.2f}%")
        col3.metric("Predicted Change", f"${change:+.2f}")

    st.subheader("Upload Custom Data (Optional)")
    uploaded = st.file_uploader("Upload a CSV with at least 60 rows of 'Adj Close' prices", type="csv")
    if uploaded is not None:
        custom_df = pd.read_csv(uploaded)
        if 'Adj Close' in custom_df.columns and len(custom_df) >= WINDOW_SIZE:
            window = custom_df['Adj Close'].values[-WINDOW_SIZE:].reshape(-1, 1)
            window_scaled = scaler.transform(window)
            X_custom = window_scaled.reshape(1, WINDOW_SIZE, 1)
            pred_scaled = model.predict(X_custom, verbose=0)
            pred_price = scaler.inverse_transform(pred_scaled)[0][0]
            st.success(f"Predicted next closing price: ${pred_price:.2f}")
        else:
            st.error(f"CSV must contain an 'Adj Close' column with at least {WINDOW_SIZE} rows.")

    st.markdown("---")
    st.caption("⚠️ Disclaimer: This prediction is for educational purposes only and should not be used for actual trading or investment decisions.")

except FileNotFoundError as e:
    st.error(f"Required file not found: {e}. Please ensure 'best_overall_model.keras', 'scaler.pkl', and 'TSLA.csv' are in the app directory.")
