import streamlit as st
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# Set page config FIRST
st.set_page_config(page_title="AI Trading Terminal", page_icon="⚡", layout="wide")

# Inject Custom CSS to perfectly match the Cinematic Portfolio
custom_css = """
<style>
    /* Global App Background */
    .stApp {
        background-color: #050508;
        color: #ffffff;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Courier New', Courier, monospace !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Neon Glow for the Main Title */
    .stApp h1 {
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
    }

    /* Streamlit Metric Styling (for Current Price, Predicted Price) */
    [data-testid="stMetricValue"] {
        color: #00E5FF !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.6) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Primary Button Styling */
    .stButton>button {
        background-color: #ffffff;
        color: #000000;
        border: none;
        border-radius: 0px;
        padding: 10px 24px;
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #00E5FF;
        color: #000000;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.5);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 10, 15, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Input Boxes */
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.05);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.2);
        font-family: 'Courier New', Courier, monospace;
        border-radius: 0px;
    }
    
    .stSelectbox>div>div>div {
        background-color: rgba(255, 255, 255, 0.05);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.2);
        font-family: 'Courier New', Courier, monospace;
        border-radius: 0px;
    }
    
    /* Top Progress Bar */
    .stProgress > div > div > div > div {
        background-color: #00E5FF !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Import backend functions AFTER page config
from step1_data import fetch_and_engineer_data
from step2_preprocess import split_and_scale_data
from step3_model import build_hybrid_model, train_and_save_model

st.title("AI TRADING TERMINAL ⚡")
st.markdown("<p style='color: rgba(255,255,255,0.5); font-size: 14px; letter-spacing: 1px; text-transform: uppercase;'>Live 12-Dimensional Deep Learning Inference Engine</p>", unsafe_allow_html=True)

st.sidebar.markdown("### CONFIGURATION")
ticker = st.sidebar.text_input("TICKER SYMBOL", value="BTC-USD")
mode = st.sidebar.selectbox("TIMEFRAME", ["daily", "hourly", "minute"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='font-size: 10px; color: rgba(255,255,255,0.3);'>Powered by Hybrid 1D CNN + GRU</p>", unsafe_allow_html=True)

if st.sidebar.button("RUN INFERENCE"):
    st.markdown(f"### SYSTEM INITIALIZING: `{ticker}` [{mode.upper()}]")
    
    # 1. Fetch Data
    with st.spinner("FETCHING LIVE DATA & MACRO FEATURES (NASDAQ, DXY)..."):
        try:
            df = fetch_and_engineer_data(ticker, mode=mode)
            if len(df) < 100:
                st.error(f"INSUFFICIENT HISTORICAL DATA FOR {ticker}.")
                st.stop()
        except Exception as e:
            st.error(f"DATA FETCH FAILED: {e}")
            st.stop()
            
    st.success(f"DATA ACQUIRED: {len(df)} TIMEFRAMES")
    
    # Display Chart with custom Plotly Theme matching portfolio
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    increasing_line_color='#00E5FF', decreasing_line_color='#FF0044')])
    
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Courier New, monospace", color="rgba(255,255,255,0.7)"),
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    )
    st.plotly_chart(fig, use_container_width=True)

    # 2. Process and Split
    with st.spinner("COMPILING 3D TENSORS..."):
        X_train, y_train, X_test, y_test, f_scaler, t_scaler = split_and_scale_data(df)
        
    # 3. Model Loading / Training
    model_path = f'models/{ticker}_{mode}_model.keras'
    if os.path.exists(model_path):
        st.info(f"LOADING PRE-TRAINED WEIGHTS...")
        model = load_model(model_path)
    else:
        st.warning(f"TRAINING FRESH NEURAL NETWORK...")
        with st.spinner("OPTIMIZING WEIGHTS..."):
            os.makedirs("models", exist_ok=True)
            model = build_hybrid_model((X_train.shape[1], X_train.shape[2]))
            model, history = train_and_save_model(model, X_train, y_train, X_test, y_test, f"{ticker}_{mode}")

    # 4. Make Live Prediction
    if mode == 'daily': timeframe = 'TOMORROW'
    elif mode == 'hourly': timeframe = 'NEXT HOUR'
    else: timeframe = 'NEXT MINUTE'
    
    st.markdown("---")
    st.markdown(f"### INFERENCE RESULT: {timeframe}")
    
    with st.spinner(f"CALCULATING MATHEMATICAL PROBABILITY..."):
        features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'ATR', 'BB_High', 'BB_Low', 'Nasdaq', 'DXY']
        recent_data = df[features].tail(60).values
        
        scaled_recent = f_scaler.transform(recent_data)
        X_live = np.reshape(scaled_recent, (1, 60, len(features)))
        
        predicted_scaled = model.predict(X_live, verbose=0)
        predicted_price = t_scaler.inverse_transform(predicted_scaled)[0][0]
        
        current_price = float(np.squeeze(df['Close'].values)[-1])
        diff = predicted_price - current_price
        pct_change = (diff / current_price) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("CURRENT PRICE", f"${current_price:,.2f}")
        col2.metric("PREDICTED PRICE", f"${predicted_price:,.2f}", f"{pct_change:,.2f}%")
        
        if pct_change > 1.5:
            action = "STRONG BUY"
            color = "#00FF7F"
        elif pct_change > 0:
            action = "BUY"
            color = "#00E5FF"
        elif pct_change < -1.5:
            action = "STRONG SELL"
            color = "#FF0044"
        else:
            action = "SELL"
            color = "#FFA500"
            
        col3.markdown(f"**RECOMMENDATION:** <br><span style='color:{color}; font-size: 24px; font-weight: bold;'>{action}</span>", unsafe_allow_html=True)
