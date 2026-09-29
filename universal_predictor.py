import os
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from step1_data import fetch_and_engineer_data
from step2_preprocess import split_and_scale_data
from step3_model import build_hybrid_model, train_and_save_model

def predict_any_crypto(ticker, mode):
    print(f"\n==============================================")
    print(f"INITIALIZING {mode.upper()} PIPELINE FOR {ticker}")
    print(f"==============================================\n")
    
    # 1. Fetch Data
    try:
        df = fetch_and_engineer_data(ticker, mode=mode)
    except Exception as e:
        print(f"[X] Could not fetch data for {ticker}. Error: {e}")
        return
        
    if len(df) < 100:
        print(f"[X] Not enough historical data for {ticker}. Need more data to train.")
        return

    # 2. Process and Split Data
    X_train, y_train, X_test, y_test, f_scaler, t_scaler = split_and_scale_data(df)
    
    # 3. Check if model exists, if not, train it
    model_path = f'models/{ticker}_{mode}_model.keras'
    if os.path.exists(model_path):
        print(f"\n[OK] Pre-trained {mode} model found for {ticker}! Loading it...")
        model = load_model(model_path)
    else:
        print(f"\n[!] No model found. Training a fresh Hybrid CNN+GRU {mode} model for {ticker}...")
        model = build_hybrid_model((X_train.shape[1], X_train.shape[2]))
        model, _ = train_and_save_model(model, X_train, y_train, X_test, y_test, f"{ticker}_{mode}")
        
    # 4. Make a Live Prediction
    if mode == 'daily': timeframe = 'TOMORROW'
    elif mode == 'hourly': timeframe = 'NEXT HOUR'
    else: timeframe = 'NEXT MINUTE'
        
    print(f"\n[>] Predicting the {timeframe} closing price for {ticker}...")
    
    # We need the most recent 60 intervals of data
    features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'ATR', 'BB_High', 'BB_Low', 'Nasdaq', 'DXY']
    recent_data = df[features].tail(60).values
    
    # Scale it using the feature scaler
    scaled_recent_data = f_scaler.transform(recent_data)
    
    # Reshape for the model [samples, time_steps, features]
    live_input = np.array([scaled_recent_data])
    
    # Predict
    scaled_prediction = model.predict(live_input, verbose=0)
    
    # Inverse transform to USD
    future_price = t_scaler.inverse_transform(scaled_prediction)[0][0]
    
    # Current Price
    current_price = float(np.squeeze(df['Close'].values)[-1])
    
    print(f"\n==============================================")
    print(f"{ticker} Live {mode.capitalize()} Report:")
    print(f"----------------------------------------------")
    print(f"Current Price (Now): ${current_price:,.2f}")
    print(f"Predicted Price ({timeframe}): ${future_price:,.2f}")
    
    difference = future_price - current_price
    percent_change = (difference / current_price) * 100
    
    if difference > 0:
        print(f"Forecast: UP by ${difference:,.2f} (+{percent_change:.2f}%)")
    else:
        print(f"Forecast: DOWN by ${abs(difference):,.2f} ({percent_change:.2f}%)")
    print(f"==============================================\n")

if __name__ == "__main__":
    target_coin = input("\nEnter a crypto ticker (e.g., SOL-USD, ETH-USD): ").upper()
    print("\nSelect Trading Mode:")
    print("1. Daily (Swing Trading - Predicts next day)")
    print("2. Hourly (Day Trading - Predicts next hour)")
    print("3. Minute (Scalping - Predicts next minute using Binance)")
    choice = input("Enter 1, 2, or 3: ")
    
    mode_map = {'1': 'daily', '2': 'hourly', '3': 'minute'}
    selected_mode = mode_map.get(choice, 'daily')
    
    predict_any_crypto(target_coin, selected_mode)
