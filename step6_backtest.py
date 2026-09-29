import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from step1_data import fetch_and_engineer_data
from step2_preprocess import split_and_scale_data

def run_backtest(ticker, mode='daily', initial_capital=10000.0, trading_fee=0.001):
    print(f"\n==============================================")
    print(f"INITIALIZING BACKTEST ENGINE FOR {ticker}")
    print(f"==============================================\n")
    
    # 1. Fetch Data
    df = fetch_and_engineer_data(ticker, mode=mode)
    X_train, y_train, X_test, y_test, f_scaler, t_scaler = split_and_scale_data(df)
    
    # 2. Load Model
    model_path = f'models/{ticker}_{mode}_model.keras'
    fallback_path = f'models/{ticker}_lstm_model.keras'
    
    if os.path.exists(model_path):
        model = load_model(model_path)
    elif os.path.exists(fallback_path):
        model = load_model(fallback_path)
    else:
        print("X No model found. Please run the universal predictor first to train a model.")
        return
        
    # 3. Get Predictions for the Test Set
    print("\n[>] Running AI predictions on historical test data...")
    predictions = model.predict(X_test, verbose=0)
    
    predicted_prices = t_scaler.inverse_transform(predictions).flatten()
    actual_prices = t_scaler.inverse_transform(y_test).flatten()
    
    # 4. Simulate Trading
    print(f"\n[>] Simulating Trading with ${initial_capital:,.2f} starting capital...")
    balance_usd = initial_capital
    balance_crypto = 0.0
    
    # Trackers for plotting
    portfolio_history = []
    buy_and_hold_history = []
    
    # Buy and Hold metrics (If we just bought on day 1 and did nothing)
    initial_crypto_price = actual_prices[0]
    buy_and_hold_crypto = (initial_capital * (1 - trading_fee)) / initial_crypto_price
    
    trades_executed = 0
    
    # We loop through the test set. For each day 'i', the model predicts 'i+1'.
    # Note: We need the actual price of day 'i' to execute the trade for 'i+1'.
    # Because of how we sequenced the data, actual_prices[i] is the target day.
    # To avoid lookahead bias, if prediction > current_price, we buy.
    
    # Aligning the timeline:
    # sequence is 60 days. The input ends at day T. target is T+1.
    # We need the price at day T to decide if we buy for T+1.
    
    # Reconstruct the "current day" price array (day T)
    features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'ATR', 'BB_High', 'BB_Low', 'Nasdaq', 'DXY']
    close_idx = features.index('Close')
    
    # X_test shape: (samples, 60, 12). We want the 59th timestep of the 'Close' feature, unscaled.
    current_scaled_close = X_test[:, -1, close_idx].reshape(-1, 1)
    current_actual_prices = t_scaler.inverse_transform(current_scaled_close).flatten()
    
    for i in range(len(predicted_prices)):
        current_price = current_actual_prices[i]
        predicted_tomorrow = predicted_prices[i]
        actual_tomorrow = actual_prices[i] # What actually happened the next day
        
        # Strategy: If predicted price is higher than current price, go ALL IN.
        # If predicted price is lower, SELL EVERYTHING.
        
        if predicted_tomorrow > current_price:
            # BUY SIGNAL
            if balance_usd > 0:
                # We have USD, let's buy crypto
                crypto_bought = (balance_usd * (1 - trading_fee)) / current_price
                balance_crypto += crypto_bought
                balance_usd = 0.0
                trades_executed += 1
        else:
            # SELL SIGNAL
            if balance_crypto > 0:
                # We have Crypto, let's sell for USD
                usd_gained = (balance_crypto * current_price) * (1 - trading_fee)
                balance_usd += usd_gained
                balance_crypto = 0.0
                trades_executed += 1
                
        # Calculate daily net worth
        current_net_worth = balance_usd + (balance_crypto * actual_tomorrow)
        portfolio_history.append(current_net_worth)
        
        # Calculate buy and hold net worth
        bh_net_worth = buy_and_hold_crypto * actual_tomorrow
        buy_and_hold_history.append(bh_net_worth)
        
    final_net_worth = portfolio_history[-1]
    bh_final_net_worth = buy_and_hold_history[-1]
    
    ai_profit_pct = ((final_net_worth - initial_capital) / initial_capital) * 100
    bh_profit_pct = ((bh_final_net_worth - initial_capital) / initial_capital) * 100
    
    print("\n==============================================")
    print(f"BACKTEST RESULTS ({len(predicted_prices)} intervals)")
    print(f"----------------------------------------------")
    print(f"Initial Capital   : ${initial_capital:,.2f}")
    print(f"Total Trades Made : {trades_executed}")
    print(f"----------------------------------------------")
    print(f"AI Strategy Final : ${final_net_worth:,.2f} ({ai_profit_pct:+.2f}%)")
    print(f"Buy & Hold Final  : ${bh_final_net_worth:,.2f} ({bh_profit_pct:+.2f}%)")
    print(f"==============================================\n")
    
    # 5. Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_history, label='AI Trading Strategy', color='blue', linewidth=2)
    plt.plot(buy_and_hold_history, label='Buy & Hold Strategy', color='orange', linestyle='--')
    plt.title(f'{ticker} Backtest: AI vs Buy & Hold (Starting ${initial_capital:,.0f})')
    plt.xlabel('Time (Intervals)')
    plt.ylabel('Portfolio Value (USD)')
    plt.legend()
    plt.grid(True)
    
    plot_path = f'models/{ticker}_backtest.png'
    plt.savefig(plot_path)
    print(f"Backtest chart saved to {plot_path}")

if __name__ == "__main__":
    ticker = input("\nEnter ticker to backtest (e.g., BTC-USD): ").upper()
    print("Select Trading Mode:")
    print("1. Daily")
    print("2. Hourly")
    choice = input("Enter 1 or 2: ")
    mode = 'hourly' if choice == '2' else 'daily'
    
    run_backtest(ticker, mode=mode)
