import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

def evaluate_and_plot(ticker):
    from step1_data import fetch_and_engineer_data
    from step2_preprocess import split_and_scale_data
    
    print(f"Loading data and model for {ticker}...")
    df = fetch_and_engineer_data(ticker)
    X_train, y_train, X_test, y_test, f_scaler, t_scaler = split_and_scale_data(df)
    
    model_path = f'models/{ticker}_lstm_model.keras'
    model = load_model(model_path)
    
    # Predict
    print("Making predictions...")
    predictions = model.predict(X_test)
    
    # Inverse transform to get actual USD values
    y_test_inv = t_scaler.inverse_transform(y_test)
    predictions_inv = t_scaler.inverse_transform(predictions)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
    print(f"\n================================")
    print(f"RMSE for {ticker}: ${rmse:.2f}")
    print(f"================================\n")
    
    # Aligning index for plotting
    train_size = int(len(df) * 0.8)
    seq_length = 60
    
    valid = df.iloc[train_size + seq_length:].copy()
    valid['Predictions'] = predictions_inv
    
    train = df.iloc[:train_size + seq_length]
    
    # Plotting
    plt.figure(figsize=(16, 8))
    plt.title(f'{ticker} Multivariate LSTM Price Prediction')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Close Price USD ($)', fontsize=14)
    plt.plot(train.index, train['Close'], label='Train Data')
    plt.plot(valid.index, valid['Close'], label='Actual Price (Test Data)')
    plt.plot(valid.index, valid['Predictions'], label='Predicted Price')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plot_path = f'models/{ticker}_prediction_plot.png'
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
if __name__ == "__main__":
    evaluate_and_plot('BTC-USD')
