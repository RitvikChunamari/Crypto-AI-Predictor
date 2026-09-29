from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
from step1_data import fetch_and_engineer_data
from step2_preprocess import split_and_scale_data
from tensorflow.keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Crypto AI Backend is Running"}

@app.get("/predict")
def predict_crypto(ticker: str = "BTC-USD", mode: str = "daily"):
    try:
        # Fetch data
        df = fetch_and_engineer_data(ticker, mode=mode)
        if len(df) < 100:
            return {"error": "Not enough data"}

        # Process
        X_train, y_train, X_test, y_test, f_scaler, t_scaler = split_and_scale_data(df)

        # Load Model
        model_path = f'models/{ticker}_{mode}_lstm_model.keras'
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            return {"error": "Model not pre-trained for this ticker. Use BTC-USD, ETH-USD, or SOL-USD."}

        # Predict
        features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'ATR', 'BB_High', 'BB_Low', 'Nasdaq', 'DXY']
        recent_data = df[features].tail(60).values
        scaled_recent = f_scaler.transform(recent_data)
        X_live = np.reshape(scaled_recent, (1, 60, len(features)))
        
        predicted_scaled = model.predict(X_live, verbose=0)
        predicted_price = float(t_scaler.inverse_transform(predicted_scaled)[0][0])
        
        current_price = float(np.squeeze(df['Close'].values)[-1])
        diff = predicted_price - current_price
        pct_change = (diff / current_price) * 100
        
        if pct_change > 1.5:
            action = "STRONG BUY (BULLISH)"
        elif pct_change > 0:
            action = "BUY (BULLISH)"
        elif pct_change < -1.5:
            action = "STRONG SELL (BEARISH)"
        else:
            action = "SELL (BEARISH)"

        # Calculate a pseudo-confidence score based on momentum vs prediction
        confidence = round(min(99.9, max(50.0, 75.0 + abs(pct_change) * 5)), 1)

        return {
            "ticker": ticker,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "pct_change": pct_change,
            "action": action,
            "confidence": confidence, "historical_data": [float(np.squeeze(x)) for x in df['Close'].tail(30).values.tolist()]
        }

    except Exception as e:
        return {"error": str(e)}
