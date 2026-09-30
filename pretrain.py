import os
from step1_data import fetch_and_engineer_data
from step2_preprocess import split_and_scale_data
from step3_model import build_hybrid_model, train_and_save_model

os.makedirs("models", exist_ok=True)

coins = ["BTC-USD", "ETH-USD", "SOL-USD"]
mode = "daily"

for ticker in coins:
    print(f"Pre-training {ticker}...")
    df = fetch_and_engineer_data(ticker, mode=mode)
    if len(df) > 100:
        X_train, y_train, X_test, y_test, f_scaler, t_scaler = split_and_scale_data(df)
        model = build_hybrid_model((X_train.shape[1], X_train.shape[2]))
        train_and_save_model(model, X_train, y_train, X_test, y_test, f"{ticker}_{mode}")
        print(f"Saved {ticker}_{mode}_model.keras")
