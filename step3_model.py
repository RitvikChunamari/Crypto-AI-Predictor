import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Conv1D, MaxPooling1D, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def build_hybrid_model(input_shape):
    """
    Builds a 1D CNN + GRU Hybrid model for high efficiency and accuracy.
    input_shape: (time_steps, num_features)
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # 1D CNN acts as a rapid feature extractor for short-term volatility/spikes
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # GRU captures the long-term temporal trends (faster and more efficient than LSTM)
        GRU(units=50, return_sequences=True),
        Dropout(0.2),
        GRU(units=50, return_sequences=False),
        Dropout(0.2),
        
        Dense(units=25, activation='relu'),
        Dense(units=1) # Predict the target (Close price)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_save_model(model, X_train, y_train, X_test, y_test, ticker):
    """
    Trains the model with EarlyStopping to prevent overfitting, and saves the best model.
    """
    print(f"Training model for {ticker}...")
    
    # Create directory for models
    if not os.path.exists('models'):
        os.makedirs('models')
        
    model_path = f'models/{ticker}_lstm_model.keras'
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    print(f"Model saved to {model_path}")
    return model, history

if __name__ == "__main__":
    from step1_data import fetch_and_engineer_data
    from step2_preprocess import split_and_scale_data
    
    ticker = 'BTC-USD'
    df = fetch_and_engineer_data(ticker)
    X_train, y_train, X_test, y_test, f_scaler, t_scaler = split_and_scale_data(df)
    
    # Build model using shape [time_steps, features]
    model = build_hybrid_model((X_train.shape[1], X_train.shape[2]))
    model.summary()
    
    # Train
    train_and_save_model(model, X_train, y_train, X_test, y_test, ticker)
