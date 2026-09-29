import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, target, seq_length):
    """
    Creates sequences of length `seq_length` for time-series forecasting.
    data: The feature columns.
    target: The target column to predict (e.g., Close price).
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)


def split_and_scale_data(df, seq_length=60, train_split=0.8):
    """
    Splits the data chronologically and applies MinMaxScaling without data leakage.
    Returns scaled 3D sequences ready for an LSTM.
    """
    # Define features to use
    features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'ATR', 'BB_High', 'BB_Low', 'Nasdaq', 'DXY']
    
    # Extract data
    data_values = df[features].values
    target_values = df['Close'].values.reshape(-1, 1) # Close is what we predict
    
    # 1. Chronological Split (NO LEAKAGE)
    train_size = int(len(data_values) * train_split)
    
    train_data = data_values[:train_size]
    test_data = data_values[train_size:]
    
    train_target = target_values[:train_size]
    test_target = target_values[train_size:]
    
    # 2. Fit Scalers ONLY on training data
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_train_data = feature_scaler.fit_transform(train_data)
    scaled_test_data = feature_scaler.transform(test_data)
    
    scaled_train_target = target_scaler.fit_transform(train_target)
    scaled_test_target = target_scaler.transform(test_target)
    
    # 3. Create Sequences (3D arrays: [samples, time_steps, features])
    X_train, y_train = create_sequences(scaled_train_data, scaled_train_target, seq_length)
    X_test, y_test = create_sequences(scaled_test_data, scaled_test_target, seq_length)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test, feature_scaler, target_scaler

if __name__ == "__main__":
    from step1_data import fetch_and_engineer_data
    df = fetch_and_engineer_data('BTC-USD')
    X_train, y_train, X_test, y_test, f_scaler, t_scaler = split_and_scale_data(df)
