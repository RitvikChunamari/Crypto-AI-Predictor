!pip install tensorflow
!pip install pandas
!pip install numpy
!pip install yfinance
!pip install math
!pip install scikit
!pip install matplotlib
!pip install openai
!pip install newspaper3k
!pip install ta
!pip install textblob
!pip install seaborn
!pip install datetime
!pip install pickle

# --- CELL ---

!pip3 install pandas_datareader

# --- CELL ---

import math
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tensorflow.keras.models import load_model
from textblob import TextBlob
import json
import openai
import ta
from newspaper import Article, Config, build
from dateutil.relativedelta import relativedelta
import pandas_datareader as pdr
yf.pdr_override()


%matplotlib inline
sns.set_style("whitegrid")

# --- CELL ---

import os
import openai
openai.organization = "org-YOUR_ORG_ID_HERE"
openai.api_key = "sk-YOUR_API_KEY_HERE"

# --- CELL ---

cryptos = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'USDT-USD']

# --- CELL ---

def fetch_crypto_data(crypto, period='10y', interval='1d'):
    data = yf.download(crypto, start='2010-07-17', end=datetime.now().strftime('%Y-%m-%d'), interval=interval)
    return data

# --- CELL ---

# Fetch the cryptocurrency data
dataframes = {}
for crypto in cryptos:
    dataframes[crypto] = fetch_crypto_data(crypto)

# --- CELL ---

def add_technical_indicators(df):
    # Add simple moving averages
    df['SMA20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()

    # Add exponential moving averages
    df['EMA20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['EMA50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()

    # Add Relative Strength Index (RSI)
    df['RSI14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()

    # Add Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_UPPER'] = bollinger.bollinger_hband()
    df['BB_LOWER'] = bollinger.bollinger_lband()

    return df

for crypto in cryptos:
    dataframes[crypto] = add_technical_indicators(dataframes[crypto])
    dataframes[crypto].dropna(inplace=True)

queries = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "LTC": "Litecoin",
    "USDT": "Tether"
}

# --- CELL ---

def fetch_news_sentiment(query, start_date, end_date, website="https://money.cnn.com"):
    # Set up the configuration for newspaper3k
    config = Config()
    config.fetch_images = False
    config.memoize_articles = False
    config.cache_dir = False

    # Scrape news articles from the website
    news_data = build(website, config=config)

    # Filter articles based on date range and query
    filtered_articles = []
    for article in news_data.articles:
        try:
            article.download()
            article.parse()
            article_date = datetime.strptime(article.publish_date.strftime('%Y-%m-%d'), '%Y-%m-%d')
            if start_date <= article.publish_date <= end_date and query.lower() in article.title.lower():
                filtered_articles.append(article)
        except Exception as e:
            pass

    # Calculate sentiment for each article
    sentiment_scores = []
    for article in filtered_articles:
        analysis = TextBlob(article.text)
        sentiment_scores.append(analysis.sentiment.polarity)

    # Calculate average sentiment
    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0

    return avg_sentiment

start_date = (datetime.now() - relativedelta(months=1)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

for crypto in cryptos:
    short_code = crypto.split('-')[0]
    query = queries[short_code]
    avg_sentiment = fetch_news_sentiment(query, start_date, end_date)
    dataframes[crypto]['sentiment'] = avg_sentiment

# --- CELL ---

def create_xy(data, look_back):
    x, y = [], []
    for i in range(look_back, len(data)):
        x.append(data[i - look_back:i, :])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

# --- CELL ---

def create_train_test_data(scaled_data):
    training_data_len = int(np.ceil(len(scaled_data) * 0.8))

    train_data = scaled_data[0:training_data_len, :]

    x_train, y_train = create_xy(train_data, 60)

    test_data = scaled_data[training_data_len - 60:, :]

    x_test, y_test = create_xy(test_data, 60)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test, training_data_len


# --- CELL ---

def create_train_test_split(df):
    # Scale only the 'Close' column
    data = df.filter(['Close']).values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create training and test sets
    x_train, y_train, x_test, y_test, training_data_len = create_train_test_data(scaled_data)

    return x_train, y_train, x_test, y_test, scaler, training_data_len

from keras.layers import Input, average
from keras.models import Model

def ensemble_models(models, model_input_shape):
    # Create a model input
    model_input = Input(shape=model_input_shape)
    
    # Collect the outputs of all models
    outputs = [model(model_input) for model in models]
    
    # Average the outputs
    avg = average(outputs)
    
    # Build a model using the same input and averaged outputs
    model_ensembled = Model(inputs=model_input, outputs=avg, name='ensemble')
    
    return model_ensembled

# Individual model architectures
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    return model

def build_gru_model(input_shape):
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=input_shape),
        GRU(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    return model

def build_bidirectional_lstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(50, return_sequences=True, input_shape=input_shape)),
        Bidirectional(LSTM(50, return_sequences=False)),
        Dense(25),
        Dense(1)
    ])
    return model

# Models dictionary
models = {
    'LSTM': lambda input_shape: build_lstm_model(input_shape),
    'GRU': lambda input_shape: build_gru_model(input_shape),
    'Bidirectional LSTM': lambda input_shape: build_bidirectional_lstm_model(input_shape),
    'Ensembled': lambda input_shape: ensemble_models(
        [build_lstm_model(input_shape), build_gru_model(input_shape), build_bidirectional_lstm_model(input_shape)],
        input_shape
    )
}


# --- CELL ---

rmse_scores = {}
for crypto in cryptos:
    x_train, y_train, x_test, y_test, scaler, training_data_len = create_train_test_split(dataframes[crypto])
    input_shape = (x_train.shape[1], x_train.shape[2])

    rmse_scores[crypto] = {}
    for model_name, model_fn in models.items():
        model = model_fn(input_shape)
        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, batch_size=1, epochs=5, verbose=0)

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        rmse_scores[crypto][model_name] = rmse
        
rmse_table = pd.DataFrame(rmse_scores)
print(rmse_table)


# --- CELL ---

def save_models_and_scores(crypto, trained_models, rmse_scores):
    model_dir = f'models/{crypto}'
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    for model_name, model in trained_models.items():
        model.save(f'{model_dir}/{model_name}.h5')
    
    with open(f'{model_dir}/rmse_scores.txt', 'w') as file:
        file.write("Model: RMSE\n")
        for model_name, rmse in rmse_scores.items():
            file.write(f'{model_name}: {rmse}\n')
    save_models_and_scores(crypto, trained_models[crypto], rmse_scores[crypto])

# --- CELL ---

import os

# --- CELL ---

def plot_actual_vs_predicted_prices(crypto, y_test, predictions, training_data_len):
    train = dataframes[crypto][:training_data_len]
    valid = dataframes[crypto][training_data_len:]
    valid['Predictions'] = predictions

    plt.figure(figsize=(16, 8))
    plt.title(f'{crypto} Actual vs Predicted Prices')
    plt.xlabel('Year', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

for crypto in cryptos:
    x_train, y_train, x_test, y_test, scaler, training_data_len = create_train_test_split(dataframes[crypto])
    input_shape = (x_train.shape[1], x_train.shape[2])
    model = models['Ensembled'](input_shape)  # Use the Ensembled model for plotting
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=5, verbose=0)

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    plot_actual_vs_predicted_prices(crypto, y_test, predictions[:, 0], training_data_len)


# --- CELL ---

def create_train_test_split(df):
    data = df.filter(['Close']).values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    training_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[0:training_data_len, :]

    x_train, y_train = create_xy(train_data, 60)

    test_data = scaled_data[training_data_len - 60:, :]
    x_test, y_test = create_xy(test_data, 60)

    return x_train, y_train, x_test, y_test, scaler, training_data_len

input_date = input("Enter a date in the format yyyy-mm-dd: ")
future_prices = {}

for crypto in cryptos:
    x_train, y_train, x_test, y_test, scaler, training_data_len = create_train_test_split(dataframes[crypto])
    input_shape = (x_train.shape[1], x_train.shape[2])
    model = models['Ensembled'](input_shape)  # Use the Ensembled model for future price prediction
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=5, verbose=0)

    future_price = predict_future_price(crypto, model, input_date, scaler)
    future_prices[crypto] = future_price

print("\nPredicted future prices:")
for crypto, future_price in future_prices.items():
    print(f"{crypto}: ${future_price:.2f}")


# --- CELL ---

!pip install lime
!pip install shap

# --- CELL ---

from lime import lime_tabular

def explain_prediction_with_lime(model, x_test, scaler, feature_names):
    # Reshape the input data
    x_test_2d = x_test.reshape(x_test.shape[0], -1)
    
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=x_test_2d,  # Use the entire x_test_2d as training data for LIME
        feature_names=feature_names, 
        class_names=['price'],
        mode='regression'
    )
    
    exp = explainer.explain_instance(
        x_test_2d[0],  # Use the first instance from the test set
        model.predict, 
        num_features=len(feature_names)
    )
    
    return exp

for crypto in cryptos:
    x_train, y_train, x_test, y_test, scaler, training_data_len = create_train_test_split(dataframes[crypto])
    input_shape = (x_train.shape[1], x_train.shape[2])
    model = models['Ensembled'](input_shape)
    
    # Adjust the feature names to account for multiple timesteps
    feature_names = [f"Price_day_{i}_feature_{j}" for i in range(x_train.shape[1]) for j in range(x_train.shape[2])]
    
    exp = explain_prediction_with_lime(model, x_test, scaler, feature_names)
    exp.show_in_notebook(show_table=True)


# --- CELL ---

import shap

def explain_prediction_with_shap(model, x_train, x_test):
    # Flatten the timesteps and features dimensions for SHAP
    x_train_flattened = x_train.reshape(x_train.shape[0], -1)
    x_test_flattened = x_test.reshape(x_test.shape[0], -1)
    
    # Transform the model's predict function to work with SHAP
    def f(x):
        # Reshape the data to its original shape before feeding to the model
        x_reshaped = x.reshape(x.shape[0], x_train.shape[1], x_train.shape[2])
        return model.predict(x_reshaped).flatten()

    # Create a SHAP explainer using KernelExplainer
    explainer = shap.KernelExplainer(f, x_train_flattened)
    
    # Calculate SHAP values for the flattened test set
    shap_values = explainer.shap_values(x_test_flattened)
    
    return shap_values

for crypto in cryptos:
    x_train, y_train, x_test, y_test, scaler, training_data_len = create_train_test_split(dataframes[crypto])
    input_shape = (x_train.shape[1], x_train.shape[2])
    model = models['Ensembled'](input_shape)
    
    shap_values = explain_prediction_with_shap(model, x_train, x_test)
    
    # Visualize the SHAP values for the first prediction
    # Note: The visualization will be for the flattened data
    shap.force_plot(explainer.expected_value, shap_values[0], x_test.reshape(x_test.shape[0], -1)[0])
