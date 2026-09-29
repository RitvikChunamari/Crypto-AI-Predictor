import yfinance as yf
import pandas as pd
import ta
import warnings
import requests
import datetime
warnings.filterwarnings('ignore')

def fetch_binance_data(ticker, interval='1m', limit=50000):
    """
    Fetches historical high-frequency data from Binance public API.
    (100% free, no API key required)
    """
    # Convert Yahoo ticker (BTC-USD) to Binance ticker (BTCUSDT)
    symbol = ticker.replace('-', '')
    if not symbol.endswith('T'):
        symbol += 'T'
        
    url = "https://api.binance.us/api/v3/klines"
    klines = []
    end_time = int(datetime.datetime.now().timestamp() * 1000)
    
    print(f"Downloading ~{limit} rows of {interval} data from Binance...")
    
    while len(klines) < limit:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(1000, limit - len(klines)),
            'endTime': end_time
        }
        res = requests.get(url, params=params)
        data = res.json()
        
        if not data or type(data) == dict:
            break
            
        klines = data + klines
        end_time = data[0][0] - 1 # update end time to fetch previous chunk
        
    df = pd.DataFrame(klines, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
        
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def fetch_and_engineer_data(ticker, mode='daily'):
    """
    Fetches crypto data and adds technical indicators based on trading mode.
    """
    print(f"Fetching {mode.upper()} data for {ticker}...")
    
    if mode == 'daily':
        df = yf.download(ticker, period='5y', interval='1d')
    elif mode == 'hourly':
        df = yf.download(ticker, period='730d', interval='1h') # Yahoo max for 1h is 730 days
    elif mode == 'minute':
        # We download 50,000 minutes (approx 34 days) to keep training time reasonable (under 10 mins).
        # Downloading 5 years of minutes would crash a standard PC.
        df = fetch_binance_data(ticker, interval='1m', limit=50000)
    else:
        raise ValueError("Invalid mode selected.")
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
        
    df.dropna(inplace=True)
    
    # Squeeze to 1D Series
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    
    # Fetch Macro-Economic Data (Nasdaq and US Dollar Index)
    print("Fetching Macro-Economic Data (Nasdaq & DXY)...")
    if mode in ['daily', 'hourly']:
        period = '5y' if mode == 'daily' else '730d'
        interval = '1d' if mode == 'daily' else '1h'
        
        nasdaq = yf.download('^IXIC', period=period, interval=interval)['Close']
        dxy = yf.download('DX-Y.NYB', period=period, interval=interval)['Close']
        
        if mode == 'hourly':
            df.index = df.index.tz_convert('UTC')
            nasdaq.index = nasdaq.index.tz_convert('UTC').round('h')
            dxy.index = dxy.index.tz_convert('UTC').round('h')
            nasdaq = nasdaq[~nasdaq.index.duplicated(keep='first')]
            dxy = dxy[~dxy.index.duplicated(keep='first')]
        df['Nasdaq'] = nasdaq
        df['DXY'] = dxy
        
        # Stock markets close on weekends and nights, so we forward-fill the last known price
        df['Nasdaq'] = df['Nasdaq'].ffill().bfill()
        df['DXY'] = df['DXY'].ffill().bfill()
    else:
        # For minute-by-minute Binance data, we just use a dummy flatline or fetch daily and ffill
        # To keep it simple and avoid massive API alignment issues on 1m, we leave them as 0 for now
        df['Nasdaq'] = 0
        df['DXY'] = 0

    # 1. Moving Averages
    df['SMA_20'] = ta.trend.sma_indicator(close, window=20)
    df['SMA_50'] = ta.trend.sma_indicator(close, window=50)
    df['EMA_20'] = ta.trend.ema_indicator(close, window=20)
    
    # 2. Momentum Indicators
    df['RSI'] = ta.momentum.rsi(close, window=14)
    df['MACD'] = ta.trend.macd(close)
    df['MACD_Signal'] = ta.trend.macd_signal(close)
    
    # 3. Volatility Indicators
    df['BB_High'] = ta.volatility.bollinger_hband(close, window=20)
    df['BB_Low'] = ta.volatility.bollinger_lband(close, window=20)
    df['ATR'] = ta.volatility.average_true_range(high, low, close, window=14)
    
    # Drop rows with NaN values created by indicators
    df.dropna(inplace=True)
    
    # Clean column names (Squeeze multi-index if necessary from Yahoo)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    print(f"Data shape for {ticker} after feature engineering: {df.shape}")
    return df
