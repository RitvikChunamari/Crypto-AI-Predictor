# Institutional-Grade Crypto Price Predictor 🚀📈

An advanced, multi-timeframe cryptocurrency forecasting pipeline built with **Deep Learning (1D CNN + GRU)**, **Macro-Economic Correlations**, and an **Algorithmic Backtesting Engine**.

## 🧠 Overview
Unlike standard technical analysis bots that rely solely on lagging indicators, this AI engine is designed to predict cryptocurrency price action by analyzing a 12-dimensional feature set that includes global macro-economic health. 

It predicts future prices across three dynamic timeframes:
* **Daily (Swing Trading):** Analyzes 5 years of historical data to predict tomorrow's closing price.
* **Hourly (Day Trading):** Analyzes 2 years of intra-day data to predict the next hour.
* **Minute-by-Minute (Scalping):** Hooks into the Binance Public API to analyze millions of rows for high-frequency trading.

## ⚙️ Core Architecture

### 1. The Neural Network (Hybrid 1D CNN + GRU)
* Uses a **1D Convolutional Neural Network (CNN)** to extract spatial feature relationships (e.g., how the MACD interacts with the Bollinger Bands).
* Passes the feature maps into a **Gated Recurrent Unit (GRU)** to process the chronological sequence and identify long-term time-series dependencies.
* Heavily regularized with `Dropout(0.3)` and optimized using `EarlyStopping` to prevent overfitting.

### 2. The 12-Dimensional Feature Pipeline
Engineered 12 strict features to feed the neural network without data leakage:
* **Raw Action:** `Close`, `Volume`
* **Moving Averages:** `SMA_20`, `SMA_50`, `EMA_20`
* **Momentum & Volatility:** `RSI`, `MACD`, `ATR`, `Bollinger Bands (High/Low)`
* **Macro-Economics:** `Nasdaq (^IXIC)`, `US Dollar Index (DXY)`

### 3. Integrated Gradients (Explainability)
To prevent the AI from becoming a "black box", the pipeline includes native TensorFlow **Integrated Gradients**. It calculates the exact mathematical derivative of the model's activations to prove which features drove the prediction (e.g., proving that the US Dollar Index had a 12% impact on the model's decision making).

### 4. The Backtesting Simulator
Includes a custom-built backtesting engine that simulates real-world algorithmic trading. It feeds the AI a virtual `$10,000`, forces it to trade on unseen test data, and deducts a strict `0.1%` exchange fee on every single transaction to calculate true Net Profit against a baseline "Buy & Hold" strategy.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Data Engineering:** Pandas, NumPy, yfinance, Binance API
* **Technical Analysis:** `ta` library
* **Visualization:** Matplotlib

## 📈 Real-World Results
During a simulated 300-day bear market backtest on Bitcoin (where a standard Buy & Hold strategy lost **-7.83%** of its value), the AI successfully detected incoming crashes via Macro-Economic correlations, sold its positions to cash, and ended the period with a positive profit of **+0.97%**.

---
*Developed by Ritvik Chunamari.*
