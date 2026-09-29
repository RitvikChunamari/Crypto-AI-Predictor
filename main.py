from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Trading Terminal | Deep Learning Engine</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&display=swap');
            body { background-color: #050508; color: #ffffff; font-family: 'JetBrains Mono', monospace; margin: 0; overflow-x: hidden; }
            .neon-text { text-shadow: 0 0 10px rgba(0,229,255,0.5); }
            .neon-border { box-shadow: 0 0 20px rgba(0,229,255,0.1); }
            .grid-bg {
                background-image: linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
                background-size: 50px 50px;
            }
        </style>
    </head>
    <body class="min-h-screen flex flex-col p-4 md:p-10 grid-bg relative">
        <div class="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-[#00E5FF] to-transparent opacity-50"></div>
        
        <header class="flex flex-col md:flex-row justify-between items-start md:items-end mb-12 border-b border-white/10 pb-6 w-full max-w-7xl mx-auto z-10">
            <div>
                <h1 class="text-4xl md:text-6xl font-extrabold tracking-tighter mb-2">LIVE AI TERMINAL</h1>
                <p class="text-[#00E5FF] text-xs md:text-sm uppercase tracking-widest flex items-center gap-3">
                    <span class="w-2 h-2 rounded-full bg-[#00E5FF] animate-pulse"></span>
                    TENSORFLOW CLOUD INFERENCE ENGINE
                </p>
            </div>
            <div class="flex gap-3 mt-6 md:mt-0" id="coin-buttons">
                <button onclick="selectCoin('BTC-USD')" class="coin-btn bg-[#00E5FF] text-black px-6 py-2 text-sm tracking-widest shadow-[0_0_15px_rgba(0,229,255,0.4)] transition-all">BTC-USD</button>
                <button onclick="selectCoin('ETH-USD')" class="coin-btn bg-transparent text-white/50 border border-white/20 hover:border-[#00E5FF] hover:text-[#00E5FF] px-6 py-2 text-sm tracking-widest transition-all">ETH-USD</button>
                <button onclick="selectCoin('SOL-USD')" class="coin-btn bg-transparent text-white/50 border border-white/20 hover:border-[#00E5FF] hover:text-[#00E5FF] px-6 py-2 text-sm tracking-widest transition-all">SOL-USD</button>
            </div>
        </header>

        <main class="flex flex-col lg:flex-row gap-8 w-full max-w-7xl mx-auto flex-1 z-10">
            <!-- CHART -->
            <div class="flex-1 relative bg-black/50 border border-white/10 rounded-xl neon-border overflow-hidden p-6 flex flex-col min-h-[400px]">
                <div id="loader" class="absolute inset-0 flex flex-col items-center justify-center bg-black/80 z-20 transition-opacity duration-300">
                    <div class="w-12 h-12 border-2 border-white/10 border-t-[#00E5FF] rounded-full animate-spin mb-4"></div>
                    <div class="text-[#00E5FF] text-xs tracking-[0.3em] uppercase animate-pulse">Pinging Cloud API...</div>
                </div>
                
                <svg id="chart" class="absolute inset-0 w-full h-full overflow-visible py-12 px-4" preserveAspectRatio="none" viewBox="0 0 100 100">
                    <defs>
                        <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stop-color="rgba(0, 229, 255, 0.4)" />
                            <stop offset="100%" stop-color="rgba(0, 229, 255, 0)" />
                        </linearGradient>
                    </defs>
                    <path id="chartFill" fill="url(#chartGradient)" class="opacity-30 transition-all duration-1000" d=""></path>
                    <path id="chartLine" fill="none" stroke="#00E5FF" stroke-width="2" class="drop-shadow-[0_0_8px_rgba(0,229,255,0.8)] transition-all duration-1000" vector-effect="non-scaling-stroke" d=""></path>
                </svg>
                <div id="scanline" class="absolute top-0 bottom-0 w-[2px] bg-[#00E5FF] shadow-[0_0_20px_#00E5FF] opacity-0 z-10" style="left:0%"></div>
            </div>

            <!-- METRICS -->
            <div class="lg:w-[400px] flex flex-col gap-6">
                <div class="bg-black/50 border border-white/10 rounded-xl p-8 flex flex-col items-center justify-center text-center relative overflow-hidden">
                    <div class="absolute top-0 left-0 w-full h-full bg-[#00E5FF]/5"></div>
                    <span class="text-[10px] text-white/50 uppercase tracking-[0.2em] mb-3 z-10">AI PREDICTION SIGNAL</span>
                    <span id="ui-action" class="text-3xl font-bold tracking-tight z-10 text-white/30">AWAITING...</span>
                </div>
                
                <div class="grid grid-cols-2 gap-4">
                    <div class="bg-black/50 border border-white/10 rounded-xl p-6 flex flex-col justify-center">
                        <span class="text-[10px] text-white/50 uppercase tracking-widest mb-2">Confidence</span>
                        <span id="ui-conf" class="text-2xl text-white">--</span>
                    </div>
                    <div class="bg-black/50 border border-white/10 rounded-xl p-6 flex flex-col justify-center">
                        <span class="text-[10px] text-white/50 uppercase tracking-widest mb-2">Est. Move</span>
                        <span id="ui-pct" class="text-2xl text-white">--</span>
                    </div>
                    <div class="bg-black/50 border border-white/10 rounded-xl p-6 flex flex-col justify-center">
                        <span class="text-[10px] text-white/50 uppercase tracking-widest mb-2">Current</span>
                        <span id="ui-curr" class="text-2xl text-white/80">--</span>
                    </div>
                    <div class="bg-black/50 border border-white/10 rounded-xl p-6 flex flex-col justify-center">
                        <span class="text-[10px] text-white/50 uppercase tracking-widest mb-2">Target</span>
                        <span id="ui-targ" class="text-2xl text-white">--</span>
                    </div>
                </div>

                <button onclick="runAnimation()" id="run-btn" class="mt-auto w-full relative px-8 py-6 bg-transparent border border-[#00E5FF]/40 text-sm tracking-[0.2em] uppercase overflow-hidden cursor-pointer transition-all hover:border-[#00E5FF] group disabled:opacity-50">
                    <span class="relative z-10 font-bold text-[#00E5FF] transition-colors duration-300 group-hover:text-black">EXECUTE INFERENCE ANIMATION</span>
                    <div class="absolute inset-0 bg-[#00E5FF] translate-y-[100%] group-hover:translate-y-0 transition-transform duration-300 ease-out z-0"></div>
                </button>
            </div>
        </main>

        <script>
            let currentSymbol = 'BTC-USD';
            let chartData = [];

            async function selectCoin(coin) {
                currentSymbol = coin;
                document.querySelectorAll('.coin-btn').forEach(btn => {
                    if(btn.innerText === coin) {
                        btn.className = 'coin-btn bg-[#00E5FF] text-black px-6 py-2 text-sm tracking-widest shadow-[0_0_15px_rgba(0,229,255,0.4)] transition-all';
                    } else {
                        btn.className = 'coin-btn bg-transparent text-white/50 border border-white/20 hover:border-[#00E5FF] hover:text-[#00E5FF] px-6 py-2 text-sm tracking-widest transition-all';
                    }
                });
                fetchData();
            }

            async function fetchData() {
                document.getElementById('loader').style.opacity = '1';
                document.getElementById('loader').style.pointerEvents = 'auto';
                
                // Reset UI
                document.getElementById('ui-action').innerText = 'AWAITING...';
                document.getElementById('ui-action').className = 'text-3xl font-bold tracking-tight z-10 text-white/30';
                document.getElementById('ui-conf').innerText = '--';
                document.getElementById('ui-pct').innerText = '--';
                document.getElementById('ui-curr').innerText = '--';
                document.getElementById('ui-targ').innerText = '--';
                document.getElementById('chartFill').setAttribute('d', '');
                document.getElementById('chartLine').setAttribute('d', '');

                try {
                    const res = await fetch(`/predict?ticker=${currentSymbol}&mode=daily`);
                    const data = await res.json();
                    
                    if(data.historical_data) {
                        chartData = data.historical_data;
                        drawChart(chartData);
                        
                        document.getElementById('ui-action').innerText = data.action;
                        if(data.action.includes('BUY')) {
                            document.getElementById('ui-action').className = 'text-3xl font-bold tracking-tight z-10 text-green-400 drop-shadow-[0_0_10px_rgba(74,222,128,0.5)]';
                        } else {
                            document.getElementById('ui-action').className = 'text-3xl font-bold tracking-tight z-10 text-red-400 drop-shadow-[0_0_10px_rgba(248,113,113,0.5)]';
                        }
                        
                        document.getElementById('ui-conf').innerText = `${data.confidence}%`;
                        document.getElementById('ui-pct').innerText = `${data.pct_change > 0 ? '+' : ''}${data.pct_change.toFixed(2)}%`;
                        document.getElementById('ui-pct').className = `text-2xl ${data.pct_change > 0 ? 'text-green-400' : 'text-red-400'}`;
                        document.getElementById('ui-curr').innerText = `$${data.current_price.toLocaleString(undefined, {maximumFractionDigits:2})}`;
                        document.getElementById('ui-targ').innerText = `$${data.predicted_price.toLocaleString(undefined, {maximumFractionDigits:2})}`;
                    }
                } catch(e) {
                    console.error(e);
                }
                
                document.getElementById('loader').style.opacity = '0';
                document.getElementById('loader').style.pointerEvents = 'none';
            }

            function drawChart(dataArr) {
                const max = Math.max(...dataArr, 1);
                const min = Math.min(...dataArr, 0);
                const range = max - min || 1;
                
                let path = '';
                for(let i=0; i<dataArr.length; i++) {
                    const x = (i / (dataArr.length - 1)) * 100;
                    const y = 100 - (((dataArr[i] - min) / range) * 100);
                    path += `${i===0 ? 'M' : 'L'} ${x} ${y} `;
                }
                
                document.getElementById('chartLine').setAttribute('d', path);
                document.getElementById('chartFill').setAttribute('d', `${path} L 100 100 L 0 100 Z`);
            }

            function runAnimation() {
                if(chartData.length === 0) return;
                const sl = document.getElementById('scanline');
                const btn = document.getElementById('run-btn');
                btn.disabled = true;
                
                gsap.fromTo(sl, 
                    { left: '0%', opacity: 1 },
                    { left: '100%', duration: 2.5, ease: 'power2.inOut', onComplete: () => {
                        gsap.to(sl, { opacity: 0, duration: 0.5 });
                        btn.disabled = false;
                    }}
                );
            }

            // Initial load
            fetchData();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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
        
        # Get historical closing prices for the frontend chart (last 30 points)
        historical_closes = df['Close'].tail(30).values.tolist()
        historical_closes = [float(np.squeeze(x)) for x in historical_closes]

        return {
            "ticker": ticker,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "pct_change": pct_change,
            "action": action,
            "confidence": confidence,
            "historical_data": historical_closes
        }

    except Exception as e:
        return {"error": str(e)}
