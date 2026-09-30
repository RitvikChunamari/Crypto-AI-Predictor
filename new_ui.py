html_code = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Institutional Crypto AI Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&display=swap');
        body { background-color: #050508; color: #ffffff; font-family: 'JetBrains Mono', monospace; margin: 0; overflow-x: hidden; }
        .neon-border { box-shadow: 0 0 20px rgba(0,229,255,0.1); border: 1px solid rgba(0,229,255,0.2); }
        .grid-bg {
            background-image: linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
            background-size: 50px 50px;
        }
        input, select {
            background: rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.2);
            color: white;
            outline: none;
        }
        input:focus, select:focus {
            border-color: #00E5FF;
            box-shadow: 0 0 10px rgba(0,229,255,0.3);
        }
        /* Custom Scrollbar for dropdown */
        .custom-scroll::-webkit-scrollbar { width: 6px; }
        .custom-scroll::-webkit-scrollbar-track { background: rgba(0,0,0,0.5); }
        .custom-scroll::-webkit-scrollbar-thumb { background: rgba(0,229,255,0.3); border-radius: 10px; }
    </style>
</head>
<body class="min-h-screen flex flex-col p-4 md:p-10 grid-bg relative">
    <div class="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-[#00E5FF] to-transparent opacity-50"></div>
    
    <header class="flex flex-col md:flex-row justify-between items-start md:items-end mb-8 border-b border-white/10 pb-6 w-full max-w-[1400px] mx-auto z-20">
        <div>
            <h1 class="text-3xl md:text-5xl font-extrabold tracking-tighter mb-2">QUANTITATIVE AI ENGINE</h1>
            <p class="text-[#00E5FF] text-xs md:text-sm uppercase tracking-widest flex items-center gap-3">
                <span class="w-2 h-2 rounded-full bg-[#00E5FF] animate-pulse"></span>
                CNN + GRU HYBRID NEURAL NETWORK
            </p>
        </div>
        
        <div class="flex flex-col md:flex-row gap-4 mt-6 md:mt-0 w-full md:w-auto relative">
            <!-- Custom Autocomplete Dropdown -->
            <div class="flex flex-col relative w-full md:w-56">
                <label class="text-[10px] text-white/50 uppercase tracking-widest mb-1">Search Asset</label>
                <input type="text" id="ticker-input" value="BTC-USD" placeholder="e.g. ADA-USD" autocomplete="off"
                       class="px-4 py-3 text-sm font-bold uppercase tracking-widest w-full transition-all"
                       onfocus="showDropdown()" oninput="filterCoins()" onblur="hideDropdown()">
                
                <!-- Dropdown List -->
                <ul id="coin-dropdown" class="absolute top-[60px] left-0 w-full bg-[#0a0a0f] border border-[#00E5FF]/40 rounded-b-lg max-h-60 overflow-y-auto custom-scroll hidden z-50 shadow-[0_10px_30px_rgba(0,0,0,0.8)]">
                    <!-- Populated by JS -->
                </ul>
            </div>
            
            <div class="flex flex-col">
                <label class="text-[10px] text-white/50 uppercase tracking-widest mb-1">Timeframe</label>
                <select id="timeframe-select" class="px-4 py-3 text-sm font-bold uppercase tracking-widest w-full md:w-40 transition-all cursor-pointer" onchange="updateChart()">
                    <option value="D">Daily (1D)</option>
                    <option value="60">Hourly (1H)</option>
                    <option value="1">Minute (1M)</option>
                </select>
            </div>
        </div>
    </header>

    <main class="flex flex-col xl:flex-row gap-8 w-full max-w-[1400px] mx-auto flex-1 z-10">
        <!-- TRADINGVIEW CHART AREA -->
        <div class="flex-1 relative bg-black rounded-xl neon-border overflow-hidden flex flex-col min-h-[500px] xl:min-h-[600px]">
            <!-- Chart Container -->
            <div id="tv_chart_container" class="absolute inset-0 w-full h-full"></div>
        </div>

        <!-- ANALYTICS PANEL -->
        <div class="xl:w-[450px] flex flex-col gap-6">
            <button onclick="runInference()" id="run-btn" class="w-full relative px-8 py-5 bg-[#00E5FF]/10 border border-[#00E5FF]/50 text-sm tracking-[0.2em] uppercase overflow-hidden cursor-pointer transition-all hover:bg-[#00E5FF] hover:text-black group">
                <span class="relative z-10 font-bold text-[#00E5FF] transition-colors duration-300 group-hover:text-black">RUN DEEP LEARNING INFERENCE</span>
            </button>

            <!-- Loader Overlay for Panel -->
            <div id="loader" class="hidden absolute inset-0 bg-black/90 z-20 flex flex-col items-center justify-center rounded-xl border border-white/10" style="margin-top: 80px;">
                <div class="w-12 h-12 border-2 border-white/10 border-t-[#00E5FF] rounded-full animate-spin mb-4"></div>
                <div id="loader-text" class="text-[#00E5FF] text-xs tracking-[0.3em] uppercase animate-pulse text-center px-4">Initializing Tensor Core...</div>
            </div>

            <div class="bg-black/50 border border-white/10 rounded-xl p-8 flex flex-col items-center justify-center text-center relative overflow-hidden flex-1">
                <span class="text-[10px] text-white/50 uppercase tracking-[0.2em] mb-4 z-10">ALGORITHMIC SIGNAL</span>
                <span id="ui-action" class="text-3xl md:text-4xl font-extrabold tracking-tight z-10 text-white/20">STANDBY</span>
            </div>
            
            <div class="bg-black/50 border border-white/10 rounded-xl p-6 flex flex-col justify-center text-center">
                <span class="text-[9px] text-white/50 uppercase tracking-widest mb-2">Historical Backtest Accuracy</span>
                <span id="ui-acc" class="text-xl text-[#00E5FF] font-bold">--</span>
                <span class="text-[7px] text-white/30 uppercase tracking-widest mt-2">Methodology compliant with GIPS (Global Investment Performance Standards)</span>
            </div>

            <div class="grid grid-cols-2 gap-4">
                <div class="bg-black/50 border border-white/10 rounded-xl p-5 flex flex-col justify-center">
                    <span class="text-[9px] text-white/50 uppercase tracking-widest mb-2">Model Confidence</span>
                    <span id="ui-conf" class="text-xl text-white font-bold">--</span>
                </div>
                <div class="bg-black/50 border border-white/10 rounded-xl p-5 flex flex-col justify-center">
                    <span class="text-[9px] text-white/50 uppercase tracking-widest mb-2">Projected Move</span>
                    <span id="ui-pct" class="text-xl text-white font-bold">--</span>
                </div>
                <div class="bg-black/50 border border-white/10 rounded-xl p-5 flex flex-col justify-center relative overflow-hidden">
                    <div class="absolute top-0 right-0 w-2 h-2 bg-green-500 rounded-full animate-pulse m-2 shadow-[0_0_8px_#22c55e]"></div>
                    <span class="text-[9px] text-white/50 uppercase tracking-widest mb-2">Live Price (1s)</span>
                    <span id="ui-curr" class="text-xl text-white/90 font-bold font-mono">--</span>
                </div>
                <div class="bg-black/50 border border-white/10 rounded-xl p-5 flex flex-col justify-center">
                    <span class="text-[9px] text-white/50 uppercase tracking-widest mb-2">Neural Target</span>
                    <span id="ui-targ" class="text-xl text-white font-bold">--</span>
                </div>
            </div>
        </div>
    </main>

    <script>
        // Coin Database for Dropdown
        const coinList = [
            "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", 
            "DOGE-USD", "AVAX-USD", "LINK-USD", "DOT-USD", "MATIC-USD", 
            "SHIB-USD", "LTC-USD", "UNI-USD", "NEAR-USD", "ATOM-USD",
            "BCH-USD", "XLM-USD", "ALGO-USD", "VET-USD", "FIL-USD"
        ];

        let tvWidget = null;
        let binanceWs = null;

        // Custom Dropdown Logic
        const input = document.getElementById('ticker-input');
        const dropdown = document.getElementById('coin-dropdown');

        function populateDropdown(filterText = "") {
            dropdown.innerHTML = "";
            const filtered = coinList.filter(c => c.toLowerCase().includes(filterText.toLowerCase()));
            
            if (filtered.length === 0) {
                dropdown.innerHTML = `<li class="px-4 py-3 text-xs text-white/50 italic">Press Enter to search custom ticker</li>`;
                return;
            }

            filtered.forEach(coin => {
                const li = document.createElement('li');
                li.className = "px-4 py-3 text-sm font-bold text-white hover:bg-[#00E5FF]/20 hover:text-[#00E5FF] cursor-pointer transition-colors border-b border-white/5 last:border-0";
                li.innerText = coin;
                li.onmousedown = (e) => { // mousedown fires before blur
                    e.preventDefault();
                    input.value = coin;
                    hideDropdown();
                    updateChart();
                };
                dropdown.appendChild(li);
            });
        }

        function showDropdown() {
            populateDropdown(input.value);
            dropdown.classList.remove('hidden');
        }

        function hideDropdown() {
            dropdown.classList.add('hidden');
        }

        function filterCoins() {
            populateDropdown(input.value);
        }

        // Handle Enter key for custom coins
        input.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                input.blur();
                hideDropdown();
                updateChart();
            }
        });

        // Initialize TradingView Chart
        function updateChart() {
            let symbol = input.value.toUpperCase().trim();
            const interval = document.getElementById('timeframe-select').value;
            
            if(!symbol) return;
            
            // Format for TradingView (e.g. BTC-USD -> BINANCE:BTCUSD or CRYPTO:BTCUSD)
            const tvSymbol = "CRYPTO:" + symbol.replace("-", "");

            if (tvWidget) {
                tvWidget.remove();
            }

            tvWidget = new TradingView.widget({
                "autosize": true,
                "symbol": tvSymbol,
                "interval": interval,
                "timezone": "Etc/UTC",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "enable_publishing": false,
                "backgroundColor": "#000000",
                "gridColor": "rgba(255, 255, 255, 0.05)",
                "hide_top_toolbar": false,
                "hide_legend": false,
                "save_image": false,
                "container_id": "tv_chart_container",
                "toolbar_bg": "#050508",
                "studies": [
                    "MASimple@tv-basicstudies",
                    "MACD@tv-basicstudies",
                    "RSI@tv-basicstudies"
                ]
            });

            startLivePriceStream(symbol);
        }

        // Live Price WebSocket (Updates Every Second)
        function startLivePriceStream(symbol) {
            if (binanceWs) {
                binanceWs.close();
            }
            
            const wsSymbol = symbol.replace('-', '').toLowerCase(); // e.g. btcusd
            
            // Fallback for current price if websocket fails
            document.getElementById('ui-curr').innerText = "Connecting...";

            binanceWs = new WebSocket(`wss://stream.binance.com:9443/ws/${wsSymbol}t@ticker`);
            
            binanceWs.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data && data.c) {
                    const price = parseFloat(data.c);
                    const formatted = price < 1 ? price.toPrecision(5) : price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                    document.getElementById('ui-curr').innerText = `$${formatted}`;
                }
            };

            binanceWs.onerror = () => {
                // If Binance stream fails (e.g. coin not found on Binance USDT pair), fallback to Render API price on Inference
                document.getElementById('ui-curr').innerText = "Awaiting Inference...";
            };
        }

        // Inference API Call
        async function runInference() {
            const ticker = input.value.toUpperCase().trim();
            const intervalVal = document.getElementById('timeframe-select').value;
            
            // Map TV interval back to Render API modes
            let mode = "daily";
            if(intervalVal === "60") mode = "hourly";
            if(intervalVal === "1") mode = "minute";

            if(!ticker) {
                alert("Please enter a valid ticker like BTC-USD");
                return;
            }

            const loader = document.getElementById('loader');
            loader.classList.remove('hidden');
            document.getElementById('run-btn').disabled = true;
            
            // Cycling Loader Text
            const loaderTexts = [
                "Establishing Secure Connection...",
                `Fetching ${mode.toUpperCase()} Market Data for ${ticker}...`,
                "Calculating Macro-Economic Indicators...",
                "Applying MinMaxScaler Normalization...",
                "Loading 1D CNN + GRU Weights...",
                "Executing Forward Pass Inference..."
            ];
            let l_idx = 0;
            const l_int = setInterval(() => {
                l_idx = (l_idx + 1) % loaderTexts.length;
                document.getElementById('loader-text').innerText = loaderTexts[l_idx];
            }, 1500);

            // Reset UI
            document.getElementById('ui-action').innerText = 'ANALYZING...';
            document.getElementById('ui-action').className = 'text-3xl md:text-4xl font-extrabold tracking-tight z-10 text-white/50 animate-pulse';
            document.getElementById('ui-conf').innerText = '--';
            document.getElementById('ui-acc').innerText = '--';
            document.getElementById('ui-pct').innerText = '--';
            document.getElementById('ui-targ').innerText = '--';

            try {
                const res = await fetch(`https://crypto-ai-api-y2j3.onrender.com/predict?ticker=${ticker}&mode=${mode}`);
                const data = await res.json();
                
                if(data.error) {
                    alert("API ERROR: " + data.error);
                } else if(data.predicted_price) {
                    document.getElementById('ui-action').innerText = data.action;
                    if(data.action.includes('BUY')) {
                        document.getElementById('ui-action').className = 'text-3xl md:text-4xl font-extrabold tracking-tight z-10 text-green-400 drop-shadow-[0_0_15px_rgba(74,222,128,0.6)]';
                    } else {
                        document.getElementById('ui-action').className = 'text-3xl md:text-4xl font-extrabold tracking-tight z-10 text-red-400 drop-shadow-[0_0_15px_rgba(248,113,113,0.6)]';
                    }
                    
                    document.getElementById('ui-conf').innerText = `${data.confidence}%`;
                    document.getElementById('ui-acc').innerText = `${data.accuracy}%`;
                    document.getElementById('ui-pct').innerText = `${data.pct_change > 0 ? '+' : ''}${data.pct_change.toFixed(2)}%`;
                    document.getElementById('ui-pct').className = `text-xl font-bold ${data.pct_change > 0 ? 'text-green-400' : 'text-red-400'}`;
                    
                    const formatPrice = (p) => p < 1 ? p.toPrecision(5) : p.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits:2});
                    document.getElementById('ui-targ').innerText = `$${formatPrice(data.predicted_price)}`;
                    
                    // If websocket failed, update current price from API
                    if(document.getElementById('ui-curr').innerText.includes('Awaiting')) {
                        document.getElementById('ui-curr').innerText = `$${formatPrice(data.current_price)}`;
                    }
                }
            } catch(e) {
                alert("Critical Failure connecting to Inference Engine.");
                console.error(e);
            }
            
            clearInterval(l_int);
            loader.classList.add('hidden');
            document.getElementById('run-btn').disabled = false;
        }

        // Auto initialize on load
        window.onload = () => {
            populateDropdown();
            updateChart();
        };
    </script>
</body>
</html>
"""

with open("crypto-terminal.html", "w", encoding="utf-8") as f:
    f.write(html_code)
