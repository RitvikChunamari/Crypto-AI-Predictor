import re

with open("pro_ui.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix WebSocket in pro_ui.py
old_ws = """        function startLivePriceStream(symbol) {
            if (binanceWs) binanceWs.close();
            const wsSymbol = symbol.replace('-', '').toLowerCase();
            document.getElementById('ui-curr').innerText = "Loading...";

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
                document.getElementById('ui-curr').innerText = "Awaiting...";
            };
        }"""

new_ws = """        function startLivePriceStream(symbol) {
            if (binanceWs) binanceWs.close();
            const wsSymbol = symbol.replace('-', '').toLowerCase();
            document.getElementById('ui-curr').innerText = "Loading...";

            // Connect to Binance.us first (US Users), fallback to Binance.com (Global Users)
            function connectWs(url) {
                binanceWs = new WebSocket(url);
                binanceWs.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data && data.c) {
                        const price = parseFloat(data.c);
                        const formatted = price < 1 ? price.toPrecision(5) : price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                        document.getElementById('ui-curr').innerText = `$${formatted}`;
                        document.getElementById('live-dot').classList.remove('bg-rose-500');
                        document.getElementById('live-dot').classList.add('bg-green-500', 'animate-pulse');
                    }
                };
                binanceWs.onerror = () => {
                    if (url.includes('.us')) {
                        console.log("Binance.us failed, falling back to Binance.com...");
                        connectWs(`wss://stream.binance.com:9443/ws/${wsSymbol}t@ticker`);
                    } else {
                        document.getElementById('ui-curr').innerText = "Awaiting Inference...";
                        document.getElementById('live-dot').classList.remove('bg-green-500', 'animate-pulse');
                        document.getElementById('live-dot').classList.add('bg-rose-500');
                    }
                };
            }
            connectWs(`wss://stream.binance.us:9443/ws/${wsSymbol}t@ticker`);
        }"""

content = content.replace(old_ws, new_ws)

# Add Nasdaq and DXY visually to the top header
old_header = """<p class="text-white/40 text-[11px] uppercase tracking-[0.1em] font-medium flex items-center gap-2">
                <span class="w-1.5 h-1.5 rounded-full bg-white/40"></span>
                CNN + GRU Inference Model
            </p>"""
new_header = """<p class="text-white/40 text-[11px] uppercase tracking-[0.1em] font-medium flex items-center gap-2">
                <span class="w-1.5 h-1.5 rounded-full bg-white/40"></span>
                CNN + GRU Inference Model
            </p>
            <div class="flex items-center gap-3 mt-2">
                <span class="text-[9px] text-white/30 uppercase tracking-widest border border-white/5 rounded px-2 py-0.5">Macro Edge: Nasdaq (NDX)</span>
                <span class="text-[9px] text-white/30 uppercase tracking-widest border border-white/5 rounded px-2 py-0.5">Macro Edge: US Dollar (DXY)</span>
            </div>"""

content = content.replace(old_header, new_header)

# Fix live-dot ID
content = content.replace('<div class="w-1.5 h-1.5 bg-green-500 rounded-full"></div>', '<div id="live-dot" class="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"></div>')

with open("pro_ui.py", "w", encoding="utf-8") as f:
    f.write(content)
