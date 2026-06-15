from flask import Flask, render_template_string
import yfinance as yf
import pandas as pd

app = Flask(__name__)

# ================= 股票池 =================
TICKER_INFO = {
    # 台股（含指數）
    "^TWII": "台股指數",
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2382.TW": "廣達",
    "2308.TW": "台達電", "2603.TW": "長榮", "2886.TW": "兆豐金", "0050.TW": "台灣50",

    # 原有美股
    "NVDA": "輝達", "AAPL": "蘋果", "TSLA": "特斯拉", "PLTR": "Palantir", "AMD": "超微",

    # 新增美股（截圖中出現的）
    "CRSP": "CRISPR Therapeutics", "META": "Meta Platforms", "QQQ": "NASDAQ100指數ETF",
    "AMZN": "亞馬遜", "GOOGL": "谷歌-A", "SITM": "SiTime", "WRD": "文遠知行",
    "SBET": "SharpLink", "TEAM": "Atlassian", "NXPI": "恩智浦", "BE": "Bloom Energy",
    "NOK": "諾基亞", "USAR": "USA Rare Earth", "SOUN": "SoundHound AI", "RKLB": "Rocket Lab",
    "MSFT": "微軟", "CEG": "Constellation Energy", "CRM": "Salesforce", "ADBE": "Adobe",
    "CRWD": "CrowdStrike", "CRWV": "CoreWeave", "COIN": "Coinbase", "MU": "美光科技",
    "AEP": "美國電力", "SMR": "NuScale Power", "TSM": "台積電 ADR", "INOD": "Innodata",
    "ALTO": "Alto Ingredients", "ADMA": "ADMA Biologics", "CLOV": "Clover Health",
    "TBLA": "Taboola Com", "YOU": "Clear Secure", "GTE": "Gran Tierra Energy",
    "IAU": "黃金信托ETF", "ARM": "Arm Holdings", "AVGO": "博通", "LRCX": "泛林集團",
    "AMAT": "應用材料", "ASML": "阿斯麥", "INTC": "英特爾", "GLW": "康寧",
    "QCOM": "高通", "USO": "美國原油ETF", "GDX": "Gold Miners ETF", "NOW": "ServiceNow"
}

# 頂部情緒指數
SENTIMENT_INDICES = {
    "^VIX": "恐慌指數 (VIX)",
    "^SKEW": "黑天鵝指數 (SKEW)",
    "BAMLH0A0HYM2": "信用利差"
}

# 美股主要指數（順序：道瓊 → 那斯達克 → 標普500）
US_MAJOR_INDICES = {
    "^DJI": "道瓊工業指數",
    "^IXIC": "那斯達克",
    "^GSPC": "標普500指數"
}

def fetch_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo")
        if df.empty or 'Close' not in df.columns:
            return None
        close_series = df['Close'].dropna()
        if len(close_series) < 20:
            return None
        return pd.DataFrame({'Close': close_series})
    except Exception as e:
        print(f"下載 {ticker} 失敗: {e}")
        return None

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = delta.clip(upper=0).abs().rolling(period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2) if not pd.isna(rsi.iloc[-1]) else 50.0

def compute_macd_norm(close, price):
    if len(close) < 26:
        return 0.0
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_raw = (ema12 - ema26).iloc[-1]
    return round((macd_raw / (price + 1e-9)) * 1000, 2)

def compute_ma(close, periods=[5,20]):
    ma_values = {}
    for p in periods:
        if len(close) >= p:
            ma = close.rolling(p).mean().iloc[-1]
            ma_values[p] = round(ma, 2)
        else:
            ma_values[p] = None
    return ma_values

def compute_ai_score(rsi, macd_norm):
    raw = 50 + (50 - rsi) * 0.4 + (macd_norm * 0.5)
    return max(0, min(100, round(raw, 1)))

def get_stock_data(ticker, name):
    df = fetch_data(ticker)
    if df is None:
        return None
    close = df['Close']
    try:
        price = float(close.iloc[-1])
    except Exception:
        return None
    rsi = compute_rsi(close)
    macd = compute_macd_norm(close, price)
    score = compute_ai_score(rsi, macd)
    ma_dict = compute_ma(close)
    return {
        'name': name,
        'price': price,
        'rsi': rsi,
        'macd': macd,
        'score': score,
        'ticker': ticker,
        'sma5': ma_dict.get(5),
        'sma20': ma_dict.get(20)
    }

@app.route('/')
def home():
    # 情緒指數
    sentiment_data = []
    for ticker, name in SENTIMENT_INDICES.items():
        data = get_stock_data(ticker, name)
        if data:
            sentiment_data.append(data)

    # 美股主要指數
    major_indices_data = []
    for ticker, name in US_MAJOR_INDICES.items():
        data = get_stock_data(ticker, name)
        if data:
            major_indices_data.append(data)

    # 一般標的
    all_normal = []
    for ticker, name in TICKER_INFO.items():
        data = get_stock_data(ticker, name)
        if data:
            all_normal.append(data)

    # 分離台股與美股
    taiwan = [d for d in all_normal if '.TW' in d['ticker'] or d['ticker'] in ['^TWII', '0050.TW']]
    us_stocks = [d for d in all_normal if d not in taiwan]

    # 台股排序：台股指數固定第一欄
    taiwan_twii = None
    taiwan_others = []
    for d in taiwan:
        if d['ticker'] == '^TWII':
            taiwan_twii = d
        else:
            taiwan_others.append(d)
    taiwan_others.sort(key=lambda x: x['score'], reverse=True)
    taiwan_sorted = [taiwan_twii] + taiwan_others if taiwan_twii else taiwan_others

    # 美股排序：主要指數固定最前，其餘按分數排序
    us_major = []
    for ticker, name in US_MAJOR_INDICES.items():
        found = None
        for d in major_indices_data:
            if d['name'] == name:
                found = d
                break
        if not found:
            for d in us_stocks:
                if d['name'] == name:
                    found = d
                    break
        if found:
            us_major.append(found)
    us_rest = [d for d in us_stocks if d not in us_major]
    us_rest.sort(key=lambda x: x['score'], reverse=True)

    # ========== 關鍵修正：明確設定顯示名稱 ==========
    for d in taiwan_sorted:
        d['display_name'] = f"🇹🇼 {d['name']}"          # 台股顯示中文名稱
    for d in us_major:
        d['display_name'] = f"📊 {d['name']}"            # 美股主要指數顯示中文名稱
    for d in us_rest:
        d['display_name'] = f"🇺🇸 {d['ticker'].upper()}" # 美股一般個股顯示「代號」
    for d in sentiment_data:
        d['display_name'] = f"⚠️ {d['name']}"            # 情緒指數顯示中文名稱

    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>量化監控台 - 美股顯示代號版</title>
        <style>
            body { background: #121214; color: #fff; font-family: 'Segoe UI', sans-serif; padding: 20px; }
            h1 { color: #00adb5; }
            h2 { color: #ccc; margin-top: 30px; }
            .sentiment-box {
                background: #1a1a20;
                border: 1px solid #00adb5;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 25px;
                display: flex;
                gap: 30px;
                justify-content: space-around;
                flex-wrap: wrap;
            }
            .sentiment-card {
                background: #25252c;
                border-radius: 8px;
                padding: 10px 20px;
                text-align: center;
                min-width: 200px;
            }
            .sentiment-card h3 { margin: 5px 0; color: #00adb5; }
            .sentiment-card .value { font-size: 1.4em; font-weight: bold; margin: 5px 0; }
            .sentiment-card .label { font-size: 0.8em; color: #aaa; }
            table { width: 100%; border-collapse: collapse; background: #1e1e24; margin-bottom: 20px; }
            th, td { border: 1px solid #333; padding: 6px 10px; text-align: center; }
            th { background: #2a2a30; color: #00adb5; }
            tr:hover { background: #2c2c34; }
            .ma-up { color: #ff7e7e; }
            .ma-down { color: #7eff7e; }
        </style>
    </head>
    <body>
        <h1>📊 量化監控台 (美股強制顯示代號)</h1>

        <div class="sentiment-box">
            {% for s in sentiment_data %}
            <div class="sentiment-card">
                <h3>{{ s.display_name }}</h3>
                <div class="value">{{ "{:.2f}".format(s.price) }}</div>
                <div class="label">RSI: {{ s.rsi }} | MACD: {{ s.macd }} | AI: {{ s.score }}%</div>
            </div>
            {% endfor %}
        </div>

        <h2>🇹🇼 台股</h2>
        <table>
            <thead><tr><th>標的</th><th>市價</th><th>SMA5</th><th>SMA20</th><th>RSI</th><th>MACD</th><th>AI分數</th></tr></thead>
            <tbody>
                {% for r in taiwan %}
                <tr>
                    <td>{{ r.display_name }}</td>
                    <td>{{ "{:.2f}".format(r.price) }}</td>
                    <td class="{% if r.sma5 and r.price > r.sma5 %}ma-up{% elif r.sma5 and r.price < r.sma5 %}ma-down{% endif %}">{{ r.sma5 if r.sma5 else 'N/A' }}</td>
                    <td class="{% if r.sma20 and r.price > r.sma20 %}ma-up{% elif r.sma20 and r.price < r.sma20 %}ma-down{% endif %}">{{ r.sma20 if r.sma20 else 'N/A' }}</td>
                    <td>{{ r.rsi }}</td>
                    <td>{{ r.macd }}</td>
                    <td>{{ r.score }}%</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <h2>🇺🇸 美股（主要指數顯示中文，個股顯示代號）</h2>
        <table>
            <thead><tr><th>標的</th><th>市價</th><th>SMA5</th><th>SMA20</th><th>RSI</th><th>MACD</th><th>AI分數</th></tr></thead>
            <tbody>
                {% for r in us_major %}
                <tr>
                    <td>{{ r.display_name }}</td>
                    <td>{{ "{:.2f}".format(r.price) }}</td>
                    <td class="{% if r.sma5 and r.price > r.sma5 %}ma-up{% elif r.sma5 and r.price < r.sma5 %}ma-down{% endif %}">{{ r.sma5 if r.sma5 else 'N/A' }}</td>
                    <td class="{% if r.sma20 and r.price > r.sma20 %}ma-up{% elif r.sma20 and r.price < r.sma20 %}ma-down{% endif %}">{{ r.sma20 if r.sma20 else 'N/A' }}</td>
                    <td>{{ r.rsi }}</td>
                    <td>{{ r.macd }}</td>
                    <td>{{ r.score }}%</td>
                </tr>
                {% endfor %}
                {% for r in us_rest %}
                <tr>
                    <td>{{ r.display_name }}</td>
                    <td>{{ "{:.2f}".format(r.price) }}</td>
                    <td class="{% if r.sma5 and r.price > r.sma5 %}ma-up{% elif r.sma5 and r.price < r.sma5 %}ma-down{% endif %}">{{ r.sma5 if r.sma5 else 'N/A' }}</td>
                    <td class="{% if r.sma20 and r.price > r.sma20 %}ma-up{% elif r.sma20 and r.price < r.sma20 %}ma-down{% endif %}">{{ r.sma20 if r.sma20 else 'N/A' }}</td>
                    <td>{{ r.rsi }}</td>
                    <td>{{ r.macd }}</td>
                    <td>{{ r.score }}%</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        <p style="color:#888;">顏色：<span style="color:#ff7e7e;">紅</span>價格>均線 <span style="color:#7eff7e;">綠</span>價格<均線</p>
    </body>
    </html>
    """, sentiment_data=sentiment_data, taiwan=taiwan_sorted, us_major=us_major, us_rest=us_rest)

if __name__ == '__main__':
    app.run(port=5005, debug=True)