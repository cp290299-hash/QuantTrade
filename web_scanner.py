import shioaji as sj
import os
import yfinance as yf
import pandas as pd
from flask import Flask, render_template_string
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# --- 永豐 API 初始化 ---
api = sj.Shioaji()
api.login(api_key=os.getenv("SHIOAJI_API_KEY"), secret_key=os.getenv("SHIOAJI_SECRET_KEY"))

# --- 完整股票池 ---
TICKER_INFO = {
    "^TWII": "台股指數", "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", 
    "2382.TW": "廣達", "2308.TW": "台達電", "2603.TW": "長榮", "2886.TW": "兆豐金", 
    "0050.TW": "台灣50", "NVDA": "輝達", "AAPL": "蘋果", "TSLA": "特斯拉", 
    "PLTR": "Palantir", "AMD": "超微", "CRSP": "CRISPR", "META": "Meta", 
    "QQQ": "NASDAQ100", "AMZN": "亞馬遜", "GOOGL": "谷歌-A", "SITM": "SiTime", 
    "WRD": "文遠知行", "SBET": "SharpLink", "TEAM": "Atlassian", "NXPI": "恩智浦", 
    "BE": "Bloom Energy", "NOK": "諾基亞", "USAR": "USA Rare Earth", "SOUN": "SoundHound", 
    "RKLB": "Rocket Lab", "MSFT": "微軟", "CEG": "Constellation", "CRM": "Salesforce", 
    "ADBE": "Adobe", "CRWD": "CrowdStrike", "CRWV": "CoreWeave", "COIN": "Coinbase", 
    "MU": "美光科技", "AEP": "美國電力", "SMR": "NuScale", "TSM": "台積電 ADR", 
    "INOD": "Innodata", "ALTO": "Alto Ingredients", "ADMA": "ADMA Biologics", 
    "CLOV": "Clover Health", "TBLA": "Taboola", "YOU": "Clear Secure", 
    "GTE": "Gran Tierra", "IAU": "黃金ETF", "ARM": "Arm", "AVGO": "博通", 
    "LRCX": "泛林集團", "AMAT": "應用材料", "ASML": "阿斯麥", "INTC": "英特爾", 
    "GLW": "康寧", "QCOM": "高通", "USO": "原油ETF", "GDX": "金礦ETF", "NOW": "ServiceNow"
}

def compute_indicators(df, current_vol):
    """計算 RSI, SMA, 及量能倍數"""
    close = df['Close']
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = delta.clip(upper=0).abs().rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = round((100 - (100 / (1 + rs))).iloc[-1], 2)
    
    # SMA
    sma5 = round(close.rolling(5).mean().iloc[-1], 2)
    sma20 = round(close.rolling(20).mean().iloc[-1], 2)
    
    # 量能倍數
    avg_vol = df['Volume'].iloc[-6:-1].mean()
    vol_ratio = round(current_vol / avg_vol, 2) if avg_vol > 0 else 0
    
    return rsi, sma5, sma20, vol_ratio

def get_data_for_all():
    results = []
    tw_tickers = [t.replace('.TW', '') for t in TICKER_INFO.keys() if '.TW' in t]
    
    live_data = {}
    try:
        snapshots = api.snapshots(tw_tickers)
        for s in snapshots:
            live_data[s['code']] = {'price': s['close'], 'vol': s['vol']}
    except: pass

    for ticker, name in TICKER_INFO.items():
        try:
            df = yf.Ticker(ticker).history(period="2mo")
            if df.empty:
                results.append({'name': name, 'ticker': ticker, 'price': "N/A", 'rsi': 0, 'sma5': 0, 'sma20': 0, 'vol_ratio': 0})
                continue
            
            code = ticker.replace('.TW', '')
            if code in live_data:
                price = live_data[code]['price']
                vol = live_data[code]['vol']
            else:
                price = float(df['Close'].iloc[-1])
                vol = float(df['Volume'].iloc[-1])
            
            rsi, sma5, sma20, vol_ratio = compute_indicators(df, vol)
            
            results.append({
                'name': name, 'ticker': ticker, 'price': f"{price:.2f}",
                'rsi': rsi, 'sma5': sma5, 'sma20': sma20, 'vol_ratio': f"{vol_ratio}x"
            })
        except Exception:
            results.append({'name': name, 'ticker': ticker, 'price': "Err", 'rsi': 0, 'sma5': 0, 'sma20': 0, 'vol_ratio': 0})
    return results

@app.route('/')
def home():
    data = get_data_for_all()
    return render_template_string("""
    <body style="background:#121214; color:#fff; font-family:sans-serif; padding:20px;">
        <h1>📊 量化監控台 (含技術指標)</h1>
        <table border="1" style="width:100%; border-collapse:collapse; color:#fff; text-align:center;">
            <tr style="background:#2a2a30;">
                <th>標的</th><th>價格</th><th>SMA5</th><th>SMA20</th><th>RSI</th><th>量能倍數</th>
            </tr>
            {% for r in data %}
            <tr>
                <td>{{ r.name }} ({{ r.ticker }})</td>
                <td>{{ r.price }}</td>
                <td>{{ r.sma5 }}</td>
                <td>{{ r.sma20 }}</td>
                <td>{{ r.rsi }}</td>
                <td>{{ r.vol_ratio }}</td>
            </tr>
            {% endfor %}
        </table>
    </body>
    """, data=data)

if __name__ == '__main__':
    app.run(port=5005, debug=True)