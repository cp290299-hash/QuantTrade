import os, shioaji as sj, yfinance as yf, pandas as pd, webbrowser, threading, socket
from flask import Flask, render_template_string
from dotenv import load_dotenv

# --- 1. 環境設定 ---
load_dotenv()
api = sj.Shioaji(simulation=True)
api.login(os.getenv("SHIOAJI_API_KEY"), os.getenv("SHIOAJI_SECRET_KEY"))
ca_path = os.getenv("SHIOAJI_CA_PATH").replace('\\', '/')
api.activate_ca(ca_path=ca_path, ca_passwd=os.getenv("SHIOAJI_CA_PASS"), person_id=os.getenv("SHIOAJI_PERSON_ID"))

app = Flask(__name__)

# --- 2. 核心數據獲取與 AI 計算引擎 ---
def get_data(ticker):
    try:
        # A. 讀取 CSV 中的 AI 數據 (若無檔案或無資料，給予預設值)
        boost = 1.0
        if os.path.exists('hot_stock_pool.csv'):
            df = pd.read_csv('hot_stock_pool.csv', encoding='utf-8-sig')
            match = df[df['代號'] == ticker]
            if not match.empty:
                boost = float(match['爆量倍數'].iloc[0])
        
        # B. 獲取市場價格與漲跌
        price, source, change = "N/A", "未知", 0.0
        
        # 判斷是否為台股 (代號為數字 或 .TW 結尾)
        if ticker.replace('.TW', '').isdigit():
            code = ticker.replace('.TW', '')
            contract = api.Contracts.Stocks.get(code)
            if contract:
                snap = api.snapshots([contract])[0]
                price = float(snap['close'])
                change = float(snap['change_price'])
                source = "永豐API"
        else: # 美股
            data = yf.Ticker(ticker).history(period="2d")
            if len(data) >= 2:
                price = float(data['Close'].iloc[-1])
                change = float(data['Close'].iloc[-1] - data['Close'].iloc[-2])
                source = "Yahoo"
        
        # C. 計算 AI 共振分 (固定邏輯)
        res_score = min(100, 50 + (boost * 5) + (change * 1))
        
        return f"{price:.2f}", boost, f"{res_score:.1f}", f"{change:.2f}", source
    except:
        return "N/A", 1.0, "50.0", "0.00", "錯誤"

# --- 3. 網頁渲染 ---
@app.route('/')
def home():
    # 確保以 utf-8-sig 讀取，防止亂碼
    with open('my_stocks.txt', 'r', encoding='utf-8-sig') as f:
        tickers = [line.strip() for line in f if line.strip()]
    
    results = [{'t': t, 'd': get_data(t)} for t in tickers]
    
    html = """
    <meta charset="utf-8">
    <body style="background:#121214; color:#fff; font-family:sans-serif; padding:20px;">
        <h1 style="color:#00adb5;">🚀 AI 量化數據監控中心</h1>
        <table style="width:100%; border-collapse:collapse; background:#1e1e24; text-align:center;">
            <tr style="background:#2a2a30; color:#ddd; height:40px;">
                <th>代號</th><th>現價</th><th>爆量倍數</th><th>AI共振分</th><th>漲跌</th><th>來源</th>
            </tr>
            {% for r in results %}
            <tr style="border-bottom:1px solid #333; height:35px;">
                <td>{{r.t}}</td>
                <td>{{r.d[0]}}</td>
                <td>{{r.d[1]}}</td>
                <td style="color:gold; font-weight:bold;">{{r.d[2]}}</td>
                <td style="color: {{ 'red' if r.d[3]|float > 0 else 'green' if r.d[3]|float < 0 else 'white' }}">{{r.d[3]}}</td>
                <td>{{r.d[4]}}</td>
            </tr>
            {% endfor %}
        </table>
    </body>"""
    return render_template_string(html, results=results)

# --- 4. 啟動機制 (防雙開) ---
if __name__ == '__main__':
    def open_browser():
        webbrowser.open("http://127.0.0.1:5005")
    
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if sock.connect_ex(('127.0.0.1', 5005)) != 0:
            threading.Timer(1.5, open_browser).start()
        sock.close()
        
    app.run(port=5005, use_reloader=False)