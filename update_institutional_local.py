import psycopg2
import requests
from datetime import datetime, timedelta
import time

# 請貼上您從 Render PostgreSQL 複製的 External Database URL
EXTERNAL_DB_URL = "postgresql://quant_stocks_db_user:XFXRFRNFeSzoj5S6lK7bgbia6sJu6UHj@dpg-d8n2nmflk1mc73993h6g-a.ohio-postgres.render.com/quant_stocks_db"

def fetch_institutional_holders(stock_id, date=None):
    """從證交所抓取個股三大法人買賣超"""
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    url = f"https://www.twse.com.tw/fund/T86?response=json&date={date}&selectType=ALLBUT0999"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('stat') == 'OK' and 'data' in data:
                for row in data['data']:
                    if len(row) >= 6 and row[0].strip() == stock_id:
                        foreign = int(row[3].replace(',', '')) if row[3] else 0
                        trust = int(row[4].replace(',', '')) if row[4] else 0
                        dealer = int(row[5].replace(',', '')) if row[5] else 0
                        return foreign, trust, dealer, date
        return None, None, None, None
    except Exception as e:
        print(f"抓取 {stock_id} 失敗: {e}")
        return None, None, None, None

def update_institutional_data(ticker):
    stock_id = ticker.replace('.TW', '')
    foreign, trust, dealer, data_date = fetch_institutional_holders(stock_id)
    if foreign is None:
        print(f"{ticker} 無資料")
        return
    conn = psycopg2.connect(EXTERNAL_DB_URL)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO institutional_ownership (stock_id, date, foreign_investors, investment_trust, dealers)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (stock_id, date) DO UPDATE SET
            foreign_investors = EXCLUDED.foreign_investors,
            investment_trust = EXCLUDED.investment_trust,
            dealers = EXCLUDED.dealers
    """, (ticker, data_date, foreign, trust, dealer))
    conn.commit()
    cur.close()
    conn.close()
    print(f"更新 {ticker} ({data_date}): 外資={foreign}, 投信={trust}, 自營={dealer}")

def get_all_tw_tickers():
    # 從檔案讀取您的台股自選股（請依照實際路徑調整）
    file_path = "my_stocks_tw.txt"
    tickers = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().upper()
            if t:
                if not t.endswith('.TW'):
                    t = t + '.TW'
                tickers.append(t)
    return tickers

if __name__ == "__main__":
    tickers = get_all_tw_tickers()
    print(f"將更新 {len(tickers)} 檔股票...")
    for t in tickers:
        update_institutional_data(t)
        time.sleep(0.5)  # 避免請求過快
    print("更新完成")