# test_institutional.py
import requests
import sqlite3
import os
from datetime import datetime

# 測試三大法人資料抓取
def test_fetch_institutional():
    stock_id = "2330"  # 台積電
    date = datetime.now().strftime('%Y%m%d')
    url = f"https://www.twse.com.tw/fund/TWT44U?stockCode={stock_id}&date={date}&response=json"
    print(f"請求 URL: {url}")
    try:
        resp = requests.get(url, timeout=10)
        print(f"HTTP 狀態碼: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"回應 stat: {data.get('stat')}")
            if data.get('stat') == 'OK' and 'data' in data and data['data']:
                row = data['data'][0]
                print(f"資料列: {row}")
                foreign = int(row[2].replace(',', '')) if row[2] else 0
                trust = int(row[3].replace(',', '')) if row[3] else 0
                dealer = int(row[4].replace(',', '')) if row[4] else 0
                print(f"外資買賣超: {foreign}, 投信: {trust}, 自營商: {dealer}")
            else:
                print("無資料或 stat 非 OK")
    except Exception as e:
        print(f"錯誤: {e}")

# 測試千張大戶資料抓取
def test_fetch_large_shareholders():
    stock_id = "2330"
    url = f"https://opendata.tdcc.com.tw/api/opendata/{stock_id}"
    print(f"請求 URL: {url}")
    try:
        resp = requests.get(url, timeout=10)
        print(f"HTTP 狀態碼: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            if data and 'data' in data and data['data']:
                total_shares = None
                large_shares = 0
                for item in data['data']:
                    range_str = item['持股分級']
                    shares = int(item['股數'].replace(',', ''))
                    if range_str == "合計":
                        total_shares = shares
                    elif any(level in range_str for level in ["1,000", "5,000", "10,000", "50,000", "100,000"]):
                        large_shares += shares
                if total_shares and total_shares > 0:
                    ratio = large_shares / total_shares * 100
                    print(f"千張大戶持股比率: {ratio:.2f}%")
                else:
                    print("無法計算比率")
            else:
                print("無資料")
    except Exception as e:
        print(f"錯誤: {e}")

if __name__ == "__main__":
    print("=== 測試三大法人 API ===")
    test_fetch_institutional()
    print("\n=== 測試集保千張大戶 API ===")
    test_fetch_large_shareholders()