import requests
import sqlite3
import os
from datetime import datetime

stock_id = "2330"
date = datetime.now().strftime('%Y%m%d')
url = f"https://www.twse.com.tw/fund/TWT44U?stockCode={stock_id}&date={date}&response=json"
print("請求 URL:", url)
resp = requests.get(url, timeout=10)
print("狀態碼:", resp.status_code)
if resp.status_code == 200:
    data = resp.json()
    print("stat:", data.get('stat'))
    if data.get('stat') == 'OK' and data.get('data'):
        row = data['data'][0]
        print("外資:", row[2], "投信:", row[3], "自營:", row[4])
    else:
        print("無資料或 stat 非 OK")
else:
    print("請求失敗")