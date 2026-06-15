import os
import json

# 確保 data 資料夾存在
os.makedirs('data', exist_ok=True)

# 建立空的 earnings_cache.json
cache_path = os.path.join('data', 'earnings_cache.json')
if not os.path.exists(cache_path):
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump({}, f)
    print("✅ 已建立 data/earnings_cache.json")
else:
    print("ℹ️ data/earnings_cache.json 已存在")

# 檢查 my_stocks_us.txt 是否至少有一檔美股
us_file = 'my_stocks_us.txt'
if not os.path.exists(us_file):
    with open(us_file, 'w', encoding='utf-8') as f:
        f.write("AAPL\n")
    print("✅ 已建立 my_stocks_us.txt 並加入 AAPL")
else:
    with open(us_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if len(lines) == 0 or all(not line.strip() for line in lines):
        with open(us_file, 'a', encoding='utf-8') as f:
            f.write("AAPL\n")
        print("✅ 已新增 AAPL 到 my_stocks_us.txt")
    else:
        print("ℹ️ my_stocks_us.txt 已有股票")

print("🎉 修復完成，請重新啟動 Flask 並測試 /earnings_calendar")