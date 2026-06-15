import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def run_volume_scanner():
    print("📡 正在透過全球 Yahoo 伺服器清洗台股爆量數據...")
    
    # 🎯 挑選台灣市場最具爆發力的熱門股與中小型股池（含你關注的個股），讓大腦去裡面撈妖股
    target_stocks = [
        '2330.TW', '2317.TW', '2891.TW', '2886.TW', '1301.TW', '2427.TW', '2002.TW', 
        '1325.TW', '9919.TW', '4755.TW', '2303.TW', '2603.TW', '2609.TW', '2615.TW',
        '2382.TW', '2308.TW', '2357.TW', '3231.TW', '2324.TW', '2449.TW', '2344.TW',
        '3711.TW', '2408.TW', '3037.TW', '3035.TW', '2363.TW', '6116.TW', '2618.TW'
    ]
    
    scanned_results = []
    
    for ticker in target_stocks:
        try:
            # 抓取過去 5 天的日 K 線資料（Bar）
            stock = yf.Ticker(ticker)
            df = stock.history(period="5d")
            
            if len(df) < 4:
                continue
                
            # 📊 提取成交量 (Volume)
            today_volume = df['Volume'].iloc[-1]       # 今天的成交量
            v_minus_1 = df['Volume'].iloc[-2]          # 昨天的成交量
            v_minus_2 = df['Volume'].iloc[-3]          # 前天的成交量
            v_minus_3 = df['Volume'].iloc[-4]          # 大前天的成交量
            
            # 計算近三天的平均成交量
            avg_3d_volume = (v_minus_1 + v_minus_2 + v_minus_3) / 3
            
            if avg_3d_volume == 0:
                continue
                
            # 🧮 計算暴增倍數 (今天的量是過去三天的幾倍)
            volume_ratio = today_volume / avg_3d_volume
            current_price = df['Close'].iloc[-1]
            
            scanned_results.append({
                '代號': ticker,
                '當前股價': round(current_price, 2),
                '今日成交量(張)': int(today_volume / 1000), # 換算成張數
                '近三日平均(張)': int(avg_3d_volume / 1000),
                '爆量倍數': round(volume_ratio, 2)
            })
            print(f"🍏 掃描完成: {ticker.ljust(8)} | 爆量倍數: {volume_ratio:.2f} 倍")
            
        except Exception as e:
            continue
            
    # 3. 把所有結果排成表格
    result_df = pd.DataFrame(scanned_results)
    
    if not result_df.empty:
        # 👑 核心邏輯：依照「爆量倍數」從大到小排序，抓出前 50 名（這邊先有多少要多少）
        result_df = result_df.sort_values(by='爆量倍數', ascending=False).reset_index(drop=True)
        
        print("\n" + "="*60)
        print("🔥 🚀 溢滿園量化引擎：今日台股【成交量暴增】妖股雷達追蹤榜 🚀 🔥")
        print("="*60)
        print(result_df.to_string(index=False))
        print("="*60)
        
        # 自動把這份爆量名單存成 Excel 表格，方便主程式隨時調用
        result_df.to_csv("hot_stock_pool.csv", index=False, encoding="utf-8-sig")
        print("💾 爆量個股名單已成功儲存至 -> hot_stock_pool.csv")
    else:
        print("🛑 偵測完成，今日市場平靜，暫無符合爆量 2 倍以上的標的。")

if __name__ == "__main__":
    run_volume_scanner()