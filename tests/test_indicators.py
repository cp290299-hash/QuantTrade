import pytest
import pandas as pd
import numpy as np
import sys
import os

# 讓測試程式能找到上層的 app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import calculate_rsi, calculate_macd

def test_calculate_rsi():
    # 測試上漲趨勢：RSI 應該接近 100
    close = pd.Series([100 + i for i in range(20)])
    rsi = calculate_rsi(close, period=14)
    assert rsi.iloc[-1] > 90, f"上漲時 RSI 應大於 90，實際得到 {rsi.iloc[-1]}"

    # 測試下跌趨勢：RSI 應該接近 0
    close = pd.Series([100 - i for i in range(20)])
    rsi = calculate_rsi(close, period=14)
    assert rsi.iloc[-1] < 10, f"下跌時 RSI 應小於 10，實際得到 {rsi.iloc[-1]}"

    # 測試資料不足時應回傳 50（數值，不是 Series）
    short_close = pd.Series([100, 101])
    rsi = calculate_rsi(short_close, period=14)
    assert rsi == 50, "資料不足時 RSI 應回傳 50"
def test_macd_status():
    # 創建一個足夠長度的價格序列（至少 35 天）
    close = pd.Series([100 + i for i in range(40)])
    macd, signal, hist, status = calculate_macd(close)
    assert macd is not None
    assert status in ["🔝 金叉", "🔻 死叉", "─ 持中"]