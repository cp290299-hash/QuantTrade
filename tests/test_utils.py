import pytest
import sys
import os

# 讓測試能找到 app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import clean_nan, get_vwap_status, calculate_vwap
import pandas as pd
import numpy as np
import math

def test_clean_nan():
    """測試 clean_nan 函數是否能處理 NaN、inf、None"""
    # 正常數字
    assert clean_nan(10.5) == 10.5
    assert clean_nan(3) == 3
    
    # NaN 和 inf 應該變成 'N/A'
    assert clean_nan(float('nan')) == 'N/A'
    assert clean_nan(float('inf')) == 'N/A'
    assert clean_nan(None) == 'N/A'
    
    # 測試字典
    d = {'price': float('nan'), 'value': 100}
    cleaned = clean_nan(d)
    assert cleaned['price'] == 'N/A'
    assert cleaned['value'] == 100
    
    # 測試列表
    lst = [1, float('nan'), 3]
    cleaned_lst = clean_nan(lst)
    assert cleaned_lst[0] == 1
    assert cleaned_lst[1] == 'N/A'
    assert cleaned_lst[2] == 3

def test_get_vwap_status():
    """測試 VWAP 狀態判斷"""
    # 價格 > VWAP
    status = get_vwap_status(current_price=150, vwap=140)
    assert "站上" in status
    assert "VWAP" in status
    
    # 價格 < VWAP
    status = get_vwap_status(current_price=130, vwap=140)
    assert "跌破" in status
    
    # VWAP 為 None
    status = get_vwap_status(current_price=100, vwap=None)
    assert status == "N/A"

def test_calculate_vwap():
    """測試 VWAP 計算（使用假數據）"""
    # 創建一個簡單的 DataFrame
    df = pd.DataFrame({
        'High': [100, 101, 102],
        'Low': [99, 100, 101],
        'Close': [100, 101, 102],
        'Volume': [1000, 1500, 2000]
    })
    vwap = calculate_vwap(df)
    # 手動計算預期值
    typical_prices = [(100+99+100)/3, (101+100+101)/3, (102+101+102)/3]  # [99.6667, 100.6667, 101.6667]
    cum_typical_vol = (99.6667*1000 + 100.6667*1500 + 101.6667*2000)
    cum_vol = 1000+1500+2000
    expected = cum_typical_vol / cum_vol
    assert abs(vwap - expected) < 0.01
    
    # 空 DataFrame
    empty_df = pd.DataFrame()
    assert calculate_vwap(empty_df) is None