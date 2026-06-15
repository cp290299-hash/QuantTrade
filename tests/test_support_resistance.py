import pytest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import calculate_support_resistance

def test_calculate_support_resistance():
    """測試支撐壓力計算函數"""
    # 建立一個有 30 天數據的 DataFrame
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    close = [100 + i for i in range(30)]  # 逐步上漲
    high = [c + 2 for c in close]
    low = [c - 2 for c in close]
    df = pd.DataFrame({
        'Close': close,
        'High': high,
        'Low': low
    }, index=dates)
    
    current_price = close[-1]  # 129
    result = calculate_support_resistance(df, current_price)
    
    # 檢查回傳的鍵是否存在
    assert 'resistance' in result
    assert 'target' in result
    assert 'stop_loss' in result
    assert 'atr' in result
    
    # 壓力應該 >= 目前價格
    assert result['resistance'] >= current_price
    # 停損應該 < 目前價格
    assert result['stop_loss'] < current_price
    
    # 測試空 DataFrame
    empty_df = pd.DataFrame()
    result_empty = calculate_support_resistance(empty_df, 100)
    assert result_empty == {"resistance":0,"target":0,"stop_loss":0}