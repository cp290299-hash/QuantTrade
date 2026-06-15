import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import calculate_features, FEATURE_NAMES, get_feature_vector

def test_feature_names_length():
    # FEATURE_NAMES 應該有 22 個特徵（你可以去 app.py 數一下）
    assert len(FEATURE_NAMES) == 23

def test_calculate_features():
    # 創建一個有 100 天歷史資料的 DataFrame
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'Open': np.random.randn(100) + 100,
        'High': np.random.randn(100) + 101,
        'Low': np.random.randn(100) + 99,
        'Close': np.random.randn(100) + 100,
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    features = calculate_features(df)
    assert features is not None, "長度足夠時應回傳特徵字典"
    
    # 檢查每個特徵名稱都存在
    for name in FEATURE_NAMES:
        assert name in features, f"缺少特徵 {name}"
    
    # 測試資料長度不足（少於 60 天）
    short_df = df.head(30)
    features_short = calculate_features(short_df)
    assert features_short is None, "資料不足應回傳 None"

def test_get_feature_vector():
    # 建立一個包含所有特徵的字典，值設為索引編號
    features = {name: i for i, name in enumerate(FEATURE_NAMES)}
    vec = get_feature_vector(features)
    assert len(vec) == len(FEATURE_NAMES)
    assert vec[0] == 0
    assert vec[-1] == len(FEATURE_NAMES) - 1
    
    # 測試缺少部分特徵時，缺漏的應補 0
    features2 = {}
    vec2 = get_feature_vector(features2)
    assert all(v == 0 for v in vec2), "缺少特徵應補 0"