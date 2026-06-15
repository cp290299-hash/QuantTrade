# -*- coding: utf-8 -*-
import os
import sys
import json
import math
import time
import logging
import random
import re
import requests
import numpy as np
import pandas as pd
import threading
from datetime import datetime, timedelta
from collections import defaultdict
from flask import Flask, render_template_string, request, redirect, url_for, make_response, jsonify
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler

# 機器學習套件
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# XGBoost 與 LightGBM
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost 未安裝，將跳過 XGBoost 模型")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("⚠️ LightGBM 未安裝，將跳過 LightGBM 模型")

# 情緒分析
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️ TextBlob 未安裝，情緒分析將使用模擬數據")
    class TextBlob:
        def __init__(self, text):
            self.text = text
        @property
        def sentiment(self):
            class Sentiment:
                polarity = 0.0
            return Sentiment()

# Shioaji (永豐金 API)
try:
    import shioaji as sj
    SHIOAJI_AVAILABLE = True
except ImportError:
    SHIOAJI_AVAILABLE = False
    print("⚠️ Shioaji 未安裝，台股將使用 yfinance 延遲報價")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'quant-ai-dashboard'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOCK_FILE_TW = os.path.join(BASE_DIR, "my_stocks_tw.txt")
STOCK_FILE_US = os.path.join(BASE_DIR, "my_stocks_us.txt")
POSITIONS_FILE = os.path.join(BASE_DIR, "positions.json")
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCALERS_DIR = os.path.join(BASE_DIR, "scalers")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCALERS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

AI_SCORES_FILE = os.path.join(DATA_DIR, "ai_scores.json")
ALERTS_FILE = os.path.join(DATA_DIR, "alerts.json")
WATCHLIST_FILE = os.path.join(DATA_DIR, "watchlist.json")

DEFAULT_SETTINGS = {
    "refresh_seconds": 60,
    "bonding_threshold": 3.0,
    "enable_ensemble": True,
    "prediction_days": 3,
    "train_days": 180,
    "retrain_hours": 24,
    "auto_clear_cache_hours": 24,
    "background_update_minutes": 15,
    "cache_ttl_minutes": 5,
    "max_loss_per_trade": 5.0,
    "max_loss_per_day": 10.0,
    "max_concentration": 40.0,
    "cash_balance": 0.0,
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
            for k, v in DEFAULT_SETTINGS.items():
                if k not in loaded:
                    loaded[k] = v
            return loaded
    return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)

settings = load_settings()

for f in [STOCK_FILE_TW, STOCK_FILE_US]:
    if not os.path.exists(f):
        with open(f, "w", encoding="utf-8") as _:
            pass
if not os.path.exists(POSITIONS_FILE):
    with open(POSITIONS_FILE, "w", encoding='utf-8') as f:
        json.dump([], f)
if not os.path.exists(WATCHLIST_FILE):
    default_watchlist = ["NVDA", "TSLA", "OPEN", "MRVL", "PLTR", "ANET", "VRT", "SMCI", "AMD"]
    with open(WATCHLIST_FILE, "w", encoding='utf-8') as f:
        json.dump(default_watchlist, f, ensure_ascii=False)

# ================== 全域快取 ==================
_data_cache = {}
_model_cache = {}
_indicators_cache = {}
_ai_scores_cache = {}
_alerts_cache = {}
_cache_lock = threading.Lock()
_model_lock = threading.Lock()

def get_cached_data(ticker, period="2y", ttl_seconds=None):
    if ttl_seconds is None:
        ttl_seconds = settings.get('cache_ttl_minutes', 5) * 60
    key = f"{ticker}_{period}"
    now = time.time()
    with _cache_lock:
        if key in _data_cache and now - _data_cache[key]['time'] < ttl_seconds:
            return _data_cache[key]['data']
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            _data_cache[key] = {'data': df, 'time': now}
            return df
        except:
            return pd.DataFrame()

# ================== Shioaji 連線管理 ==================
_sj = None
_sj_login_time = 0

def get_shioaji_status():
    if not SHIOAJI_AVAILABLE:
        return "❌ 未安裝"
    if _sj is not None:
        return "🟢 即時模式 (<1秒)"
    return "🔴 延遲模式 (15-20分)"

def get_shioaji():
    global _sj, _sj_login_time
    now = time.time()
    if _sj is not None and now - _sj_login_time < 23 * 3600:
        return _sj
    if not SHIOAJI_AVAILABLE:
        return None
    api_key = os.getenv("SHIOAJI_API_KEY")
    secret_key = os.getenv("SHIOAJI_SECRET_KEY")
    person_id = os.getenv("SHIOAJI_PERSON_ID")
    passwd = os.getenv("SHIOAJI_PASSWD")
    try:
        if _sj is not None:
            try:
                _sj.logout()
            except:
                pass
        _sj = sj.Shioaji(simulation=False)
        if api_key and secret_key:
            _sj.login(api_key=api_key, secret_key=secret_key)
            logger.info("Shioaji 登入成功 (使用 API Key)")
        elif person_id and passwd:
            _sj.login(person_id=person_id, passwd=passwd)
            logger.info("Shioaji 登入成功 (使用帳號密碼)")
        else:
            logger.warning("未提供任何登入憑證")
            return None
        _sj_login_time = now
        return _sj
    except Exception as e:
        logger.error(f"Shioaji 登入失敗: {e}")
        return None

def get_tw_stock_realtime_shioaji(ticker):
    sj_api = get_shioaji()
    if sj_api is None:
        return None, None, None
    stock_id = ticker.replace('.TW', '').upper()
    try:
        contract = sj_api.Contracts.Stocks[stock_id]
        snapshot = sj_api.snapshot([contract])
        if not snapshot:
            return None, None, None
        tick = snapshot[0]
        curr = tick.close if tick.close else tick.last_price
        if curr is None:
            return None, None, None
        prev = tick.reference
        if prev is None or prev == 0:
            return None, None, None
        change = curr - prev
        pct = (change / prev) * 100
        return curr, change, pct
    except Exception as e:
        return None, None, None

# ================== 市場狀態判斷 ==================
def get_market_regime_en():
    try:
        vix = yf.Ticker("^VIX").history(period="2d")['Close'].iloc[-1]
        if vix > 25:
            return "risk_off"
        elif vix < 15:
            return "risk_on"
        return "neutral"
    except:
        return "neutral"

# ================== 技術指標函數 ==================
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_vwap(df):
    if df.empty or len(df) < 2:
        return None
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap.iloc[-1]

def get_vwap_status(current_price, vwap):
    if vwap is None:
        return "N/A"
    if current_price > vwap:
        return f"🟢 站上 VWAP ({vwap:.2f})"
    else:
        return f"🔴 跌破 VWAP ({vwap:.2f})"

def calculate_macd(close, fast=12, slow=26, signal=9):
    if len(close) < slow + signal:
        return None, None, None, "-"
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    macd_curr = macd_line.iloc[-1]
    signal_curr = signal_line.iloc[-1]
    macd_prev = macd_line.iloc[-2] if len(macd_line) > 1 else macd_curr
    signal_prev = signal_line.iloc[-2] if len(signal_line) > 1 else signal_curr
    if macd_curr > signal_curr and macd_prev <= signal_prev:
        status = "🔝 金叉"
    elif macd_curr < signal_curr and macd_prev >= signal_prev:
        status = "🔻 死叉"
    else:
        status = "─ 持中"
    hist_val = histogram.iloc[-1]
    return macd_curr, signal_curr, hist_val, status

def calculate_support_resistance(df, current_price):
    if df.empty or len(df) < 20:
        return {"resistance": 0, "target": 0, "stop_loss": 0}
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    high_20 = np.max(high[-20:])
    tr = high - low
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
    ma20 = np.mean(close[-20:])
    resistance = max(high_20, ma20 + 2 * np.std(close[-20:]))
    target = current_price + atr * 2.5
    stop_loss = current_price - atr * 1.5
    return {"resistance": round(resistance, 2), "target": round(target, 2), "stop_loss": round(stop_loss, 2), "atr": round(atr, 2)}

def calculate_ai_resonance_score(rsi, macd_status, vwap_status, volume_ratio):
    score = 0
    if rsi < 30:
        score += 25
    elif rsi < 50:
        score += 15
    elif rsi < 70:
        score += 10
    else:
        score += 5
    if "金叉" in macd_status:
        score += 25
    elif "持中" in macd_status:
        score += 10
    else:
        score += 0
    if "站上" in vwap_status:
        score += 25
    else:
        score += 5
    try:
        vol_ratio = float(volume_ratio) if volume_ratio else 1
        if vol_ratio > 1.5:
            score += 25
        elif vol_ratio > 1.2:
            score += 15
        else:
            score += 5
    except:
        score += 5
    return min(100, max(0, score))

# ================== 特徵工程 ==================
FEATURE_NAMES = [
    'return_1d', 'return_5d', 'return_10d', 'return_20d',
    'price_vs_ma5', 'price_vs_ma10', 'price_vs_ma20', 'price_vs_ma60',
    'ma5_vs_ma20', 'ma20_vs_ma60',
    'volume_ratio_5', 'volume_ratio_10',
    'rsi', 'bb_position', 'bb_width',
    'kd_k', 'kd_d',
    'volatility_5d', 'volatility_20d',
    'high_low_ratio', 'high_20d_ratio', 'low_20d_ratio', 'trend_strength'
]

def calculate_features(df):
    if df.empty or len(df) < 60:
        return None
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values if 'Volume' in df.columns else np.ones(len(close))
    features = {}
    features['return_1d'] = (close[-1] - close[-2]) / close[-2] if len(close) >= 2 else 0
    features['return_5d'] = (close[-1] - close[-6]) / close[-6] if len(close) >= 6 else 0
    features['return_10d'] = (close[-1] - close[-11]) / close[-11] if len(close) >= 11 else 0
    features['return_20d'] = (close[-1] - close[-21]) / close[-21] if len(close) >= 21 else 0
    ma5 = np.mean(close[-5:]) if len(close) >= 5 else close[-1]
    ma10 = np.mean(close[-10:]) if len(close) >= 10 else close[-1]
    ma20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
    ma60 = np.mean(close[-60:]) if len(close) >= 60 else close[-1]
    features['price_vs_ma5'] = (close[-1] - ma5) / ma5 if ma5 > 0 else 0
    features['price_vs_ma10'] = (close[-1] - ma10) / ma10 if ma10 > 0 else 0
    features['price_vs_ma20'] = (close[-1] - ma20) / ma20 if ma20 > 0 else 0
    features['price_vs_ma60'] = (close[-1] - ma60) / ma60 if ma60 > 0 else 0
    features['ma5_vs_ma20'] = (ma5 - ma20) / ma20 if ma20 > 0 else 0
    features['ma20_vs_ma60'] = (ma20 - ma60) / ma60 if ma60 > 0 else 0
    vol_ma5 = np.mean(volume[-5:]) if len(volume) >= 5 else volume[-1]
    vol_ma10 = np.mean(volume[-10:]) if len(volume) >= 10 else volume[-1]
    features['volume_ratio_5'] = volume[-1] / vol_ma5 if vol_ma5 > 0 else 1
    features['volume_ratio_10'] = volume[-1] / vol_ma10 if vol_ma10 > 0 else 1
    if len(close) >= 15:
        deltas = np.diff(close[-15:])
        gain = np.mean(deltas[deltas > 0]) if len(deltas[deltas > 0]) > 0 else 0
        loss = -np.mean(deltas[deltas < 0]) if len(deltas[deltas < 0]) > 0 else 0
        rs = gain / loss if loss > 0 else 1
        features['rsi'] = 100 - (100 / (1 + rs))
    else:
        features['rsi'] = 50
    std20 = np.std(close[-20:]) if len(close) >= 20 else 0
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    features['bb_position'] = (close[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
    features['bb_width'] = (bb_upper - bb_lower) / ma20 if ma20 > 0 else 0
    k, d = calculate_kd(high, low, close)
    features['kd_k'] = k
    features['kd_d'] = d
    features['volatility_5d'] = np.std(close[-5:]) / ma5 if ma5 > 0 else 0
    features['volatility_20d'] = np.std(close[-20:]) / ma20 if ma20 > 0 else 0
    features['high_low_ratio'] = (high[-1] - low[-1]) / close[-1] if close[-1] > 0 else 0
    features['high_20d_ratio'] = (high[-1] - np.max(high[-20:])) / close[-1] if len(high) >= 20 and close[-1] > 0 else 0
    features['low_20d_ratio'] = (low[-1] - np.min(low[-20:])) / close[-1] if len(low) >= 20 and close[-1] > 0 else 0
    features['trend_strength'] = abs(close[-1] - close[-20]) / ma20 if ma20 > 0 else 0
    return features

def get_feature_vector(features):
    return [features.get(name, 0) for name in FEATURE_NAMES]

def calculate_kd(high, low, close, n=9):
    if len(close) < n:
        return 50, 50
    low_n = np.min(low[-n:])
    high_n = np.max(high[-n:])
    if high_n == low_n:
        rsv = 50
    else:
        rsv = (close[-1] - low_n) / (high_n - low_n) * 100
    k = rsv
    d = np.mean([rsv, 50])
    return k, d

# ================== RandomForest 模型 ==================
def get_models_with_cache(ticker, force_retrain=False):
    cache_key = ticker.replace('.TW', '')
    now = time.time()
    with _model_lock:
        if not force_retrain and cache_key in _model_cache:
            model, scaler, timestamp = _model_cache[cache_key]
            if now - timestamp < 3600:
                return model, scaler
    logger.info(f"開始訓練/載入 {ticker} 模型...")
    model, scaler = train_single_model(ticker, force_retrain)
    if model:
        with _model_lock:
            _model_cache[cache_key] = (model, scaler, now)
    return model, scaler

def train_single_model(ticker, force_retrain=False):
    model_name = ticker.replace('.TW', '')
    model_path = os.path.join(MODELS_DIR, f"{model_name}_rf.joblib")
    scaler_path = os.path.join(SCALERS_DIR, f"{model_name}_scaler.joblib")
    if not force_retrain and os.path.exists(model_path) and os.path.exists(scaler_path):
        mtime = os.path.getmtime(model_path)
        if time.time() - mtime < settings.get('retrain_hours', 24) * 3600:
            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                return model, scaler
            except:
                pass
    X, y = prepare_training_data(ticker, settings.get('train_days', 180))
    if X is None or len(X) < 100:
        logger.warning(f"{ticker} 訓練資料不足")
        return None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    rf_params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'random_state': 42, 'n_jobs': -1}
    model = RandomForestRegressor(**rf_params)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"{ticker} RandomForest 模型訓練完成")
    return model, scaler

def predict_with_ensemble(ticker, df, model=None, scaler=None):
    if not settings.get('enable_ensemble', True):
        return None, None, None, None
    if model is None or scaler is None:
        model, scaler = get_models_with_cache(ticker)
    if model is None:
        return None, None, None, None
    features = calculate_features(df)
    if features is None:
        return None, None, None, None
    X_pred = np.array(get_feature_vector(features)).reshape(1, -1)
    X_pred_scaled = scaler.transform(X_pred)
    pred = model.predict(X_pred_scaled)[0]
    change_pct = pred * 100
    if change_pct > 2:
        trend = "預期上漲"
    elif change_pct < -2:
        trend = "預期下跌"
    else:
        trend = "預期盤整"
    confidence = min(100, int(60 + abs(pred) * 40))
    details = f"RandomForest:{pred*100:+.1f}%"
    return round(change_pct, 1), trend, confidence, details

def prepare_training_data(ticker, days=180):
    try:
        df = yf.Ticker(ticker).history(period=f"{days+60}d", auto_adjust=True)
        if df.empty or len(df) < 60:
            return None, None
        X = []
        y = []
        for i in range(60, len(df) - settings.get('prediction_days', 3)):
            segment = df.iloc[i-60:i]
            features = calculate_features(segment)
            if features is None:
                continue
            future_close = df['Close'].iloc[i + settings.get('prediction_days', 3)]
            current_close = df['Close'].iloc[i]
            label = (future_close - current_close) / current_close
            X.append(get_feature_vector(features))
            y.append(label)
        if len(X) < 100:
            return None, None
        return np.array(X), np.array(y)
    except Exception as e:
        logger.error(f"準備 {ticker} 訓練資料失敗: {e}")
        return None, None

# ================== 情緒分析 (Reddit) ==================
def get_reddit_sentiment(ticker, subreddit='wallstreetbets', limit=5):
    if not TEXTBLOB_AVAILABLE:
        return 0.0
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    url = f"https://www.reddit.com/r/{subreddit}/search.json?q={ticker}&restrict_sr=1&limit={limit}"
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return 0.0
        data = resp.json()
        posts = data.get('data', {}).get('children', [])
        if not posts:
            return 0.0
        polarities = []
        for post in posts:
            title = post.get('data', {}).get('title', '')
            if title:
                blob = TextBlob(title)
                polarities.append(blob.sentiment.polarity)
        if polarities:
            return np.mean(polarities)
        return 0.0
    except Exception as e:
        logger.debug(f"Reddit 情緒分析失敗 {ticker}: {e}")
        return 0.0

def get_news_sentiment(ticker, limit=5):
    return 0.0

def get_combined_sentiment(ticker):
    reddit_score = get_reddit_sentiment(ticker)
    news_score = get_news_sentiment(ticker)
    combined = (reddit_score + news_score) / 2.0
    return max(-1.0, min(1.0, combined))

# ================== 籌碼分數 (Smart Money) ==================
def get_smart_money_score(ticker):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        inst_own = info.get('institutionPercent', None)
        if inst_own is not None:
            inst_score = inst_own * 100
        else:
            inst_score = 50
        short_float = info.get('shortPercentOfFloat', 0)
        if short_float:
            short_score = max(0, 100 - short_float * 100 / 0.3)
        else:
            short_score = 50
        smart_money = (inst_score * 0.5 + short_score * 0.5)
        return round(smart_money, 1)
    except:
        return 50.0

# ================== 熱度分數 (Hype) ==================
def get_hype_score(ticker, subreddit='wallstreetbets', limit=20):
    if not TEXTBLOB_AVAILABLE:
        return 50.0
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f"https://www.reddit.com/r/{subreddit}/search.json?q={ticker}&restrict_sr=1&limit={limit}"
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return 50.0
        data = resp.json()
        posts = data.get('data', {}).get('children', [])
        if not posts:
            return 50.0
        polarities = []
        for post in posts:
            title = post.get('data', {}).get('title', '')
            if title:
                blob = TextBlob(title)
                polarities.append(blob.sentiment.polarity)
        avg_polarity = np.mean(polarities) if polarities else 0.0
        hype = abs(avg_polarity) * 100
        hype = min(100, hype + min(len(posts), 20) / 2)
        return round(hype, 1)
    except Exception:
        return 50.0

# ================== 趨勢確認分數 ==================
def get_trend_score(df):
    if df.empty or len(df) < 200:
        return 50.0
    close = df['Close']
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    current_price = close.iloc[-1]
    rsi_series = calculate_rsi(close)
    rsi = rsi_series.iloc[-1] if not rsi_series.isna().all() else 50
    _, _, _, macd_status = calculate_macd(close)
    vwap = calculate_vwap(df)
    score = 0
    if current_price > ma20: score += 15
    if ma20 > ma50: score += 15
    if ma50 > ma200: score += 20
    if rsi > 60: score += 15
    if "金叉" in macd_status: score += 15
    if vwap is not None and current_price > vwap: score += 20
    return min(100, score)

# ================== 成長分數 (細化版) ==================
def get_growth_score(ticker, df):
    score = 0
    details = []
    try:
        tk = yf.Ticker(ticker)
        financials = tk.quarterly_financials
        revenue_growth = 0
        if not financials.empty and 'Total Revenue' in financials.index:
            revenues = financials.loc['Total Revenue'].dropna()
            if len(revenues) >= 2:
                latest = revenues.iloc[0]
                prev = revenues.iloc[1]
                revenue_growth = (latest - prev) / prev * 100
                if revenue_growth > 30:
                    score += 30
                    details.append(f"營收增速 {revenue_growth:.1f}%")
                elif revenue_growth > 15:
                    score += 20
                elif revenue_growth > 0:
                    score += 10
        income_stmt = tk.quarterly_income_stmt
        eps_growth = 0
        if not income_stmt.empty and 'Net Income' in income_stmt.index:
            earnings = income_stmt.loc['Net Income'].dropna()
            if len(earnings) >= 2:
                latest_eps = earnings.iloc[0]
                prev_eps = earnings.iloc[1]
                if prev_eps != 0:
                    eps_growth = (latest_eps - prev_eps) / abs(prev_eps) * 100
                    if eps_growth > 30:
                        score += 30
                        details.append(f"EPS增速 {eps_growth:.1f}%")
                    elif eps_growth > 15:
                        score += 20
                    elif eps_growth > 0:
                        score += 10
        if not income_stmt.empty and 'Gross Profit' in income_stmt.index and 'Total Revenue' in income_stmt.index:
            gp = income_stmt.loc['Gross Profit'].dropna()
            rev = income_stmt.loc['Total Revenue'].dropna()
            if len(gp) >= 2 and len(rev) >= 2:
                gm_latest = gp.iloc[0] / rev.iloc[0] * 100
                gm_prev = gp.iloc[1] / rev.iloc[1] * 100
                gm_change = gm_latest - gm_prev
                if gm_change > 5:
                    score += 20
                    details.append(f"毛利率 +{gm_change:.1f}%")
                elif gm_change > 0:
                    score += 10
        industry_keywords = ["AI", "SEMICONDUCTOR", "CLOUD", "NUCLEAR", "QUANTUM", "ROBOTICS", "BIOTECH"]
        sector_growth = 0
        try:
            sector = tk.info.get('sector', '')
            industry = tk.info.get('industry', '')
            full = (sector + " " + industry).upper()
            for kw in industry_keywords:
                if kw in full:
                    sector_growth = 20
                    break
        except:
            pass
        score += sector_growth
        if sector_growth > 0:
            details.append("高成長產業")
    except:
        pass
    final_score = min(100, max(0, score))
    return final_score, details

# ================== 十倍股潛力分數 (Ten-Bagger) ==================
def get_ten_bagger_score(ticker):
    ticker_upper = ticker.upper()
    score = 0
    keywords = {
        "AI": ["AI", "ARTIFICIAL INTELLIGENCE", "MACHINE LEARNING"],
        "ROBOTICS": ["ROBOT", "AUTOMATION"],
        "NUCLEAR": ["NUCLEAR", "OKLO", "SMR"],
        "QUANTUM": ["QUANTUM", "IONQ", "RGTI"],
        "CLOUD": ["CLOUD", "DATACENTER", "SERVER", "NVDA", "AMD", "MRVL", "VRT"],
        "BIOTECH": ["RECURSION", "CRISPR", "GENE", "RNA"],
        "BLOCKCHAIN": ["BLOCKCHAIN", "CRYPTO", "BITCOIN"]
    }
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        long_business_summary = info.get('longBusinessSummary', '')
        sector = info.get('sector', '')
        industry = info.get('industry', '')
        full_text = (ticker_upper + " " + sector + " " + industry + " " + long_business_summary).upper()
        for category, words in keywords.items():
            for w in words:
                if w in full_text:
                    score += 20
                    break
        if any(x in ticker_upper for x in ["NVDA", "AMD", "AVGO", "MRVL", "VRT", "SMCI", "PLTR", "OKLO", "IONQ"]):
            score += 30
    except:
        pass
    final_score = min(100, score)
    return final_score

# ================== XGBoost / LightGBM 訓練 ==================
def train_xgb_model(ticker, force_retrain=False):
    if not XGB_AVAILABLE:
        return None
    model_name = ticker.replace('.TW', '')
    model_path = os.path.join(MODELS_DIR, f"{model_name}_xgb.pkl")
    if not force_retrain and os.path.exists(model_path):
        mtime = os.path.getmtime(model_path)
        if time.time() - mtime < settings.get('retrain_hours', 24) * 3600:
            try:
                return joblib.load(model_path)
            except:
                pass
    df = get_cached_data(ticker, period="2y")
    if df.empty or len(df) < 100:
        return None
    X_list, y_list = [], []
    for i in range(60, len(df)-5):
        seg = df.iloc[i-60:i]
        feats = calculate_features(seg)
        if feats is None:
            continue
        future_ret = (df['Close'].iloc[i+3] / df['Close'].iloc[i]) - 1
        label = 1 if future_ret > 0.02 else 0
        X_list.append(get_feature_vector(feats))
        y_list.append(label)
    if len(X_list) < 100:
        return None
    X = np.array(X_list)
    y = np.array(y_list)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    joblib.dump(model, model_path)
    logger.info(f"{ticker} XGBoost 模型訓練完成")
    return model

def train_lgb_model(ticker, force_retrain=False):
    if not LGB_AVAILABLE:
        return None
    model_name = ticker.replace('.TW', '')
    model_path = os.path.join(MODELS_DIR, f"{model_name}_lgb.pkl")
    if not force_retrain and os.path.exists(model_path):
        mtime = os.path.getmtime(model_path)
        if time.time() - mtime < settings.get('retrain_hours', 24) * 3600:
            try:
                return joblib.load(model_path)
            except:
                pass
    df = get_cached_data(ticker, period="2y")
    if df.empty or len(df) < 100:
        return None
    X_list, y_list = [], []
    for i in range(60, len(df)-5):
        seg = df.iloc[i-60:i]
        feats = calculate_features(seg)
        if feats is None:
            continue
        future_ret = (df['Close'].iloc[i+3] / df['Close'].iloc[i]) - 1
        label = 1 if future_ret > 0.02 else 0
        X_list.append(get_feature_vector(feats))
        y_list.append(label)
    if len(X_list) < 100:
        return None
    X = np.array(X_list)
    y = np.array(y_list)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
    joblib.dump(model, model_path)
    logger.info(f"{ticker} LightGBM 模型訓練完成")
    return model

# ================== Ensemble 最終預測 (多因子 + 共識分數) ==================
def ensemble_predict(ticker, df):
    import math
    feats = calculate_features(df)
    if feats is None:
        return None, None, {}, None, None, None, None, None, None, None, None
    X = np.array(get_feature_vector(feats)).reshape(1, -1)
    
    scores = {}
    details = {}

    # RandomForest (轉為上漲機率)
    rf_model, rf_scaler = get_models_with_cache(ticker)
    if rf_model and rf_scaler:
        X_scaled = rf_scaler.transform(X)
        rf_pred = rf_model.predict(X_scaled)[0]
        rf_up_prob = 100 / (1 + math.exp(-rf_pred * 10))
        scores['RF'] = rf_up_prob
        details['RF'] = f"預期報酬 {rf_pred*100:+.1f}% → 上漲機率 {rf_up_prob:.0f}%"

    # XGBoost
    if XGB_AVAILABLE:
        xgb_model = train_xgb_model(ticker, force_retrain=False)
        if xgb_model:
            prob = xgb_model.predict_proba(X)[0][1]
            scores['XGB'] = prob * 100
            details['XGB'] = f"上漲機率 {prob*100:.1f}%"

    # LightGBM
    if LGB_AVAILABLE:
        lgb_model = train_lgb_model(ticker, force_retrain=False)
        if lgb_model:
            prob = lgb_model.predict_proba(X)[0][1]
            scores['LGB'] = prob * 100
            details['LGB'] = f"上漲機率 {prob*100:.1f}%"

    # 情緒分數
    sentiment = get_combined_sentiment(ticker)
    sent_score = (sentiment + 1) / 2 * 100
    scores['SENT'] = sent_score
    details['SENT'] = f"情緒 {sentiment:+.2f} → {sent_score:.0f}"

    # 籌碼分數
    smart_score = get_smart_money_score(ticker)
    scores['SMART'] = smart_score
    details['SMART'] = f"籌碼 {smart_score:.0f}"

    # 熱度分數
    hype_score = get_hype_score(ticker)
    scores['HYPE'] = hype_score
    details['HYPE'] = f"熱度 {hype_score:.0f}"

    # 趨勢確認分數
    trend_score = get_trend_score(df)
    scores['TREND'] = trend_score
    details['TREND'] = f"趨勢 {trend_score:.0f}"

    # 成長分數 (不加入加權，僅顯示)
    growth_score, growth_details = get_growth_score(ticker, df)

    # 十倍股潛力分數
    ten_bagger_score = get_ten_bagger_score(ticker)

    if not scores:
        return None, None, {}, None, None, None, None, None, None, None, None

    weights = {
        'RF': 0.20, 'XGB': 0.20, 'LGB': 0.15,
        'SMART': 0.20, 'SENT': 0.10, 'HYPE': 0.10,
        'TREND': 0.05
    }
    total_weight = sum(weights.get(m, 0) for m in scores if m in weights)
    if total_weight == 0:
        return None, None, {}, None, None, None, None, None, None, None, None

    final_score = sum(scores[m] * weights[m] for m in scores if m in weights) / total_weight
    final_score = round(final_score, 1)

    if final_score >= 85:
        signal = "強力買進"
    elif final_score >= 75:
        signal = "買進"
    elif final_score >= 65:
        signal = "試單買進"
    elif final_score >= 55:
        signal = "持有"
    elif final_score >= 45:
        signal = "觀望"
    else:
        signal = "賣出/避開"

    agreement_models = ['RF','XGB','LGB','SENT']
    bullish = sum(1 for m in agreement_models if m in scores and scores[m] >= 60)
    agreement = bullish / len(agreement_models)

    trend_consensus = 100 if trend_score > 60 else 0
    growth_consensus = 100 if (growth_score > 50 and hype_score > 50) else 0
    consensus_score = (agreement * 100 + trend_consensus + growth_consensus) / 3
    consensus_score = round(consensus_score, 1)

    return final_score, signal, details, agreement, smart_score, hype_score, trend_score, growth_score, ten_bagger_score, growth_details, consensus_score

# ================== 輔助函數 ==================
def safe_get_close(df):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return pd.Series(close.values.flatten(), index=close.index, name='Close').dropna()
    except:
        return pd.Series(dtype=float)

def get_us_stock_data(ticker):
    df = get_cached_data(ticker, period="2y")
    if df.empty:
        return 0, 0, 0, df
    try:
        tk = yf.Ticker(ticker)
        fast = tk.fast_info
        curr = fast.get('last_price', None)
        prev = fast.get('previous_close', None)
        if curr is not None and prev is not None:
            change = curr - prev
            pct = (change / prev) * 100 if prev != 0 else 0
            return curr, change, pct, df
    except:
        pass
    close_series = safe_get_close(df)
    if len(close_series) < 2:
        return 0, 0, 0, df
    curr = float(close_series.iloc[-1])
    prev = float(close_series.iloc[-2])
    change = curr - prev
    pct = (change / prev) * 100 if prev != 0 else 0
    return curr, change, pct, df

def get_tw_stock_data(ticker):
    curr, change, pct = get_tw_stock_realtime_shioaji(ticker)
    _, _, _, df_hist = get_us_stock_data(ticker)
    if curr is not None and curr > 0 and not math.isnan(curr):
        return curr, change, pct, df_hist
    else:
        if df_hist.empty or len(df_hist) < 2:
            return 0, 0, 0, df_hist
        curr = float(df_hist['Close'].iloc[-1])
        prev = float(df_hist['Close'].iloc[-2])
        if math.isnan(curr) or math.isnan(prev):
            return 0, 0, 0, df_hist
        change = curr - prev
        pct = (change / prev) * 100 if prev != 0 else 0
        return curr, change, pct, df_hist

def get_all_tickers(market):
    file_path = STOCK_FILE_TW if market == 'tw' else STOCK_FILE_US
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# ================== 背景更新 AI 評分 ==================
def compute_stock_summary(ticker, market):
    try:
        if market == 'tw':
            curr, change, pct, df = get_tw_stock_data(ticker)
        else:
            curr, change, pct, df = get_us_stock_data(ticker)
        if curr == 0 or df.empty:
            return None
        close = df['Close'].values
        if len(close) >= 15:
            deltas = np.diff(close[-15:])
            gain = np.mean(deltas[deltas > 0]) if len(deltas[deltas > 0]) > 0 else 0
            loss = -np.mean(deltas[deltas < 0]) if len(deltas[deltas < 0]) > 0 else 0
            rs = gain / loss if loss > 0 else 1
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50
        close_series = safe_get_close(df)
        if len(close_series) >= 35:
            _, _, _, macd_status = calculate_macd(close_series)
        else:
            macd_status = "-"
        vwap = calculate_vwap(df)
        vwap_status = get_vwap_status(curr, vwap) if vwap else "N/A"
        volume = df['Volume'].values
        vol_ma5 = np.mean(volume[-5:]) if len(volume) >= 5 else volume[-1]
        volume_ratio = volume[-1] / vol_ma5 if vol_ma5 > 0 else 1
        ai_score = calculate_ai_resonance_score(rsi, macd_status, vwap_status, volume_ratio)
        ma20 = np.mean(close[-20:]) if len(close) >= 20 else curr
        trend = "多頭" if curr > ma20 else "空頭"
        if ai_score >= 70 and trend == "多頭":
            signal = "買進"
        elif ai_score < 40 or trend == "空頭":
            signal = "賣出/避開"
        else:
            signal = "持有/觀望"
        return {
            'ticker': ticker,
            'price': round(curr, 2),
            'change': round(change, 2),
            'pct': round(pct, 2),
            'ai_score': ai_score,
            'trend': trend,
            'signal': signal,
            'market': market,
            'rsi': round(rsi,1),
            'macd_status': macd_status,
            'vwap_status': vwap_status,
            'volume_ratio': round(volume_ratio,2),
            'ma20': round(ma20,2)
        }
    except Exception as e:
        logger.error(f"計算 {ticker} 摘要失敗: {e}")
        return None

def update_all_ai_scores():
    all_data = []
    for market in ['tw', 'us']:
        tickers = get_all_tickers(market)
        for ticker in tickers:
            summary = compute_stock_summary(ticker, market)
            if summary:
                all_data.append(summary)
            time.sleep(0.1)
    with open(AI_SCORES_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    logger.info(f"背景更新完成，共更新 {len(all_data)} 檔股票")

# ================== 空頭回補與法人連買 ==================
def get_short_interest(ticker):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        short_float = info.get('shortPercentOfFloat', 0)
        if short_float:
            return short_float * 100
    except:
        pass
    return 0

def get_institutional_holders_tw(ticker):
    stock_id = ticker.replace('.TW', '')
    try:
        url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitution&stock_id={stock_id}&start_date={datetime.now().strftime('%Y-%m-%d')}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json().get('data', [])
            if data and len(data) >= 3:
                consecutive_buy = all(
                    d.get('foreign_investment', 0) + d.get('investment_trust', 0) +
                    d.get('dealer_self', 0) + d.get('dealer_hedge', 0) > 0
                    for d in data[:3]
                )
                return {'consecutive_buy': consecutive_buy}
    except:
        pass
    return None

# ================== 異常警報掃描 ==================
def scan_alerts():
    alerts = []
    if not os.path.exists(AI_SCORES_FILE):
        return alerts
    try:
        with open(AI_SCORES_FILE, 'r', encoding='utf-8') as f:
            stocks = json.load(f)
    except:
        return alerts

    for stock in stocks:
        ticker = stock['ticker']
        market = stock['market']
        if market == 'tw':
            curr, change, pct, df = get_tw_stock_data(ticker)
        else:
            curr, change, pct, df = get_us_stock_data(ticker)
        if df.empty:
            continue

        volume = df['Volume'].values
        vol_ma20 = np.mean(volume[-20:]) if len(volume) >= 20 else volume[-1]
        vol_ratio = volume[-1] / vol_ma20 if vol_ma20 > 0 else 1
        if vol_ratio > 3.0:
            alerts.append({"ticker": ticker, "type": "volume_surge", "message": f"成交量爆增 {vol_ratio:.1f}倍", "importance": "high"})

        close_series = safe_get_close(df)
        if len(close_series) >= 35:
            _, _, _, macd_status = calculate_macd(close_series)
            if "金叉" in macd_status:
                alerts.append({"ticker": ticker, "type": "macd_golden", "message": "MACD 黃金交叉", "importance": "medium"})

        high_52w = df['High'].rolling(252).max().iloc[-1]
        if curr >= high_52w * 0.99:
            alerts.append({"ticker": ticker, "type": "new_high", "message": "突破52週新高", "importance": "high"})

        if market == 'us':
            short_pct = get_short_interest(ticker)
            if short_pct > 15 and pct > 3:
                alerts.append({"ticker": ticker, "type": "short_squeeze", "message": f"空頭回補 (空單 {short_pct:.1f}%，漲幅 {pct:.1f}%)", "importance": "high"})

        if market == 'tw':
            inst = get_institutional_holders_tw(ticker)
            if inst and inst.get('consecutive_buy'):
                alerts.append({"ticker": ticker, "type": "institutional_buy", "message": "三大法人連買3天", "importance": "medium"})

    alerts.sort(key=lambda x: 0 if x['importance']=='high' else 1 if x['importance']=='medium' else 2)
    with open(ALERTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(alerts, f, indent=2, ensure_ascii=False)
    return alerts

# ================== 市場溫度計 ==================
def get_vix():
    try:
        df = yf.Ticker("^VIX").history(period="5d")
        if not df.empty:
            return df['Close'].iloc[-1]
    except:
        pass
    return 20.0

def get_treasury_yield():
    try:
        df = yf.Ticker("^TNX").history(period="5d")
        if not df.empty:
            return df['Close'].iloc[-1]
    except:
        pass
    return 4.2

def get_dollar_index():
    try:
        df = yf.Ticker("DX-Y.NYB").history(period="5d")
        if not df.empty:
            return df['Close'].iloc[-1]
    except:
        pass
    return 104.0

def get_market_trend():
    try:
        sp500 = yf.Ticker("^GSPC").history(period="1mo")
        nasdaq = yf.Ticker("^IXIC").history(period="1mo")
        if sp500.empty or nasdaq.empty:
            return 50
        sp_ret = (sp500['Close'].iloc[-1] - sp500['Close'].iloc[0]) / sp500['Close'].iloc[0] * 100
        nas_ret = (nasdaq['Close'].iloc[-1] - nasdaq['Close'].iloc[0]) / nasdaq['Close'].iloc[0] * 100
        score = 50 + (sp_ret + nas_ret) * 2
        return max(0, min(100, score))
    except:
        return 50

def calculate_market_temperature():
    vix = get_vix()
    yield_val = get_treasury_yield()
    dollar = get_dollar_index()
    trend = get_market_trend()
    vix_score = 30 if vix < 15 else (15 if vix < 25 else -20)
    yield_score = 20 if yield_val < 4 else (0 if yield_val < 5 else -15)
    dollar_score = 10 if dollar < 103 else (-10 if dollar > 105 else 0)
    total = trend + vix_score + yield_score + dollar_score
    total = max(0, min(100, total))
    if total >= 75:
        status = "🔴 風險偏好高，適合積極操作"
    elif total >= 45:
        status = "⚪ 中性市場，平衡配置"
    else:
        status = "🟢 避險情緒濃厚，降低持股"
    return round(total), status

# ================== 產業輪動 ==================
US_SECTOR_ETFS = {
    "AI": "AIQ",
    "電力": "XLU",
    "資料中心": "IDC",
    "網通": "IHAK",
    "機器人": "BOTZ",
    "核能": "NLR",
    "網路安全": "HACK"
}
TW_SECTOR_ETFS = {
    "半導體": "00830.TW",
    "電子": "0053.TW",
    "金融": "0055.TW",
    "傳產/中小": "00733.TW"
}

def get_us_sector_performance():
    results = []
    now = datetime.now()
    today_start = now.strftime("%Y-%m-%d")
    for name, etf in US_SECTOR_ETFS.items():
        try:
            df = yf.Ticker(etf).history(start=today_start, end=now.strftime("%Y-%m-%d"))
            if not df.empty and len(df) >= 2:
                ret = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
                results.append((name, round(ret, 2)))
            else:
                df = yf.Ticker(etf).history(period="2d")
                if not df.empty and len(df) >= 2:
                    ret = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
                    results.append((name, round(ret, 2)))
                else:
                    results.append((name, 0))
        except:
            results.append((name, 0))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def get_tw_sector_performance():
    results = []
    now = datetime.now()
    today_start = now.strftime("%Y-%m-%d")
    for name, etf in TW_SECTOR_ETFS.items():
        try:
            df = yf.Ticker(etf).history(start=today_start, end=now.strftime("%Y-%m-%d"))
            if not df.empty and len(df) >= 2:
                ret = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
                results.append((name, round(ret, 2)))
            else:
                df = yf.Ticker(etf).history(period="2d")
                if not df.empty and len(df) >= 2:
                    ret = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
                    results.append((name, round(ret, 2)))
                else:
                    results.append((name, 0))
        except:
            results.append((name, 0))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ================== 投資組合分析 ==================
def get_portfolio_analysis():
    positions = load_positions()
    if not positions:
        return {"total_value": 0, "cash": settings.get('cash_balance',0), "cash_ratio": 100, "sector_ratio": {}, "risk_score": 0, "concentration_warning": []}
    total_value = 0
    sector_map = defaultdict(float)
    for pos in positions:
        ticker = pos['ticker']
        market = 'tw' if '.TW' in ticker else 'us'
        if market == 'tw':
            curr, _, _, _ = get_tw_stock_data(ticker)
        else:
            curr, _, _, _ = get_us_stock_data(ticker)
        if curr == 0:
            curr = pos['cost']
        value = curr * pos['shares']
        total_value += value
        sector = "其他"
        if any(x in ticker for x in ["NVDA", "AMD", "INTC", "MRVL", "QCOM", "TSM"]):
            sector = "半導體"
        elif any(x in ticker for x in ["MSFT", "ORCL", "NOW", "CRM", "ADBE"]):
            sector = "軟體"
        elif any(x in ticker for x in ["TSLA", "RIVN", "LCID"]):
            sector = "電動車"
        elif any(x in ticker for x in ["OPEN", "DNUT", "PLTR"]):
            sector = "AI/大數據"
        elif "AI" in ticker:
            sector = "AI"
        sector_map[sector] += value
    concentration_warning = []
    for pos in positions:
        ticker = pos['ticker']
        market = 'tw' if '.TW' in ticker else 'us'
        if market == 'tw':
            curr, _, _, _ = get_tw_stock_data(ticker)
        else:
            curr, _, _, _ = get_us_stock_data(ticker)
        if curr == 0:
            curr = pos['cost']
        value = curr * pos['shares']
        ratio = (value / total_value) * 100 if total_value > 0 else 0
        if ratio > settings.get('max_concentration', 40):
            concentration_warning.append(f"{ticker} 佔比 {ratio:.1f}% 超過限制")
    cash = settings.get('cash_balance', 0)
    cash_ratio = (cash / (total_value + cash)) * 100 if (total_value + cash) > 0 else 100
    risk_score = 0
    if len(sector_map) < 3:
        risk_score += 30
    if concentration_warning:
        risk_score += 40
    risk_score = min(100, risk_score)
    return {
        "total_value": round(total_value, 2),
        "cash": cash,
        "cash_ratio": round(cash_ratio, 2),
        "sector_ratio": {k: round(v/total_value*100, 2) for k,v in sector_map.items()},
        "risk_score": risk_score,
        "concentration_warning": concentration_warning
    }

# ================== 風險控制 ==================
def get_risk_warnings():
    positions = load_positions()
    warnings = []
    if not positions:
        return warnings
    for pos in positions:
        ticker = pos['ticker']
        cost = pos['cost']
        market = 'tw' if '.TW' in ticker else 'us'
        if market == 'tw':
            curr, _, _, df = get_tw_stock_data(ticker)
        else:
            curr, _, _, df = get_us_stock_data(ticker)
        if curr == 0:
            curr = cost
        max_loss_pct = settings.get('max_loss_per_trade', 5.0)
        stop_price = cost * (1 - max_loss_pct / 100)
        if curr <= stop_price:
            warnings.append({"ticker": ticker, "type": "stop_loss", "message": f"觸發停損 ({curr:.2f} < {stop_price:.2f})", "current_loss": (curr-cost)/cost*100})
    portfolio = get_portfolio_analysis()
    for warn in portfolio['concentration_warning']:
        warnings.append({"ticker": "PORTFOLIO", "type": "concentration", "message": warn})
    return warnings

# ================== 機構級回測 ==================
def run_backtest_advanced(ticker, days=180, benchmark_ticker='^GSPC'):
    try:
        stock_df = get_cached_data(ticker, period="1y")
        if stock_df.empty or len(stock_df) < 50:
            return None
        try:
            bench_df = get_cached_data(benchmark_ticker, period="1y")
            bench_returns = bench_df['Close'].pct_change().dropna()
        except:
            bench_returns = None
        df_test = stock_df.tail(days)
        capital = 10000
        position = 0
        portfolio_value = []
        dates = []
        for i in range(20, len(df_test)):
            close = df_test['Close'].values[:i]
            if len(close) < 20:
                continue
            ma5 = np.mean(close[-5:])
            ma20 = np.mean(close[-20:])
            current_price = close[-1]
            deltas = np.diff(close[-15:]) if len(close) >= 15 else np.diff(close)
            gain = np.mean(deltas[deltas > 0]) if len(deltas[deltas > 0]) > 0 else 0
            loss = -np.mean(deltas[deltas < 0]) if len(deltas[deltas < 0]) > 0 else 0
            rs = gain / loss if loss > 0 else 1
            rsi = 100 - (100 / (1 + rs))
            buy_signal = current_price > ma5 and current_price > ma20 and rsi < 70
            sell_signal = current_price < ma5 or rsi > 70
            if buy_signal and position == 0 and capital > 0:
                shares = int(capital / current_price)
                if shares > 0:
                    cost = shares * current_price
                    capital -= cost
                    position = shares
            elif sell_signal and position > 0:
                capital += position * current_price
                position = 0
            current_value = capital + position * current_price
            portfolio_value.append(current_value)
            dates.append(df_test.index[i])
        portfolio_series = pd.Series(portfolio_value, index=dates)
        returns = portfolio_series.pct_change().dropna()
        total_return = (portfolio_series.iloc[-1] - 10000) / 10000 * 100
        years = len(returns) / 252
        cagr = (portfolio_series.iloc[-1] / 10000) ** (1/years) - 1 if years > 0 else 0
        cagr_pct = cagr * 100
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        risk_free_rate = 0.02 / 252
        excess_returns = returns - risk_free_rate
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() != 0 else 0
        downside_returns = returns[returns < 0]
        sortino = np.sqrt(252) * excess_returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() != 0 else 0
        if bench_returns is not None and len(bench_returns) >= len(returns):
            common_idx = returns.index.intersection(bench_returns.index)
            if len(common_idx) > 0:
                stock_ret = returns[common_idx]
                bench_ret = bench_returns[common_idx]
                cov = np.cov(stock_ret, bench_ret)[0,1]
                var = np.var(bench_ret)
                beta = cov / var if var != 0 else 1
                alpha = (stock_ret.mean() - risk_free_rate) - beta * (bench_ret.mean() - risk_free_rate)
                alpha_annual = alpha * 252 * 100
            else:
                beta = 1
                alpha_annual = 0
        else:
            beta = 1
            alpha_annual = 0
        return {
            'initial_capital': 10000,
            'final_value': round(portfolio_series.iloc[-1], 2),
            'total_return': round(total_return, 2),
            'cagr': round(cagr_pct, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe, 2),
            'sortino_ratio': round(sortino, 2),
            'alpha': round(alpha_annual, 2),
            'beta': round(beta, 2),
        }
    except Exception as e:
        logger.error(f"進階回測 {ticker} 失敗: {e}")
        return None

# ================== 十倍股雷達 ==================
def get_reddit_mentions(ticker):
    return {'change_24h': random.randint(-30, 80)}

def tenbagger_radar(ticker, market):
    try:
        if market == 'tw':
            curr, change, pct, df = get_tw_stock_data(ticker)
        else:
            curr, change, pct, df = get_us_stock_data(ticker)
        if df.empty or len(df) < 60:
            return None
        revenue_growth = 0
        try:
            tk = yf.Ticker(ticker)
            financials = tk.quarterly_financials
            if not financials.empty and 'Total Revenue' in financials.index:
                revenues = financials.loc['Total Revenue'].dropna()
                if len(revenues) >= 2:
                    latest = revenues.iloc[0]
                    prev = revenues.iloc[1]
                    revenue_growth = (latest - prev) / prev * 100
        except:
            pass
        gross_margin_growth = 0
        try:
            income_stmt = tk.quarterly_income_stmt
            if not income_stmt.empty and 'Gross Profit' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                gp = income_stmt.loc['Gross Profit'].dropna()
                rev = income_stmt.loc['Total Revenue'].dropna()
                if len(gp) >= 2 and len(rev) >= 2:
                    gm_latest = gp.iloc[0] / rev.iloc[0] * 100
                    gm_prev = gp.iloc[1] / rev.iloc[1] * 100
                    gross_margin_growth = gm_latest - gm_prev
        except:
            pass
        high_52w = df['High'].rolling(252).max().iloc[-1]
        is_52w_high = curr >= high_52w * 0.98
        volume = df['Volume'].values
        vol_ma20 = np.mean(volume[-20:]) if len(volume) >= 20 else volume[-1]
        volume_surge = volume[-1] > vol_ma20 * 1.5
        reddit_hot = get_reddit_mentions(ticker)['change_24h'] > 50
        short_high = False
        if market == 'us':
            short_pct = get_short_interest(ticker)
            short_high = short_pct > 15
        score = 0
        conditions = []
        if revenue_growth > 30:
            score += 1
            conditions.append(f"營收成長 {revenue_growth:.1f}%")
        if gross_margin_growth > 5:
            score += 1
            conditions.append(f"毛利率成長 {gross_margin_growth:.1f}%")
        if is_52w_high:
            score += 1
            conditions.append("52週新高")
        if volume_surge:
            score += 1
            conditions.append("爆量")
        if reddit_hot:
            score += 1
            conditions.append("Reddit熱議")
        if short_high:
            score += 1
            conditions.append("高空頭比例(軋空潛力)")
        return {
            'ticker': ticker,
            'price': round(curr, 2),
            'score': score,
            'conditions': conditions,
            'revenue_growth': round(revenue_growth, 1),
            'gm_growth': round(gross_margin_growth, 1),
            'is_52w_high': is_52w_high,
            'volume_surge': volume_surge,
            'reddit_hot': reddit_hot,
            'short_high': short_high
        }
    except Exception as e:
        logger.error(f"十倍股掃描 {ticker} 錯誤: {e}")
        return None

# ================== 均線糾結選股 ==================
def find_ma_bonding_stocks(tickers, ma_list=[5, 20, 60], threshold=0.03):
    results = []
    for ticker in tickers:
        try:
            df = yf.Ticker(ticker).history(period="3mo", auto_adjust=True)
            if df.empty or len(df) < max(ma_list):
                continue
            close = df['Close']
            volume = df['Volume'].values
            current_price = close.iloc[-1]
            ma_values = {}
            for ma in ma_list:
                ma_values[ma] = close.rolling(ma).mean().iloc[-1]
            ma_array = list(ma_values.values())
            ma_max = max(ma_array)
            ma_min = min(ma_array)
            ma_spread = (ma_max - ma_min) / ma_min if ma_min > 0 else 1
            if ma_spread < threshold:
                avg_volume = np.mean(volume[-20:]) if len(volume) >= 20 else volume[-1]
                volume_confirm = volume[-1] > avg_volume * 1.2
                if current_price > ma_max:
                    direction = "向上突破可能"
                elif current_price < ma_min:
                    direction = "向下突破可能"
                else:
                    direction = "糾結區內待方向"
                results.append({
                    'ticker': ticker,
                    'price': round(current_price, 2),
                    'spread': round(ma_spread * 100, 2),
                    'ma5': round(ma_values.get(5, 0), 2),
                    'ma20': round(ma_values.get(20, 0), 2),
                    'ma60': round(ma_values.get(60, 0), 2),
                    'direction': direction,
                    'volume_confirm': volume_confirm
                })
        except:
            continue
        time.sleep(0.03)
    results.sort(key=lambda x: x['spread'])
    return results

def get_all_tw_stocks_from_api():
    try:
        url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            stocks = []
            for item in data:
                code = item.get("公司代號", "")
                if code and len(code) <= 4 and code.isdigit():
                    stocks.append(f"{code}.TW")
            return stocks
    except:
        pass
    return ["2330.TW", "2317.TW", "2454.TW"]

def get_all_us_stocks():
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

# ================== 持倉與觀察名單管理 ==================
def load_positions():
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE, "r", encoding='utf-8') as f:
            return json.load(f)
    return []

def save_positions(positions):
    with open(POSITIONS_FILE, "w", encoding='utf-8') as f:
        json.dump(positions, f, indent=2, ensure_ascii=False)

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r", encoding='utf-8') as f:
            return json.load(f)
    return ["NVDA", "TSLA", "OPEN", "MRVL", "PLTR", "ANET", "VRT", "SMCI", "AMD"]

def save_watchlist(watchlist):
    with open(WATCHLIST_FILE, "w", encoding='utf-8') as f:
        json.dump(watchlist, f, indent=2, ensure_ascii=False)

# ================== 批次新增 ==================
def add_multiple_stocks(tickers_str, market):
    tickers = re.split(r'[,\s\n]+', tickers_str)
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    file_path = STOCK_FILE_TW if market == 'tw' else STOCK_FILE_US
    with open(file_path, "r", encoding="utf-8") as f:
        existing = [line.strip().upper() for line in f if line.strip()]
    new_tickers = [t for t in tickers if t not in existing]
    with open(file_path, "a", encoding="utf-8") as f:
        for t in new_tickers:
            f.write(f"{t}\n")
    return len(new_tickers), new_tickers

# ================== 指數獲取 ==================
def get_tw_index():
    try:
        df = yf.Ticker("^TWII").history(period="2d", auto_adjust=True)
        if not df.empty:
            curr = df['Close'].iloc[-1]
            prev = df['Close'].iloc[-2]
            change = curr - prev
            pct = (change / prev) * 100
            return round(curr, 2), round(change, 2), round(pct, 2)
    except:
        pass
    return 0, 0, 0

def get_otc_index():
    try:
        df = yf.Ticker("^TWOII").history(period="2d", auto_adjust=True)
        if not df.empty:
            curr = df['Close'].iloc[-1]
            prev = df['Close'].iloc[-2]
            change = curr - prev
            pct = (change / prev) * 100
            return round(curr, 2), round(change, 2), round(pct, 2)
    except:
        try:
            df = yf.Ticker("006201.TW").history(period="2d", auto_adjust=True)
            if not df.empty:
                curr = df['Close'].iloc[-1]
                prev = df['Close'].iloc[-2]
                change = curr - prev
                pct = (change / prev) * 100
                return round(curr, 2), round(change, 2), round(pct, 2)
        except:
            pass
    return 0, 0, 0

def get_dow_index():
    try:
        df = yf.Ticker("^DJI").history(period="2d", auto_adjust=True)
        if not df.empty:
            curr = df['Close'].iloc[-1]
            prev = df['Close'].iloc[-2]
            change = curr - prev
            pct = (change / prev) * 100
            return round(curr, 2), round(change, 2), round(pct, 2)
    except:
        pass
    return 0, 0, 0

def get_nasdaq_index():
    try:
        df = yf.Ticker("^IXIC").history(period="2d", auto_adjust=True)
        if not df.empty:
            curr = df['Close'].iloc[-1]
            prev = df['Close'].iloc[-2]
            change = curr - prev
            pct = (change / prev) * 100
            return round(curr, 2), round(change, 2), round(pct, 2)
    except:
        pass
    return 0, 0, 0

def get_phlx_index():
    try:
        df = yf.Ticker("^SOX").history(period="2d", auto_adjust=True)
        if not df.empty:
            curr = df['Close'].iloc[-1]
            prev = df['Close'].iloc[-2]
            change = curr - prev
            pct = (change / prev) * 100
            return round(curr, 2), round(change, 2), round(pct, 2)
    except:
        pass
    return 0, 0, 0

# ================== 路由 ==================
@app.route("/add_stock_tw", methods=["POST"])
def add_stock_tw():
    tickers_input = request.form.get("ticker", "").strip().upper()
    if not tickers_input:
        return redirect(url_for('index'))
    if ',' in tickers_input or '\n' in tickers_input or ' ' in tickers_input:
        added, new_list = add_multiple_stocks(tickers_input, 'tw')
        if added == 0:
            return "⚠️ 所有輸入的台股代號都已存在，未新增任何股票。<br><a href='/tw'>返回台股監控</a>"
        return f"✅ 成功新增 {added} 檔台股：{', '.join(new_list)}<br><a href='/tw'>返回台股監控</a>"
    else:
        with open(STOCK_FILE_TW, "r", encoding="utf-8") as f:
            existing = [line.strip().upper() for line in f if line.strip()]
        if tickers_input in existing:
            return f"⚠️ 台股 {tickers_input} 已在自選股清單中，未重複新增。<br><a href='/tw'>返回台股監控</a>"
        with open(STOCK_FILE_TW, "a", encoding="utf-8") as f:
            f.write(f"{tickers_input}\n")
        return f"✅ 成功新增台股 {tickers_input}<br><a href='/tw'>返回台股監控</a>"

@app.route("/add_stock_us", methods=["POST"])
def add_stock_us():
    tickers_input = request.form.get("ticker", "").strip().upper()
    if not tickers_input:
        return redirect(url_for('index'))
    if ',' in tickers_input or '\n' in tickers_input or ' ' in tickers_input:
        added, new_list = add_multiple_stocks(tickers_input, 'us')
        if added == 0:
            return "⚠️ 所有輸入的美股代號都已存在，未新增任何股票。<br><a href='/us'>返回美股監控</a>"
        return f"✅ 成功新增 {added} 檔美股：{', '.join(new_list)}<br><a href='/us'>返回美股監控</a>"
    else:
        with open(STOCK_FILE_US, "r", encoding="utf-8") as f:
            existing = [line.strip().upper() for line in f if line.strip()]
        if tickers_input in existing:
            return f"⚠️ 美股 {tickers_input} 已在自選股清單中，未重複新增。<br><a href='/us'>返回美股監控</a>"
        with open(STOCK_FILE_US, "a", encoding="utf-8") as f:
            f.write(f"{tickers_input}\n")
        return f"✅ 成功新增美股 {tickers_input}<br><a href='/us'>返回美股監控</a>"

@app.route("/add_position", methods=["POST"])
def add_position():
    try:
        ticker = request.form.get("pos_ticker", "").strip().upper()
        cost_str = request.form.get("cost", "").strip()
        shares_str = request.form.get("shares", "").strip()
        if not ticker or not cost_str or not shares_str:
            return "❌ 請填寫完整資訊", 400
        cost = float(cost_str)
        shares = int(shares_str)
        if cost <= 0 or shares <= 0:
            return "❌ 成本與股數必須為正數", 400
        positions = load_positions()
        found = False
        for p in positions:
            if p['ticker'] == ticker:
                p['cost'] = cost
                p['shares'] = shares
                found = True
                break
        if not found:
            positions.append({"ticker": ticker, "cost": cost, "shares": shares})
        save_positions(positions)
        return redirect(url_for('positions_page'))
    except ValueError:
        return "❌ 成本需為數字，股數需為整數", 400
    except Exception as e:
        return f"❌ 新增持倉失敗: {str(e)}", 500

@app.route("/clear_position/<ticker>")
def clear_position(ticker):
    positions = load_positions()
    new_positions = [p for p in positions if p['ticker'] != ticker.upper()]
    save_positions(new_positions)
    return redirect(url_for('positions_page'))

@app.route("/delete/tw/<ticker>")
def delete_tw(ticker):
    with open(STOCK_FILE_TW, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(STOCK_FILE_TW, "w", encoding="utf-8") as f:
        for line in lines:
            if line.strip().upper() != ticker.upper():
                f.write(line)
    return redirect(url_for('tw_page'))

@app.route("/delete/us/<ticker>")
def delete_us(ticker):
    with open(STOCK_FILE_US, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(STOCK_FILE_US, "w", encoding="utf-8") as f:
        for line in lines:
            if line.strip().upper() != ticker.upper():
                f.write(line)
    return redirect(url_for('us_page'))

@app.route("/retrain_all")
def retrain_all_models():
    tw_tickers = get_all_tickers('tw')
    us_tickers = get_all_tickers('us')
    all_tickers = tw_tickers + us_tickers
    results = []
    for ticker in all_tickers:
        rf_model, rf_scaler = train_single_model(ticker, force_retrain=True)
        xgb_model = train_xgb_model(ticker, force_retrain=True)
        lgb_model = train_lgb_model(ticker, force_retrain=True)
        ok = (rf_model is not None) or (xgb_model is not None) or (lgb_model is not None)
        results.append(f"{ticker}: {'部分成功' if ok else '失敗'}")
    with _model_lock:
        _model_cache.clear()
    return render_template_string("<br>".join(results))

@app.route("/clear_model_cache")
def clear_model_cache():
    with _model_lock:
        _model_cache.clear()
    return "✅ 模型快取已清除！下次預測將重新載入模型。"

# ================== 主要頁面 ==================
@app.route('/')
def index():
    if os.path.exists(AI_SCORES_FILE):
        try:
            with open(AI_SCORES_FILE, 'r', encoding='utf-8') as f:
                all_scores = json.load(f)
        except:
            all_scores = []
    else:
        all_scores = []
    tw_stocks = [s for s in all_scores if s.get('market') == 'tw']
    us_stocks = [s for s in all_scores if s.get('market') == 'us']
    tw_stocks.sort(key=lambda x: x.get('ai_score', 0), reverse=True)
    us_stocks.sort(key=lambda x: x.get('ai_score', 0), reverse=True)
    if os.path.exists(ALERTS_FILE):
        try:
            with open(ALERTS_FILE, 'r', encoding='utf-8') as f:
                alerts = json.load(f)
        except:
            alerts = []
    else:
        alerts = []
    market_temp, market_status = calculate_market_temperature()
    us_sector_perf = get_us_sector_performance()
    tw_sector_perf = get_tw_sector_performance()
    portfolio = get_portfolio_analysis()
    risk_warnings = get_risk_warnings()
    vix = get_vix()
    treasury = get_treasury_yield()
    dollar = get_dollar_index()
    shioaji_status = get_shioaji_status()
    return render_template_string(INDEX_HTML,
                                 tw_stocks=tw_stocks[:20], us_stocks=us_stocks[:20],
                                 alerts=alerts[:10], market_temp=market_temp, market_status=market_status,
                                 us_sector_perf=us_sector_perf, tw_sector_perf=tw_sector_perf,
                                 portfolio=portfolio, risk_warnings=risk_warnings,
                                 vix=vix, treasury=treasury, dollar=dollar, shioaji_status=shioaji_status)

@app.route('/tw')
def tw_page():
    tw_curr, tw_change, tw_pct = get_tw_index()
    otc_curr, otc_change, otc_pct = get_otc_index()
    dow_curr, dow_change, dow_pct = get_dow_index()
    nas_curr, nas_change, nas_pct = get_nasdaq_index()
    sox_curr, sox_change, sox_pct = get_phlx_index()
    if os.path.exists(AI_SCORES_FILE):
        try:
            with open(AI_SCORES_FILE, 'r', encoding='utf-8') as f:
                all_stocks = json.load(f)
        except:
            all_stocks = []
        stocks = [s for s in all_stocks if s.get('market') == 'tw']
    else:
        stocks = []
    stocks.sort(key=lambda x: x.get('ai_score', 0), reverse=True)
    return render_template_string(TW_US_HTML, market='tw', stocks=stocks, title="台股監控 - 所有自選股",
                                 tw_curr=tw_curr, tw_change=tw_change, tw_pct=tw_pct,
                                 otc_curr=otc_curr, otc_change=otc_change, otc_pct=otc_pct,
                                 dow_curr=dow_curr, dow_change=dow_change, dow_pct=dow_pct,
                                 nas_curr=nas_curr, nas_change=nas_change, nas_pct=nas_pct,
                                 sox_curr=sox_curr, sox_change=sox_change, sox_pct=sox_pct)

@app.route('/us')
def us_page():
    tw_curr, tw_change, tw_pct = get_tw_index()
    otc_curr, otc_change, otc_pct = get_otc_index()
    dow_curr, dow_change, dow_pct = get_dow_index()
    nas_curr, nas_change, nas_pct = get_nasdaq_index()
    sox_curr, sox_change, sox_pct = get_phlx_index()
    if os.path.exists(AI_SCORES_FILE):
        try:
            with open(AI_SCORES_FILE, 'r', encoding='utf-8') as f:
                all_stocks = json.load(f)
        except:
            all_stocks = []
        stocks = [s for s in all_stocks if s.get('market') == 'us']
    else:
        stocks = []
    stocks.sort(key=lambda x: x.get('ai_score', 0), reverse=True)
    return render_template_string(TW_US_HTML, market='us', stocks=stocks, title="美股監控 - 所有自選股",
                                 tw_curr=tw_curr, tw_change=tw_change, tw_pct=tw_pct,
                                 otc_curr=otc_curr, otc_change=otc_change, otc_pct=otc_pct,
                                 dow_curr=dow_curr, dow_change=dow_change, dow_pct=dow_pct,
                                 nas_curr=nas_curr, nas_change=nas_change, nas_pct=nas_pct,
                                 sox_curr=sox_curr, sox_change=sox_change, sox_pct=sox_pct)

@app.route('/ai_ranking')
def ai_ranking():
    if os.path.exists(AI_SCORES_FILE):
        try:
            with open(AI_SCORES_FILE, 'r', encoding='utf-8') as f:
                stocks = json.load(f)
        except:
            stocks = []
    else:
        stocks = []
    stocks.sort(key=lambda x: x.get('ai_score', 0), reverse=True)
    return render_template_string(AI_RANKING_HTML, stocks=stocks)

@app.route('/tenbagger')
def tenbagger_radar_page():
    all_candidates = []
    for market in ['tw', 'us']:
        tickers = get_all_tickers(market)
        for ticker in tickers:
            radar = tenbagger_radar(ticker, market)
            if radar and radar['score'] >= 3:
                all_candidates.append(radar)
    all_candidates.sort(key=lambda x: x['score'], reverse=True)
    return render_template_string(TENBAGGER_HTML, candidates=all_candidates)

@app.route('/backtest/<ticker>')
def backtest_page(ticker):
    benchmark = '^GSPC' if '.TW' not in ticker else '^TWII'
    result = run_backtest_advanced(ticker, days=180, benchmark_ticker=benchmark)
    if result is None:
        return render_template_string(BACKTEST_HTML, ticker=ticker, error="資料不足，無法回測")
    return render_template_string(BACKTEST_HTML, ticker=ticker, result=result)

@app.route('/positions')
def positions_page():
    positions = load_positions()
    enriched = []
    for pos in positions:
        ticker = pos['ticker']
        market = 'tw' if '.TW' in ticker else 'us'
        if market == 'tw':
            curr, change, pct, _ = get_tw_stock_data(ticker)
        else:
            curr, change, pct, _ = get_us_stock_data(ticker)
        if curr == 0:
            curr = pos['cost']
        value = curr * pos['shares']
        cost_value = pos['cost'] * pos['shares']
        profit = value - cost_value
        profit_pct = (profit / cost_value) * 100 if cost_value > 0 else 0
        stop_loss = pos['cost'] * (1 - settings.get('max_loss_per_trade', 5.0)/100)
        enriched.append({
            'ticker': ticker,
            'shares': pos['shares'],
            'cost': round(pos['cost'], 2),
            'current_price': round(curr, 2),
            'value': round(value, 2),
            'cost_value': round(cost_value, 2),
            'profit': round(profit, 2),
            'profit_pct': round(profit_pct, 2),
            'stop_loss': round(stop_loss, 2)
        })
    return render_template_string(POSITIONS_HTML, positions=enriched)

@app.route('/indicators/<ticker>')
def indicators_page(ticker):
    market = 'tw' if '.TW' in ticker else 'us'
    if market == 'tw':
        curr, change, pct, df = get_tw_stock_data(ticker)
    else:
        curr, change, pct, df = get_us_stock_data(ticker)
    if df.empty or curr == 0:
        return render_template_string(INDICATORS_HTML, ticker=ticker, error="無法取得資料")
    close_series = safe_get_close(df)
    if len(close_series) >= 15:
        deltas = np.diff(close_series[-15:])
        gain = np.mean(deltas[deltas > 0]) if len(deltas[deltas > 0]) > 0 else 0
        loss = -np.mean(deltas[deltas < 0]) if len(deltas[deltas < 0]) > 0 else 0
        rs = gain / loss if loss > 0 else 1
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50
    if rsi < 30:
        rsi_status = "🔴 超賣 (可能反彈)"
    elif rsi < 50:
        rsi_status = "🟡 弱勢"
    elif rsi < 70:
        rsi_status = "🟢 強勢"
    else:
        rsi_status = "⚠️ 超買 (注意回檔)"
    macd_status = "-"
    macd_hist = 0
    if len(close_series) >= 35:
        _, _, hist_val, macd_status = calculate_macd(close_series)
        macd_hist = round(hist_val, 3) if hist_val else 0
    vwap = calculate_vwap(df)
    vwap_status = get_vwap_status(curr, vwap) if vwap else "N/A"
    close = df['Close'].values
    ma20 = np.mean(close[-20:]) if len(close) >= 20 else curr
    ma60 = np.mean(close[-60:]) if len(close) >= 60 else curr
    if len(close) >= 200:
        ma200 = np.mean(close[-200:])
        ma200_display = round(ma200, 2)
    else:
        ma200_display = "資料不足"
    volume = df['Volume'].values
    vol_ma20 = np.mean(volume[-20:]) if len(volume) >= 20 else volume[-1]
    volume_ratio = volume[-1] / vol_ma20 if vol_ma20 > 0 else 1
    sr = calculate_support_resistance(df, curr)
    ai_score = calculate_ai_resonance_score(rsi, macd_status, vwap_status, volume_ratio)

    # RandomForest 預測
    ml_pred = None
    ml_trend = None
    ml_confidence = None
    ml_details = None
    if settings.get('enable_ensemble', True):
        pred_change, trend, conf, details = predict_with_ensemble(ticker, df)
        if pred_change is not None:
            ml_pred = f"{pred_change:+.1f}%"
            ml_trend = trend
            ml_confidence = f"信心:{conf}%"
            ml_details = details

    # 多因子預測
    ensemble_result = ensemble_predict(ticker, df)
    if ensemble_result[0] is None:
        ensemble_score = None
        ensemble_signal = None
        ensemble_details = {}
        agreement = 0.0
        smart_score = 50
        hype_score = 50
        trend_score = 50
        growth_score = 50
        ten_bagger_score = 50
        growth_details = []
        consensus_score = 0
    else:
        (ensemble_score, ensemble_signal, ensemble_details, agreement,
         smart_score, hype_score, trend_score, growth_score, ten_bagger_score, growth_details, consensus_score) = ensemble_result

    reasons = []
    if rsi < 30:
        reasons.append("RSI超賣區")
    elif rsi > 70:
        reasons.append("RSI超買區")
    else:
        reasons.append("RSI中性")
    if "金叉" in macd_status:
        reasons.append("MACD黃金交叉")
    elif "死叉" in macd_status:
        reasons.append("MACD死亡交叉")
    else:
        reasons.append("MACD平穩")
    if "站上" in vwap_status:
        reasons.append("站上VWAP")
    else:
        reasons.append("跌破VWAP")
    if volume_ratio > 1.5:
        reasons.append(f"量能放大 {volume_ratio:.1f}倍")
    reasons.append("AI綜合評分")
    curr = round(curr, 2)
    change = round(change, 2)
    pct = round(pct, 2)

    return render_template_string(INDICATORS_HTML,
        ticker=ticker, price=curr, change=change, pct=pct,
        rsi=round(rsi,1), rsi_status=rsi_status,
        macd_status=macd_status, macd_hist=macd_hist,
        vwap_status=vwap_status, ma20=round(ma20,2), ma60=round(ma60,2), ma200=ma200_display,
        volume_ratio=round(volume_ratio,2), ai_score=ai_score, reasons=reasons,
        resistance=sr['resistance'], target=sr['target'], stop_loss=sr['stop_loss'],
        ml_pred=ml_pred, ml_trend=ml_trend, ml_confidence=ml_confidence, ml_details=ml_details,
        ensemble_score=ensemble_score, ensemble_signal=ensemble_signal, ensemble_details=ensemble_details,
        agreement=agreement, smart_score=smart_score, hype_score=hype_score,
        trend_score=trend_score, growth_score=growth_score, ten_bagger_score=ten_bagger_score, growth_details=growth_details,
        consensus_score=consensus_score)

@app.route('/bonding')
def bonding_page():
    threshold = settings.get('bonding_threshold', 3.0) / 100
    tw_tickers = get_all_tw_stocks_from_api()
    us_tickers = get_all_us_stocks()
    tw_results = find_ma_bonding_stocks(tw_tickers, threshold=threshold)
    us_results = find_ma_bonding_stocks(us_tickers, threshold=threshold)
    return render_template_string(BONDING_HTML, tw_results=tw_results, us_results=us_results,
                                 tw_count=len(tw_results), us_count=len(us_results),
                                 tw_total=len(tw_tickers), us_total=len(us_tickers),
                                 threshold=settings.get('bonding_threshold', 3.0))

@app.route('/settings', methods=['GET', 'POST'])
def settings_page():
    if request.method == 'POST':
        settings['refresh_seconds'] = int(request.form.get('refresh_seconds', 60))
        settings['bonding_threshold'] = float(request.form.get('bonding_threshold', 3.0))
        settings['enable_ensemble'] = 'enable_ensemble' in request.form
        settings['prediction_days'] = int(request.form.get('prediction_days', 3))
        settings['train_days'] = int(request.form.get('train_days', 180))
        settings['retrain_hours'] = int(request.form.get('retrain_hours', 24))
        settings['auto_clear_cache_hours'] = int(request.form.get('auto_clear_cache_hours', 24))
        settings['background_update_minutes'] = int(request.form.get('background_update_minutes', 15))
        settings['cache_ttl_minutes'] = int(request.form.get('cache_ttl_minutes', 5))
        settings['max_loss_per_trade'] = float(request.form.get('max_loss_per_trade', 5.0))
        settings['max_loss_per_day'] = float(request.form.get('max_loss_per_day', 10.0))
        settings['max_concentration'] = float(request.form.get('max_concentration', 40.0))
        settings['cash_balance'] = float(request.form.get('cash_balance', 0.0))
        save_settings(settings)
        watchlist = request.form.get('watchlist', '').strip().upper()
        if watchlist:
            new_watchlist = [t.strip() for t in watchlist.split(',') if t.strip()]
            save_watchlist(new_watchlist)
        return redirect(url_for('settings_page'))
    watchlist = load_watchlist()
    return render_template_string(SETTINGS_HTML, settings=settings, watchlist=','.join(watchlist))

# ================== HTML 模板 ==================
INDEX_HTML = '''
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>AI量化監控 - 儀表板</title>
<style>
body{background:#121214;color:#fff;font-family:sans-serif;padding:20px;}
.card{background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:20px;}
table{width:100%;border-collapse:collapse;margin-top:20px;}
th,td{padding:10px;text-align:left;border-bottom:1px solid #333;}
th{background:#2a2a35;}
.positive{color:#ff4444;}
.negative{color:#00ff00;}
.ai-score{font-weight:bold;}
.btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;margin:5px;display:inline-block;}
.btn:hover{background:#3a3a45;}
.stock-link{color:#00ffaa;text-decoration:none;font-weight:bold;}
.stock-link:hover{color:#00ff66;text-decoration:underline;}
.warning{color:#ffaa00;}
.alert{background:#2a1a1a;border-left:4px solid #ff4444;padding:8px;margin:5px 0;}
</style>
</head>
<body>
<h1>🤖 AI 量化監控儀表板</h1>
<div class="card">
    <a href="/tw" class="btn">📋 全台股</a>
    <a href="/us" class="btn">📋 全美股</a>
    <a href="/ai_ranking" class="btn">🏆 AI 選股排行榜</a>
    <a href="/tenbagger" class="btn">🚀 十倍股雷達</a>
    <a href="/positions" class="btn">📊 持倉中心</a>
    <a href="/bonding" class="btn">📈 均線糾結選股</a>
    <a href="/settings" class="btn">⚙️ 設定</a>
    <a href="/retrain_all" class="btn" style="background:#ff9800;">🔄 重新訓練所有模型</a>
    <a href="/clear_model_cache" class="btn" style="background:#ff9800;">🗑️ 清除模型快取</a>
</div>
<div class="card">
    <h2>📊 市場溫度計</h2>
    <div>市場綜合評分: <strong>{{ market_temp }}/100</strong> - {{ market_status }}</div>
    <div>VIX: {{ vix|round(1) }} | 美債殖利率: {{ treasury|round(2) }}% | 美元指數: {{ dollar|round(1) }}</div>
    <div>🇹🇼 台股報價: {{ shioaji_status }}</div>
</div>
<div class="card">
    <h2>🏭 美股產業輪動 (今日)</h2>
    <div>{% for name, ret in us_sector_perf %}{{ name }}: {{ ret }}% &nbsp;|&nbsp; {% endfor %}</div>
</div>
<div class="card">
    <h2>🏭 台股產業輪動 (今日)</h2>
    <div>{% for name, ret in tw_sector_perf %}{{ name }}: {{ ret }}% &nbsp;|&nbsp; {% endfor %}</div>
</div>
<div class="card">
    <h2>⚠️ 風險控制</h2>
    {% for w in risk_warnings %}<div class="alert">🚨 {{ w.message }}</div>{% else %}<div>✅ 無重大風險</div>{% endfor %}
</div>
<div class="card">
    <h2>🚨 異常警報</h2>
    {% for a in alerts %}<div class="alert">🚨 {{ a.ticker }} - {{ a.message }}</div>{% else %}<div>暫無警報</div>{% endfor %}
</div>
<div class="card">
    <h2>🇹🇼 台股自選股 (AI 評分前20)</h2>
    <table>
        <thead><tr><th>代號</th><th>價格</th><th>漲跌幅</th><th>AI評分</th><th>趨勢</th><th>買賣訊號</th></tr></thead>
        <tbody>{% for s in tw_stocks %}
        <tr>
            <td><a href="/indicators/{{ s.ticker }}" class="stock-link">{{ s.ticker }}</a></td>
            <td>{{ s.price }}</td>
            <td class="{{ 'positive' if s.pct>=0 else 'negative' }}">{{ ('+' if s.pct>0 else '') ~ s.pct }}%</a>
            <td class="ai-score" style="color: {{ '#00ff00' if s.ai_score>=70 else '#ffcc00' if s.ai_score>=50 else '#ff4444' }}">{{ s.ai_score }}/100</a>
            <td>{{ s.trend }}</a>
            <td>{{ s.signal }}</a>
        </tr>
        {% else %}
        <tr><td colspan="6">尚無台股自選股，請點擊「全台股」加入</a></a>
        {% endfor %}</tbody>
    </table>
</div>
<div class="card">
    <h2>🇺🇸 美股自選股 (AI 評分前20)</h2>
    <table>
        <thead><tr><th>代號</th><th>價格</th><th>漲跌幅</th><th>AI評分</th><th>趨勢</th><th>買賣訊號</th></tr></thead>
        <tbody>{% for s in us_stocks %}
        <tr>
            <td><a href="/indicators/{{ s.ticker }}" class="stock-link">{{ s.ticker }}</a></a>
            <td>{{ s.price }}</a>
            <td class="{{ 'positive' if s.pct>=0 else 'negative' }}">{{ ('+' if s.pct>0 else '') ~ s.pct }}%</a>
            <td class="ai-score" style="color: {{ '#00ff00' if s.ai_score>=70 else '#ffcc00' if s.ai_score>=50 else '#ff4444' }}">{{ s.ai_score }}/100</a>
            <td>{{ s.trend }}</a>
            <td>{{ s.signal }}</a>
        </tr>
        {% else %}
        <tr><td colspan="6">尚無美股自選股，請點擊「全美股」加入</a></a>
        {% endfor %}</tbody>
    </table>
</div>
</body>
</html>
'''

TW_US_HTML = '''
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>{{ title }}</title>
<style>
body{background:#121214;color:#fff;font-family:sans-serif;padding:20px;}
.card{background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:20px;}
.index-grid{display:flex;gap:15px;flex-wrap:wrap;margin-bottom:20px;}
.index-item{background:#2a2a35;padding:8px 15px;border-radius:8px;min-width:150px;}
.positive{color:#ff4444;}
.negative{color:#00ff00;}
table{width:100%;border-collapse:collapse;margin-top:20px;}
th,td{padding:10px;text-align:left;border-bottom:1px solid #333;}
th{background:#2a2a35;}
.btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;margin:5px;display:inline-block;}
.btn:hover{background:#3a3a45;}
.stock-link{color:#00ffaa;text-decoration:none;font-weight:bold;}
.stock-link:hover{color:#00ff66;}
</style>
</head>
<body>
<h1>{{ title }}</h1>
<div><a href="/" class="btn">返回儀表板</a> <a href="/ai_ranking" class="btn">🏆 AI排行榜</a></div>
<div class="card">
    <h2>📈 指數行情</h2>
    <div class="index-grid">
        <div class="index-item"><div style="font-size:0.7rem;color:#aaa;">加權指數</div><div style="font-weight:bold;">{{ tw_curr }}</div><div class="{{ 'positive' if tw_change>=0 else 'negative' }}">{{ ('+' if tw_change>0 else '') ~ tw_change }} ({{ ('+' if tw_pct>0 else '') ~ tw_pct }}%)</div></div>
        <div class="index-item"><div style="font-size:0.7rem;color:#aaa;">櫃檯指數</div><div style="font-weight:bold;">{{ otc_curr }}</div><div class="{{ 'positive' if otc_change>=0 else 'negative' }}">{{ ('+' if otc_change>0 else '') ~ otc_change }} ({{ ('+' if otc_pct>0 else '') ~ otc_pct }}%)</div></div>
        <div class="index-item"><div style="font-size:0.7rem;color:#aaa;">道瓊指數</div><div style="font-weight:bold;">{{ dow_curr }}</div><div class="{{ 'positive' if dow_change>=0 else 'negative' }}">{{ ('+' if dow_change>0 else '') ~ dow_change }} ({{ ('+' if dow_pct>0 else '') ~ dow_pct }}%)</div></div>
        <div class="index-item"><div style="font-size:0.7rem;color:#aaa;">那斯達克</div><div style="font-weight:bold;">{{ nas_curr }}</div><div class="{{ 'positive' if nas_change>=0 else 'negative' }}">{{ ('+' if nas_change>0 else '') ~ nas_change }} ({{ ('+' if nas_pct>0 else '') ~ nas_pct }}%)</div></div>
        <div class="index-item"><div style="font-size:0.7rem;color:#aaa;">費城半導體</div><div style="font-weight:bold;">{{ sox_curr }}</div><div class="{{ 'positive' if sox_change>=0 else 'negative' }}">{{ ('+' if sox_change>0 else '') ~ sox_change }} ({{ ('+' if sox_pct>0 else '') ~ sox_pct }}%)</div></div>
    </div>
</div>
<div class="card">
    <form method="POST" action="/add_stock_{{ market }}" style="display:inline;">
        <input type="text" name="ticker" placeholder="股票代號" required>
        <button type="submit" class="btn">➕ 加入自選</button>
    </form>
</div>
<table>
    <thead>
        <tr><th>代號</th><th>價格</th><th>漲跌 (漲跌幅%)</th><th>AI評分</th><th>趨勢</th><th>買賣訊號</th><th>操作</th></tr>
    </thead>
    <tbody>
        {% for s in stocks %}
        <tr>
            <td><a href="/indicators/{{ s.ticker }}" class="stock-link">{{ s.ticker }}</a></td>
            <td>{{ s.price }}</td>
            <td class="{{ 'positive' if s.change>=0 else 'negative' }}">{{ ('+' if s.change>0 else '') ~ s.change }} ({{ ('+' if s.pct>0 else '') ~ s.pct }}%)</a>
            <td style="color: {{ '#00ff00' if s.ai_score>=70 else '#ffcc00' if s.ai_score>=50 else '#ff4444' }}">{{ s.ai_score }}/100</a>
            <td>{{ s.trend }}</a>
            <td>{{ s.signal }}</a>
            <td><a href="/delete/{{ market }}/{{ s.ticker }}" class="btn" style="background:#ff4444;">刪除</a></a>
        </tr>
        {% else %}
        <tr><td colspan="7">尚無自選股，請加入</a></a>
        {% endfor %}
    </tbody>
</div>
</body>
</html>
'''

AI_RANKING_HTML = '''
<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>AI選股排行榜</title>
<style>
body{background:#121214;color:#fff;font-family:sans-serif;padding:20px;}
table{width:100%;border-collapse:collapse;table-layout:fixed;}
th,td{padding:10px;text-align:left;border-bottom:1px solid #333;}
th{background:#2a2a35;}
.btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;display:inline-block;margin:5px;}
.btn:hover{background:#3a3a45;}
.stock-link{color:#00ffaa;text-decoration:none;font-weight:bold;}
.stock-link:hover{color:#00ff66;text-decoration:underline;}
th:nth-child(1){width:8%;}
th:nth-child(2){width:15%;}
th:nth-child(3){width:10%;}
th:nth-child(4){width:15%;}
th:nth-child(5){width:15%;}
th:nth-child(6){width:15%;}
th:nth-child(7){width:22%;}
</style>
</head>
<body><h1>🏆 AI 共振評分排行榜</h1><div><a href="/" class="btn">返回首頁</a></div>
<table><thead><tr><th>排名</th><th>代號</th><th>市場</th><th>價格</th><th>AI評分</th><th>趨勢</th><th>買賣訊號</th></tr></thead>
<tbody>{% for s in stocks %}
        <tr>
            <td>{{ loop.index }}</a>
            <td><a href="/indicators/{{ s.ticker }}" class="stock-link">{{ s.ticker }}</a></a>
            <td>{{ '台股' if s.market=='tw' else '美股' }}</a>
            <td>{{ s.price }}</a>
            <td style="color: {{ '#00ff00' if s.ai_score>=70 else '#ffcc00' if s.ai_score>=50 else '#ff4444' }}">{{ s.ai_score }}/100</a>
            <td>{{ s.trend }}</a>
            <td>{{ s.signal }}</a>
        </tr>
        {% else %}
        <tr><td colspan="7">尚無資料</a></a>
        {% endfor %}</tbody>
</div></body></html>
'''

TENBAGGER_HTML = '''
<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>十倍股雷達</title>
<style>
body{background:#121214;color:#fff;font-family:sans-serif;padding:20px;}
table{width:100%;border-collapse:collapse;table-layout:fixed;}
th,td{padding:8px;text-align:left;border-bottom:1px solid #333;}
th{background:#2a2a35;}
.score-high{color:#00ff00;font-weight:bold;}
.good{color:#00ff00;}
.btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;display:inline-block;margin:5px;}
.btn:hover{background:#3a3a45;}
.stock-link{color:#00ffaa;text-decoration:none;font-weight:bold;}
.stock-link:hover{color:#00ff66;text-decoration:underline;}
th:nth-child(1){width:5%;}
th:nth-child(2){width:10%;}
th:nth-child(3){width:8%;}
th:nth-child(4){width:8%;}
th:nth-child(5){width:20%;}
th:nth-child(6){width:8%;}
th:nth-child(7){width:8%;}
th:nth-child(8){width:6%;}
th:nth-child(9){width:6%;}
th:nth-child(10){width:6%;}
th:nth-child(11){width:15%;}
</style>
</head>
<body><h1>🚀 十倍股候選雷達</h1><div><a href="/" class="btn">返回首頁</a></div><div class="card"><p>⚡ 掃描條件：營收成長>30% / 毛利率成長 / 52週新高 / 爆量 / Reddit熱度上升 / 高空頭比例(軋空)</p></div>
<table><thead><tr><th>排名</th><th>代號</th><th>價格</th><th>綜合分數</th><th>符合條件</th><th>營收成長%</th><th>毛利成長%</th><th>52週高</th><th>爆量</th><th>Reddit熱</th><th>空頭高</th></tr></thead>
<tbody>{% for c in candidates %}
        <tr>
            <td>{{ loop.index }}</a>
            <td><a href="/backtest/{{ c.ticker }}" class="stock-link">{{ c.ticker }}</a></a>
            <td>{{ c.price }}</a>
            <td class="score-high">{{ c.score }}/6</a>
            <td>{{ c.conditions|join(', ') }}</a>
            <td class="{% if c.revenue_growth > 30 %}good{% endif %}">{{ c.revenue_growth }}%</a>
            <td class="{% if c.gm_growth > 5 %}good{% endif %}">{{ c.gm_growth }}%</a>
            <td>{{ '✅' if c.is_52w_high else '❌' }}</a>
            <td>{{ '✅' if c.volume_surge else '❌' }}</a>
            <td>{{ '🔥' if c.reddit_hot else '➖' }}</a>
            <td>{{ '⚠️' if c.short_high else '➖' }}</a>
        </tr>
        {% else %}
        <tr><td colspan="11">目前無符合條件的十倍股候選</a></a>
        {% endfor %}</tbody>
</div></body></html>
'''

BACKTEST_HTML = '''
<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>回測報告</title><style>
body{background:#121214;color:#fff;padding:20px;}.card{background:#1e1e1e;padding:15px;border-radius:12px;}.metric-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:15px;}.metric{background:#2a2a35;padding:10px;border-radius:8px;}.positive{color:#00ff00;}.negative{color:#ff4444;}.btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;}</style></head>
<body><h1>📊 回測報告 - {{ ticker }}</h1>
{% if error %}<div class="card">{{ error }}</div>{% else %}
<div class="card"><h3>績效指標</h3><div class="metric-grid">
<div class="metric"><strong>初始資金</strong><br>${{ result.initial_capital }}</div>
<div class="metric"><strong>最終價值</strong><br>${{ result.final_value }}</div>
<div class="metric"><strong>總報酬率</strong><br><span class="{{ 'positive' if result.total_return>=0 else 'negative' }}">{{ result.total_return }}%</span></div>
<div class="metric"><strong>年化報酬(CAGR)</strong><br><span class="{{ 'positive' if result.cagr>=0 else 'negative' }}">{{ result.cagr }}%</span><div class="interpret">📌 年化複合報酬率，越高越好</div></div>
<div class="metric"><strong>最大回撤</strong><br><span class="negative">{{ result.max_drawdown }}%</span><div class="interpret">📌 歷史最大虧損幅度，越小越好</div></div>
<div class="metric"><strong>Sharpe比率</strong><br>{{ result.sharpe_ratio }}<div class="interpret">📌 >1 較佳，>2 優秀</div></div>
<div class="metric"><strong>Sortino比率</strong><br>{{ result.sortino_ratio }}<div class="interpret">📌 專注下跌風險，越高越好</div></div>
<div class="metric"><strong>Alpha</strong><br>{{ result.alpha }}%<div class="interpret">📌 正值代表打敗大盤</div></div>
<div class="metric"><strong>Beta</strong><br>{{ result.beta }}<div class="interpret">📌 >1 波動大於大盤</div></div>
</div></div>{% endif %}
<div><a href="/" class="btn">返回首頁</a> <a href="/indicators/{{ ticker }}" class="btn">技術指標</a></div></body></html>
'''

POSITIONS_HTML = '''
<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>持倉中心</title><style>
body{background:#121214;color:#fff;padding:20px;}table{width:100%;}th,td{padding:8px;border-bottom:1px solid #333;}th{background:#2a2a35;}.positive{color:#00ff00;}.negative{color:#ff4444;}.btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;}.stock-link{color:#00ffaa;font-weight:bold;}</style></head>
<body><h1>📊 持倉管理</h1><div><form method="POST" action="/add_position"><input type="text" name="pos_ticker" placeholder="代號"><input type="number" step="0.01" name="cost" placeholder="成本"><input type="number" name="shares" placeholder="股數"><button type="submit" class="btn">➕ 新增/更新</button></form><a href="/" class="btn">返回首頁</a></div>
<table><thead><th>代號</th><th>股數</th><th>成本價</th><th>現價</th><th>市值</th><th>成本總額</th><th>損益</th><th>報酬率</th><th>停損價</th><th>操作</th></thead>
<tbody>{% for p in positions %}
        <tr>
            <td><a href="/indicators/{{ p.ticker }}" class="stock-link">{{ p.ticker }}</a></a>
            <td>{{ p.shares }}</a>
            <td>{{ p.cost }}</a>
            <td>{{ p.current_price }}</a>
            <td>{{ p.value }}</a>
            <td>{{ p.cost_value }}</a>
            <td class="{{ 'positive' if p.profit>=0 else 'negative' }}">{{ p.profit }}</a>
            <td class="{{ 'positive' if p.profit_pct>=0 else 'negative' }}">{{ p.profit_pct }}%</a>
            <td>{{ p.stop_loss }}</a>
            <td><a href="/clear_position/{{ p.ticker }}" class="btn" style="background:#ff4444;">歸零</a></a>
        </tr>
        {% else %}
        <tr><td colspan="10">無持股</a></a>
        {% endfor %}</tbody>
</div></body></html>
'''

INDICATORS_HTML = '''
<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>{{ ticker }} 技術指標</title><style>
body{background:#121214;color:#fff;padding:20px;}.card{background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:20px;}.metric-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:15px;}.metric{background:#2a2a35;padding:10px;border-radius:8px;}.btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;}</style>
</head>
<body><h1>📈 {{ ticker }} 技術指標</h1><div><a href="/" class="btn">返回首頁</a> <a href="/backtest/{{ ticker }}" class="btn">回測</a></div>
{% if error %}<div class="card">{{ error }}</div>{% else %}
<div class="card"><h3>即時報價</h3><div class="metric-grid"><div class="metric">價格: {{ "%.2f"|format(price) }}</div><div class="metric">漲跌: <span class="{{ 'positive' if change>=0 else 'negative' }}">{{ ('+' if change>0 else '') ~ change }}</span></div><div class="metric">漲跌幅: <span class="{{ 'positive' if pct>=0 else 'negative' }}">{{ ('+' if pct>0 else '') ~ pct }}%</span></div></div></div>
<div class="card"><h3>核心指標</h3><div class="metric-grid"><div class="metric">⭐ AI 共振評分: <strong style="color: {{ '#00ff00' if ai_score>=70 else '#ffcc00' if ai_score>=50 else '#ff4444' }}">{{ ai_score }}/100</strong></div><div class="metric">RSI: {{ rsi }} <span style="font-size:0.8rem;">({{ rsi_status }})</span></div><div class="metric">MACD: {{ macd_status }} (柱體: {{ macd_hist }})</div><div class="metric">VWAP: {{ vwap_status }}</div><div class="metric">量比: {{ volume_ratio }}</div></div></div>
<div class="card"><h3>🤖 機器學習預測 (RandomForest)</h3><div class="metric-grid">{% if ml_pred %}<div class="metric"><strong>預期漲跌</strong><br><span style="color: {{ '#00ff00' if ml_pred[0]=='+' else '#ff4444' }}">{{ ml_pred }}</span><div class="interpret">{{ ml_trend }} {{ ml_confidence }}</div><div class="interpret" style="font-size:0.7rem;">{{ ml_details }}</div></div>{% else %}<div class="metric">模型訓練中或未啟用</div>{% endif %}</div></div>
<div class="card"><h3>🤖 多因子綜合評分 (RF+XGB+LGB+情緒+籌碼+熱度+趨勢)</h3><div class="metric-grid">{% if ensemble_score is not none %}<div class="metric"><strong>最終信心分數</strong><br><span style="color: {{ '#00ff00' if ensemble_score>30 else '#ffcc00' if ensemble_score>-20 else '#ff4444' }}">{{ ensemble_score }} / 100</span><div class="interpret"><strong>買賣訊號：{{ ensemble_signal }}</strong></div><div class="interpret">模型一致率: {{ "%.0f"|format(agreement*100) }}%</div><div class="interpret" style="font-size:0.7rem;">{% for model, detail in ensemble_details.items() %}{{ model }}: {{ detail }} &nbsp;{% endfor %}</div></div>{% else %}<div class="metric">Ensemble 模型尚未訓練，請稍後或點擊「重新訓練所有模型」</div>{% endif %}</div></div>
<div class="card"><h3>📊 額外因子</h3><div class="metric-grid">
    <div class="metric">💰 籌碼分數: {{ smart_score|default(50) }}</div>
    <div class="metric">🔥 熱度分數: {{ hype_score|default(50) }}</div>
    <div class="metric">📈 趨勢確認: {{ trend_score|default(50) }}</div>
    <div class="metric">📈 成長分數: {{ growth_score|default(50) }}</div>
    <div class="metric">🚀 十倍股潛力: {{ ten_bagger_score|default(50) }}</div>
    <div class="metric">🤝 共識分數: {{ consensus_score|default(0) }}</div>
</div></div>
<div class="card"><h3>均線</h3><div class="metric-grid"><div class="metric">20MA: {{ ma20 }}</div><div class="metric">60MA: {{ ma60 }}</div><div class="metric">200MA: {{ ma200 }}</div></div></div>
<div class="card"><h3>壓力支撐</h3><div class="metric-grid"><div class="metric">🔴 壓力價: {{ resistance }}</div><div class="metric">🟢 目標價: {{ target }}</div><div class="metric">⚪ 停損價: {{ stop_loss }}</div></div></div>
<div class="card"><h3>📖 AI 評分解釋</h3><ul>{% for r in reasons %}<li>{{ r }}</li>{% endfor %}</ul></div>{% endif %}</body></html>
'''

BONDING_HTML = '''
<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>均線糾結選股</title><style>
body{background:#121214;color:#fff;font-family:sans-serif;padding:20px;}
table{width:100%;border-collapse:collapse;}
th,td{border:1px solid #444;padding:8px;text-align:left;}
th{background:#2a2a35;color:#fff;}
tr:nth-child(even){background:#1e1e1e;}
tr:hover{background:#2a2a35;}
.card{background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:20px;}
.note{color:#aaa;font-size:12px;margin-top:10px;}
.volume-confirm{display:inline-block;margin-left:8px;color:#00ff00;font-size:0.75rem;font-weight:bold;white-space:nowrap;}
.btn{display:inline-block;background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;margin:5px;text-decoration:none;}
.btn:hover{background:#3a3a45;}
.stock-link{color:#00ffaa;text-decoration:none;font-weight:bold;}
.stock-link:hover{color:#00ff66;text-decoration:underline;}
</style>
</head>
<body>
<h1>均線糾結選股 (MA5/MA20/MA60)</h1>
<div class="note">📊 掃描台股 {{ tw_total }} 檔 + 美股 {{ us_total }} 檔 | 糾結閾值: {{ threshold }}% | ✅ 已加入成交量確認</div>
<div class="card"><h2>台股 ({{ tw_count }} 檔均線糾結)</h2>
{% if tw_results %}
    <table><thead><th>代號</th><th>價格</th><th>糾結度%</th><th>MA5</th><th>MA20</th><th>MA60</th><th>方向</th></thead><tbody>
    {% for s in tw_results %}
        <tr>
            <td><a href="/indicators/{{ s.ticker }}" class="stock-link">{{ s.ticker }}</a></td>
            <td>{{ s.price }}</td>
            <td>{{ s.spread }}</td>
            <td>{{ s.ma5 }}</td>
            <td>{{ s.ma20 }}</td>
            <td>{{ s.ma60 }}</td>
            <td>{{ s.direction }}{% if s.volume_confirm %}<span class="volume-confirm">📊 爆量確認</span>{% endif %}</td>
        </tr>
    {% endfor %}</tbody>
    </table>
{% else %}<p>目前無均線糾結股票</p>{% endif %}</div>
<div class="card"><h2>美股 ({{ us_count }} 檔均線糾結)</h2>
{% if us_results %}
    </table><thead><th>代號</th><th>價格</th><th>糾結度%</th><th>MA5</th><th>MA20</th><th>MA60</th><th>方向</th></thead><tbody>
    {% for s in us_results %}
        <tr>
            <td><a href="/indicators/{{ s.ticker }}" class="stock-link">{{ s.ticker }}</a></td>
            <td>{{ s.price }}</td>
            <td>{{ s.spread }}</td>
            <td>{{ s.ma5 }}</td>
            <td>{{ s.ma20 }}</td>
            <td>{{ s.ma60 }}</td>
            <td>{{ s.direction }}{% if s.volume_confirm %}<span class="volume-confirm">📊 爆量確認</span>{% endif %}</td>
        </tr>
    {% endfor %}</tbody>
    </table>
{% else %}<p>目前無均線糾結股票</p>{% endif %}</div>
<div><a href="/" class="btn">返回首頁</a></div>
</body></html>
'''

SETTINGS_HTML = '''
<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>系統設定</title><style>
body{background:#121214;color:#fff;padding:20px;}.card{background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:20px;}input,button{padding:8px;margin:5px;border-radius:20px;border:none;}button{background:#4caf50;color:white;}.btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;display:inline-block;}label{display:inline-block;width:300px;margin:8px 0;}</style></head>
<body><h1>⚙️ 策略設定</h1><div><a href="/" class="btn">返回首頁</a></div>
<form method="POST"><div class="card"><h3>風險控制</h3><label>單筆最大虧損(%): <input type="number" step="0.5" name="max_loss_per_trade" value="{{ settings.max_loss_per_trade }}"></label><br>
<label>每日最大虧損(%): <input type="number" step="0.5" name="max_loss_per_day" value="{{ settings.max_loss_per_day }}"></label><br>
<label>單一持股最大佔比(%): <input type="number" step="5" name="max_concentration" value="{{ settings.max_concentration }}"></label><br>
<label>現金餘額: <input type="number" step="1000" name="cash_balance" value="{{ settings.cash_balance }}"></label><br>
</div><div class="card"><h3>觀察名單 (逗號分隔)</h3><input type="text" name="watchlist" value="{{ watchlist }}" style="width:80%;"></div>
<div class="card"><h3>系統效能</h3><label>背景更新間隔(分鐘): <input type="number" name="background_update_minutes" value="{{ settings.background_update_minutes }}"></label><br>
<label>資料快取時間(分鐘): <input type="number" name="cache_ttl_minutes" value="{{ settings.cache_ttl_minutes }}"></label><br>
<label>自動清除快取間隔(小時): <input type="number" name="auto_clear_cache_hours" value="{{ settings.auto_clear_cache_hours }}"></label><br>
<label>啟用多模型投票: <input type="checkbox" name="enable_ensemble" {% if settings.enable_ensemble %}checked{% endif %}></label><br>
<label>預測天數: <input type="number" name="prediction_days" value="{{ settings.prediction_days }}" min="1" max="10"></label><br>
<label>訓練資料天數: <input type="number" name="train_days" value="{{ settings.train_days }}" min="60" max="500"></label><br>
<label>自動重新訓練間隔(小時): <input type="number" name="retrain_hours" value="{{ settings.retrain_hours }}" min="1" max="168"></label><br>
</div><button type="submit">儲存設定</button></form></body></html>
'''

if __name__ == "__main__":
    print("🚀 啟動終極版 AI 量化監控中心 (多因子 Ensemble + 統一機率 + 成長細分 + 共識分數)")
    print("✅ 200MA 正確計算 (至少200天資料)")
    print("✅ RF 輸出轉為上漲機率，與 XGB/LGB 一致")
    print("✅ 成長分數拆解為營收、EPS、毛利率、產業")
    print("✅ 新增共識分數 (模型一致率 + 趨勢一致 + 成長/熱度一致)")
    print("✅ 安裝 xgboost lightgbm textblob 可獲得完整功能")
    app.run(host="127.0.0.1", port=5005, debug=False, threaded=True)