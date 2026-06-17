# -*- coding: utf-8 -*-
"""
完整功能 v16_final (徹底修正表格標籤、多餘字元、CSS 固定寬度)
已整合：yfinance重試、環境變數、模板分離、SSE即時推送、統一背景排程
"""
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os, sys, json, math, time, logging, random, re, threading
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os, sys, json, math, time, logging, random, re, threading
from datetime import datetime, timedelta
from collections import defaultdict
from flask import Flask, render_template_string, render_template, request, redirect, url_for, jsonify, Response, stream_with_context
from dotenv import load_dotenv
load_dotenv()

import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.ensemble import RandomForestRegressor
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib, numpy as np, pandas as pd
from scipy.stats import norm
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import delta
import sqlite3
from contextlib import closing

try:
    import xgboost as xgb; XGB_AVAILABLE = True
except ImportError: XGB_AVAILABLE = False
try:
    import lightgbm as lgb; LGB_AVAILABLE = True
except ImportError: LGB_AVAILABLE = False
try:
    import shioaji as sj; SHIOAJI_AVAILABLE = True
except ImportError: SHIOAJI_AVAILABLE = False
try:
    import plotly.graph_objects as go; import plotly.io as pio
    PLOTLY_AVAILABLE = True; pio.renderers.default = 'iframe'
except ImportError: PLOTLY_AVAILABLE = False
try:
    from scipy.stats import norm; SCIPY_AVAILABLE = True
except ImportError: SCIPY_AVAILABLE = False
try:
    import pytz; PYTZ_AVAILABLE = True
except ImportError: PYTZ_AVAILABLE = False

if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
@app.template_filter('format_number')
def format_number(value):
    """將數字格式化為帶有逗號的字串"""
    if value is None:
        return 'N/A'
    try:
        return f"{value:,}"
    except (ValueError, TypeError):
        return value
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
TENBAGGER_FILE = os.path.join(DATA_DIR, "tenbagger.json")
GEX_SCREENER_FILE = os.path.join(DATA_DIR, "gex_screener.json")
EARNINGS_CACHE_FILE = os.path.join(DATA_DIR, "earnings_cache.json")
INSTITUTIONAL_DB = os.path.join(DATA_DIR, "institutional.db")
DATABASE_URL = os.getenv('DATABASE_URL')

def get_db_connection():
    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL)
    else:
        return sqlite3.connect(INSTITUTIONAL_DB)
DEFAULT_SETTINGS = {
    "refresh_seconds": 60, "bonding_threshold": 3.0, "enable_ensemble": True,
    "prediction_days": 3, "train_days": 180, "retrain_hours": 24,
    "auto_clear_cache_hours": 24, "background_update_minutes": 15,
    "cache_ttl_minutes": 1, "max_loss_per_trade": 5.0, "max_loss_per_day": 10.0,
    "max_concentration": 40.0, "cash_balance": 0.0,
}
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
            for k, v in DEFAULT_SETTINGS.items():
                if k not in loaded: loaded[k] = v
            return loaded
    return DEFAULT_SETTINGS.copy()
def save_settings(settings):
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)
settings = load_settings()

for f in [STOCK_FILE_TW, STOCK_FILE_US]:
    if not os.path.exists(f): open(f, "w", encoding="utf-8").close()
if not os.path.exists(POSITIONS_FILE): json.dump([], open(POSITIONS_FILE, "w", encoding='utf-8'))
if not os.path.exists(WATCHLIST_FILE): json.dump(["NVDA","TSLA","AAPL","MSFT","GOOGL"], open(WATCHLIST_FILE, "w", encoding='utf-8'))

# ================== 快取系統 ==================
_data_cache = {}
_model_cache_rf = {}
_model_cache_xgb = {}
_model_cache_lgb = {}
_gex_cache = {}
_cache_lock = threading.Lock()
_model_lock_rf = threading.Lock()
_model_lock_xgb = threading.Lock()
_model_lock_lgb = threading.Lock()
_gex_lock = threading.Lock()
_last_alerts_hash = None
_alerts_lock = threading.Lock()

def get_cached_data(ticker, period="2y", ttl_seconds=None):
    if ttl_seconds is None:
        ttl_seconds = settings.get('cache_ttl_minutes',5)*60
    key = f"{ticker}_{period}"
    now = time.time()
    with _cache_lock:
        if key in _data_cache and now - _data_cache[key]['time'] < ttl_seconds:
            df = _data_cache[key]['data']
        else:
            @retry(stop=stop_after_attempt(3),
                   wait=wait_exponential(multiplier=1, min=2, max=10),
                   retry=retry_if_exception_type(Exception))
            def _fetch_yf_history(ticker, period):
                return yf.Ticker(ticker).history(period=period, auto_adjust=True)

            try:
                df = _fetch_yf_history(ticker, period)
                if df.empty:
                    df = pd.DataFrame()
                else:
                    df = df.ffill().dropna()
                _data_cache[key] = {'data': df, 'time': now}
            except Exception as e:
                logger.warning(f"取得 {ticker} 資料失敗 (重試後仍失敗): {e}")
                df = pd.DataFrame()
        return df

def get_cached_gex(ticker, ttl_seconds=3600):
    now = time.time()
    with _gex_lock:
        if ticker in _gex_cache and now - _gex_cache[ticker]['time'] < ttl_seconds:
            return _gex_cache[ticker]['data']
    return None
def set_cached_gex(ticker, data):
    with _gex_lock: _gex_cache[ticker] = {'data': data, 'time': time.time()}
def clear_cache():
    with _cache_lock: _data_cache.clear()
    with _gex_lock: _gex_cache.clear()
    logger.info("所有快取已清除")

# ================== 輔助函數：美股交易時段判斷 ==================
def is_us_market_open():
    if not PYTZ_AVAILABLE: return True
    now = datetime.now()
    if now.weekday() >= 5: return False
    est = pytz.timezone('US/Eastern')
    now_est = now.astimezone(est)
    open_time = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now_est.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_time <= now_est <= close_time

# ================== Shioaji ==================
_sj = None; _sj_login_time = 0
def get_shioaji_status():
    if not SHIOAJI_AVAILABLE: return "❌ 未安裝"
    return "🟢 即時模式" if _sj is not None else "🔴 延遲模式"
def get_shioaji():
    global _sj, _sj_login_time
    now = time.time()
    if _sj is not None and now - _sj_login_time < 23*3600: return _sj
    if not SHIOAJI_AVAILABLE: return None
    api_key = os.getenv("SHIOAJI_API_KEY")
    secret_key = os.getenv("SHIOAJI_SECRET_KEY")
    if not api_key or not secret_key: return None
    try:
        if _sj is not None: _sj.logout()
        _sj = sj.Shioaji(simulation=False)
        _sj.login(api_key=api_key, secret_key=secret_key)
        logger.info("Shioaji 登入成功")
        _sj_login_time = now
        return _sj
    except Exception as e:
        logger.error(f"Shioaji 登入失敗: {e}")
        return None
def get_tw_stock_realtime_shioaji(ticker, max_retries=3):
    """從 Shioaji 取得即時報價，若失敗則重試"""
    sj_api = get_shioaji()
    if sj_api is None:
        return None, None, None

    stock_id = ticker.replace('.TW', '').upper()
    for attempt in range(max_retries):
        try:
            contract = sj_api.Contracts.Stocks[stock_id]
            snapshot = sj_api.snapshot([contract])
            if not snapshot:
                time.sleep(0.1 * (attempt + 1))  # 每次重試間隔遞增
                continue

            tick = snapshot[0]
            curr = tick.close if tick.close else tick.last_price
            if curr is None:
                time.sleep(0.1 * (attempt + 1))
                continue

            prev = tick.reference
            if prev is None or prev == 0:
                time.sleep(0.1 * (attempt + 1))
                continue

            change = curr - prev
            pct = (change / prev) * 100
            return curr, change, pct

        except Exception as e:
            logger.debug(f"Shioaji 抓取 {ticker} 失敗 (嘗試 {attempt+1}/{max_retries}): {e}")
            time.sleep(0.1 * (attempt + 1))
            continue

    logger.error(f"Shioaji 抓取 {ticker} 最終失敗，已重試 {max_retries} 次")
    return None, None, None
# ================== 技術指標函數 ==================
def calculate_rsi(close, period=14):
    if len(close) < period: return 50
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
def calculate_vwap(df):
    if df.empty or len(df) < 2: return None
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap.iloc[-1]
def get_vwap_status(current_price, vwap):
    if vwap is None: return "N/A"
    return f"🟢 站上 VWAP ({vwap:.2f})" if current_price > vwap else f"🔴 跌破 VWAP ({vwap:.2f})"
def calculate_macd(close, fast=12, slow=26, signal=9):
    if len(close) < slow+signal: return None, None, None, "-"
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    macd_curr, signal_curr = macd_line.iloc[-1], signal_line.iloc[-1]
    macd_prev = macd_line.iloc[-2] if len(macd_line)>1 else macd_curr
    signal_prev = signal_line.iloc[-2] if len(signal_line)>1 else signal_curr
    if macd_curr > signal_curr and macd_prev <= signal_prev: status = "🔝 金叉"
    elif macd_curr < signal_curr and macd_prev >= signal_prev: status = "🔻 死叉"
    else: status = "─ 持中"
    return macd_curr, signal_curr, histogram.iloc[-1], status
def calculate_support_resistance(df, current_price):
    if df.empty or len(df) < 20: return {"resistance":0,"target":0,"stop_loss":0}
    close = df['Close'].values
    high = df['High'].values; low = df['Low'].values
    high_20 = np.max(high[-20:])
    tr = high - low
    atr = np.mean(tr[-14:]) if len(tr)>=14 else np.mean(tr)
    ma20 = np.mean(close[-20:])
    resistance = max(high_20, ma20 + 2*np.std(close[-20:]))
    target = current_price + atr*2.5
    stop_loss = current_price - atr*1.5
    return {"resistance": round(resistance,2), "target": round(target,2), "stop_loss": round(stop_loss,2), "atr": round(atr,2)}
def calculate_ai_resonance_score(rsi, macd_status, vwap_status, volume_ratio):
    score = 0
    if rsi < 30: score+=25
    elif rsi < 50: score+=15
    elif rsi < 70: score+=10
    else: score+=5
    if "金叉" in macd_status: score+=25
    elif "持中" in macd_status: score+=10
    if "站上" in vwap_status: score+=25
    else: score+=5
    try:
        vol = float(volume_ratio) if volume_ratio else 1
        if vol > 1.5: score+=25
        elif vol > 1.2: score+=15
        else: score+=5
    except: score+=5
    return min(100, max(0, score))

# ================== 模型相關 ==================
FEATURE_NAMES = [
    'return_1d','return_5d','return_10d','return_20d',
    'price_vs_ma5','price_vs_ma10','price_vs_ma20','price_vs_ma60',
    'ma5_vs_ma20','ma20_vs_ma60',
    'volume_ratio_5','volume_ratio_10',
    'rsi','bb_position','bb_width','kd_k','kd_d',
    'volatility_5d','volatility_20d',
    'high_low_ratio','high_20d_ratio','low_20d_ratio','trend_strength'
]
def calculate_features(df):
    if df.empty or len(df) < 60: return None
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values if 'Volume' in df.columns else np.ones(len(close))
    features = {}
    features['return_1d'] = (close[-1]-close[-2])/close[-2] if len(close)>=2 else 0
    features['return_5d'] = (close[-1]-close[-6])/close[-6] if len(close)>=6 else 0
    features['return_10d'] = (close[-1]-close[-11])/close[-11] if len(close)>=11 else 0
    features['return_20d'] = (close[-1]-close[-21])/close[-21] if len(close)>=21 else 0
    ma5 = np.mean(close[-5:]) if len(close)>=5 else close[-1]
    ma10 = np.mean(close[-10:]) if len(close)>=10 else close[-1]
    ma20 = np.mean(close[-20:]) if len(close)>=20 else close[-1]
    ma60 = np.mean(close[-60:]) if len(close)>=60 else close[-1]
    features['price_vs_ma5'] = (close[-1]-ma5)/ma5 if ma5>0 else 0
    features['price_vs_ma10'] = (close[-1]-ma10)/ma10 if ma10>0 else 0
    features['price_vs_ma20'] = (close[-1]-ma20)/ma20 if ma20>0 else 0
    features['price_vs_ma60'] = (close[-1]-ma60)/ma60 if ma60>0 else 0
    features['ma5_vs_ma20'] = (ma5-ma20)/ma20 if ma20>0 else 0
    features['ma20_vs_ma60'] = (ma20-ma60)/ma60 if ma60>0 else 0
    vol_ma5 = np.mean(volume[-5:]) if len(volume)>=5 else volume[-1]
    vol_ma10 = np.mean(volume[-10:]) if len(volume)>=10 else volume[-1]
    features['volume_ratio_5'] = volume[-1]/vol_ma5 if vol_ma5>0 else 1
    features['volume_ratio_10'] = volume[-1]/vol_ma10 if vol_ma10>0 else 1
    if len(close)>=15:
        deltas = np.diff(close[-15:])
        gain = np.mean(deltas[deltas>0]) if any(deltas>0) else 0
        loss = -np.mean(deltas[deltas<0]) if any(deltas<0) else 0
        rs = gain/loss if loss>0 else 1
        features['rsi'] = 100 - (100/(1+rs))
    else: features['rsi'] = 50
    std20 = np.std(close[-20:]) if len(close)>=20 else 0
    bb_upper = ma20 + 2*std20
    bb_lower = ma20 - 2*std20
    features['bb_position'] = (close[-1]-bb_lower)/(bb_upper-bb_lower) if bb_upper!=bb_lower else 0.5
    features['bb_width'] = (bb_upper-bb_lower)/ma20 if ma20>0 else 0
    if len(high)>=9 and len(low)>=9 and len(close)>=9:
        low_n = np.min(low[-9:]); high_n = np.max(high[-9:])
        rsv = (close[-1]-low_n)/(high_n-low_n)*100 if high_n!=low_n else 50
        features['kd_k'] = rsv
        features['kd_d'] = np.mean([rsv,50])
    else: features['kd_k'] = 50; features['kd_d'] = 50
    features['volatility_5d'] = np.std(close[-5:])/ma5 if ma5>0 else 0
    features['volatility_20d'] = np.std(close[-20:])/ma20 if ma20>0 else 0
    features['high_low_ratio'] = (high[-1]-low[-1])/close[-1] if close[-1]>0 else 0
    features['high_20d_ratio'] = (high[-1]-np.max(high[-20:]))/close[-1] if len(high)>=20 and close[-1]>0 else 0
    features['low_20d_ratio'] = (low[-1]-np.min(low[-20:]))/close[-1] if len(low)>=20 and close[-1]>0 else 0
    features['trend_strength'] = abs(close[-1]-close[-20])/ma20 if ma20>0 else 0
    return features
def get_feature_vector(features):
    return [features.get(name, 0) for name in FEATURE_NAMES]
# ------------------ RandomForest ------------------
def train_single_model_rf(ticker, force_retrain=False):
    model_name = ticker.replace('.TW','')
    model_path = os.path.join(MODELS_DIR, f"{model_name}_rf.joblib")
    scaler_path = os.path.join(SCALERS_DIR, f"{model_name}_scaler_rf.joblib")
    if not force_retrain and os.path.exists(model_path) and os.path.exists(scaler_path):
        mtime = os.path.getmtime(model_path)
        if time.time() - mtime < settings.get('retrain_hours',24)*3600:
            try: return joblib.load(model_path), joblib.load(scaler_path)
            except: pass
    X, y = prepare_training_data(ticker, settings.get('train_days',180))
    if X is None or len(X)<100: return None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"{ticker} RandomForest 模型訓練完成")
    return model, scaler
def get_rf_model(ticker, force_retrain=False):
    cache_key = ticker.replace('.TW','')
    now = time.time()
    with _model_lock_rf:
        if not force_retrain and cache_key in _model_cache_rf:
            model, scaler, ts = _model_cache_rf[cache_key]
            if now - ts < 3600: return model, scaler
    model, scaler = train_single_model_rf(ticker, force_retrain)
    if model:
        with _model_lock_rf: _model_cache_rf[cache_key] = (model, scaler, now)
    return model, scaler

# ------------------ XGBoost ------------------
def train_xgboost_model(ticker, force_retrain=False):
    if not XGB_AVAILABLE: return None, None
    model_name = ticker.replace('.TW','')
    model_path = os.path.join(MODELS_DIR, f"{model_name}_xgb.joblib")
    scaler_path = os.path.join(SCALERS_DIR, f"{model_name}_scaler_xgb.joblib")
    if not force_retrain and os.path.exists(model_path) and os.path.exists(scaler_path):
        mtime = os.path.getmtime(model_path)
        if time.time() - mtime < settings.get('retrain_hours',24)*3600:
            try: return joblib.load(model_path), joblib.load(scaler_path)
            except: pass
    X, y = prepare_training_data(ticker, settings.get('train_days',180))
    if X is None or len(X)<100: return None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    joblib.dump(model, model_path); joblib.dump(scaler, scaler_path)
    logger.info(f"{ticker} XGBoost 模型訓練完成")
    return model, scaler
def get_xgb_model(ticker, force_retrain=False):
    cache_key = ticker.replace('.TW','')
    now = time.time()
    with _model_lock_xgb:
        if not force_retrain and cache_key in _model_cache_xgb:
            model, scaler, ts = _model_cache_xgb[cache_key]
            if now - ts < 3600: return model, scaler
    model, scaler = train_xgboost_model(ticker, force_retrain)
    if model: _model_cache_xgb[cache_key] = (model, scaler, now)
    return model, scaler

# ------------------ LightGBM ------------------
def train_lightgbm_model(ticker, force_retrain=False):
    if not LGB_AVAILABLE: return None, None
    model_name = ticker.replace('.TW','')
    model_path = os.path.join(MODELS_DIR, f"{model_name}_lgb.joblib")
    scaler_path = os.path.join(SCALERS_DIR, f"{model_name}_scaler_lgb.joblib")
    if not force_retrain and os.path.exists(model_path) and os.path.exists(scaler_path):
        mtime = os.path.getmtime(model_path)
        if time.time() - mtime < settings.get('retrain_hours',24)*3600:
            try: return joblib.load(model_path), joblib.load(scaler_path)
            except: pass
    X, y = prepare_training_data(ticker, settings.get('train_days',180))
    if X is None or len(X)<100: return None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    joblib.dump(model, model_path); joblib.dump(scaler, scaler_path)
    logger.info(f"{ticker} LightGBM 模型訓練完成")
    return model, scaler
def get_lgb_model(ticker, force_retrain=False):
    cache_key = ticker.replace('.TW','')
    now = time.time()
    with _model_lock_lgb:
        if not force_retrain and cache_key in _model_cache_lgb:
            model, scaler, ts = _model_cache_lgb[cache_key]
            if now - ts < 3600: return model, scaler
    model, scaler = train_lightgbm_model(ticker, force_retrain)
    if model: _model_cache_lgb[cache_key] = (model, scaler, now)
    return model, scaler

# ------------------ 共用 ------------------
def prepare_training_data(ticker, days=180):
    try:
        df = get_cached_data(ticker, period=f"{days+60}d")
        if df.empty or len(df)<60: return None, None
        X, y = [], []
        for i in range(60, len(df)-settings.get('prediction_days',3)):
            segment = df.iloc[i-60:i]
            features = calculate_features(segment)
            if features is None: continue
            future_close = df['Close'].iloc[i+settings.get('prediction_days',3)]
            current_close = df['Close'].iloc[i]
            label = (future_close - current_close)/current_close
            X.append(get_feature_vector(features))
            y.append(label)
        if len(X)<100: return None, None
        return np.array(X), np.array(y)
    except Exception as e:
        logger.error(f"準備 {ticker} 訓練資料失敗: {e}")
        return None, None

def get_smart_money_score(ticker):
    try:
        info = yf.Ticker(ticker).info
        inst = info.get('institutionPercent', 0.5)*100
        short = info.get('shortPercentOfFloat', 0)*100
        short_score = max(0, 100 - short*100/0.3) if short else 50
        return round((inst*0.5 + short_score*0.5), 1)
    except: return 50.0
def get_trend_score(df):
    if df.empty or len(df)<200: return 50.0
    close = df['Close']
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    price = close.iloc[-1]
    rsi = calculate_rsi(close).iloc[-1] if not calculate_rsi(close).isna().all() else 50
    _,_,_,macd = calculate_macd(close)
    vwap = calculate_vwap(df)
    score = 0
    if price>ma20: score+=15
    if ma20>ma50: score+=15
    if ma50>ma200: score+=20
    if rsi>60: score+=15
    if "金叉" in macd: score+=15
    if vwap and price>vwap: score+=20
    return min(100, score)
def get_growth_score(ticker, df):
    score = 0
    try:
        tk = yf.Ticker(ticker)
        qf = tk.quarterly_financials
        if not qf.empty and 'Total Revenue' in qf.index:
            rev = qf.loc['Total Revenue'].dropna()
            if len(rev)>=2:
                growth = (rev.iloc[0]-rev.iloc[1])/rev.iloc[1]*100
                if growth>30: score+=30
                elif growth>15: score+=20
                elif growth>0: score+=10
        inc = tk.quarterly_income_stmt
        if not inc.empty and 'Net Income' in inc.index:
            eps = inc.loc['Net Income'].dropna()
            if len(eps)>=2 and eps.iloc[1]!=0:
                growth = (eps.iloc[0]-eps.iloc[1])/abs(eps.iloc[1])*100
                if growth>30: score+=30
                elif growth>15: score+=20
                elif growth>0: score+=10
        if not inc.empty and 'Gross Profit' in inc.index and 'Total Revenue' in inc.index:
            gp = inc.loc['Gross Profit'].dropna()
            rev = inc.loc['Total Revenue'].dropna()
            if len(gp)>=2 and len(rev)>=2:
                gm1 = gp.iloc[0]/rev.iloc[0]*100
                gm2 = gp.iloc[1]/rev.iloc[1]*100
                if gm1 - gm2 >5: score+=20
                elif gm1 - gm2 >0: score+=10
        sector = tk.info.get('sector','').upper()
        if any(k in sector for k in ["AI","SEMICONDUCTOR","CLOUD","NUCLEAR","ROBOTICS"]): score+=20
    except: pass
    return min(100, score), []
def get_ten_bagger_score(ticker):
    ticker = ticker.upper()
    score = 0
    try:
        info = yf.Ticker(ticker).info
        text = (ticker + " " + info.get('sector','') + " " + info.get('industry','') + " " + info.get('longBusinessSummary','')).upper()
        keywords = ["AI","ROBOT","NUCLEAR","QUANTUM","CLOUD","BIOTECH"]
        for kw in keywords:
            if kw in text: score+=20
        if any(x in ticker for x in ["NVDA","AMD","AVGO","MRVL","VRT","SMCI","PLTR","OKLO","IONQ"]): score+=30
    except: pass
    return min(100, score)
def ensemble_predict(ticker, df):
    import math
    feats = calculate_features(df)
    if feats is None:
        return None, None, {}, None, None, None, None, None, None, None, None
    X = np.array(get_feature_vector(feats)).reshape(1, -1)
    scores, details = {}, {}

    # RandomForest
    rf_model, rf_scaler = get_rf_model(ticker)
    if rf_model and rf_scaler:
        X_scaled = rf_scaler.transform(X)
        rf_pred = rf_model.predict(X_scaled)[0]
        rf_prob = 100 / (1 + math.exp(-rf_pred * 10))
        scores['RF'] = rf_prob
        details['RF'] = f"預期報酬 {rf_pred * 100:+.1f}% → 上漲機率 {rf_prob:.0f}%"

    # XGBoost
    xgb_model, xgb_scaler = get_xgb_model(ticker)
    if xgb_model and xgb_scaler:
        X_scaled = xgb_scaler.transform(X)
        xgb_pred = xgb_model.predict(X_scaled)[0]
        xgb_prob = 100 / (1 + math.exp(-xgb_pred * 10))
        scores['XGB'] = xgb_prob
        details['XGB'] = f"預期報酬 {xgb_pred * 100:+.1f}% → 上漲機率 {xgb_prob:.0f}%"

    # LightGBM
    lgb_model, lgb_scaler = get_lgb_model(ticker)
    if lgb_model and lgb_scaler:
        X_scaled = lgb_scaler.transform(X)
        lgb_pred = lgb_model.predict(X_scaled)[0]
        lgb_prob = 100 / (1 + math.exp(-lgb_pred * 10))
        scores['LGB'] = lgb_prob
        details['LGB'] = f"預期報酬 {lgb_pred * 100:+.1f}% → 上漲機率 {lgb_prob:.0f}%"

    # 籌碼分數（美股）
    smart = get_smart_money_score(ticker)
    scores['SMART'] = smart
    details['SMART'] = f"籌碼 {smart:.0f}"

    # 市場熱度（預設 50）
    hype = 50.0
    scores['HYPE'] = hype
    details['HYPE'] = f"熱度 {hype:.0f}"

    # 趨勢分數
    trend = get_trend_score(df)
    scores['TREND'] = trend
    details['TREND'] = f"趨勢 {trend:.0f}"

    # 成長分數
    growth, _ = get_growth_score(ticker, df)
    scores['GROWTH'] = growth
    details['GROWTH'] = f"成長 {growth:.0f}"

    # 三大法人分數（台股）
    foreign, trust, dealer, inst_date = get_institutional_data(ticker)
    inst_score = calculate_institutional_score(foreign, trust, dealer)
    scores['INST'] = inst_score
    details['INST'] = f"三大法人分數 {inst_score:.0f} (外資:{foreign if foreign else 0}, 投信:{trust if trust else 0}, 自營:{dealer if dealer else 0})"

    if not scores:
        return None, None, {}, None, None, None, None, None, None, None, None

    # 權重設定
    weights = {
        'RF': 0.22,
        'XGB': 0.22,
        'LGB': 0.18,
        'SMART': 0.08,
        'TREND': 0.08,
        'GROWTH': 0.07,
        'INST': 0.15
    }

    total_w = sum(weights[m] for m in scores if m in weights)
    if total_w == 0:
        return None, None, {}, None, None, None, None, None, None, None, None

    final = sum(scores[m] * weights[m] for m in scores if m in weights) / total_w
    final = round(final, 1)

    if final >= 85:
        signal = "強力買進"
    elif final >= 75:
        signal = "買進"
    elif final >= 65:
        signal = "試單買進"
    elif final >= 55:
        signal = "持有"
    elif final >= 45:
        signal = "觀望"
    else:
        signal = "賣出/避開"

    # 回傳值保持與原函數一致
    ten_bagger = get_ten_bagger_score(ticker)
    return final, signal, details, 0.5, smart, hype, trend, growth, ten_bagger, [], final
# ================== 期權信號計算 ==================
def get_options_chain(ticker):
    try:
        stock = yf.Ticker(ticker)
        exps = stock.options
        if not exps: return None, None, None, None
        nearest = exps[0]
        opt = stock.option_chain(nearest)
        if opt.calls.empty or opt.puts.empty: return None, None, None, None
        return opt.calls, opt.puts, nearest, stock
    except Exception as e:
        logger.debug(f"取得 {ticker} 期權鏈失敗: {e}")
        return None, None, None, None
def calculate_put_call_ratio(ticker):
    calls, puts, _, _ = get_options_chain(ticker)
    if calls is None: return None
    call_vol = calls['volume'].sum(); put_vol = puts['volume'].sum()
    if call_vol == 0: return None
    pcr = put_vol / call_vol
    if pcr < 0.7: sig, txt = "bullish", "☀️ 偏多 (Call 活跃)"
    elif pcr > 1.3: sig, txt = "bearish", "🌧️ 偏空 (Put 活跃)"
    else: sig, txt = "neutral", "⛅ 中性"
    return {"pcr": round(pcr,3), "signal": sig, "text": txt}
def calculate_max_pain(ticker):
    calls, puts, _, stock = get_options_chain(ticker)
    if calls is None: return None
    try:
        hist = stock.history(period="1d")
        if hist.empty: return None
        current = hist['Close'].iloc[-1]
    except: return None
    strikes = sorted(set(calls['strike']).union(set(puts['strike'])))
    pain = []
    for strike in strikes:
        call_pain = sum((row['strike']-strike)*row['openInterest'] for _, row in calls.iterrows() if row['strike']>strike)
        put_pain = sum((strike-row['strike'])*row['openInterest'] for _, row in puts.iterrows() if row['strike']<strike)
        pain.append((strike, call_pain+put_pain))
    if not pain: return None
    max_pain = min(pain, key=lambda x: x[1])[0]
    dist_pct = (current - max_pain)/max_pain*100
    if dist_pct > 5: signal = "⚠️ 高於 Max Pain 5%+ → 可能下跌"
    elif dist_pct < -5: signal = "📈 低於 Max Pain 5%+ → 可能上漲"
    else: signal = "📍 接近 Max Pain → 價格有吸引力"
    return {"max_pain": round(max_pain,2), "current": round(current,2), "dist_pct": round(dist_pct,2), "signal": signal}
def calculate_iv_rank(ticker):
    calls, puts, _, stock = get_options_chain(ticker)
    if calls is None: return None
    try:
        hist = stock.history(period="1d")
        if hist.empty: return None
        current = hist['Close'].iloc[-1]
    except: return None
    call_atm = calls.iloc[(calls['strike']-current).abs().argsort()[:1]]
    put_atm = puts.iloc[(puts['strike']-current).abs().argsort()[:1]]
    iv = (call_atm['impliedVolatility'].mean() + put_atm['impliedVolatility'].mean()) / 2 * 100
    iv = round(iv,2) if not np.isnan(iv) else 30
    if iv < 30: interp = "❄️ IV 偏低 (選擇權便宜)"
    elif iv > 70: interp = "🔥 IV 偏高 (選擇權貴)"
    else: interp = "⛅ IV 中性"
    return {"iv": iv, "text": interp}
def black_scholes_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return 0
    if SCIPY_AVAILABLE:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    else:
        distance_factor = max(0, 1 - abs(K - S) / S)
        return distance_factor * 0.1
def calculate_gamma_exposure(ticker, current_price=None, r=0.05):
    if '.TW' in ticker and not is_us_market_open():
        logger.info(f"⏰ 美股非交易時段，{ticker} GEX 可能為空")
    cached = get_cached_gex(ticker)
    if cached: return cached
    logger.info(f"🔍 開始獲取 {ticker} 期權數據")
    try:
        stock = yf.Ticker(ticker)
        exps = stock.options
        if not exps:
            logger.warning(f"⚠️ {ticker} 的 `.options` 返回空列表")
            return None
        logger.info(f"✅ {ticker} 取得 {len(exps)} 個到期日")
        if current_price is None:
            hist = stock.history(period="1d")
            if not hist.empty: current_price = hist['Close'].iloc[-1]
            else: return None
        hist_data = stock.history(period="60d")
        historical_vol = hist_data['Close'].pct_change().dropna().std() * np.sqrt(252) if not hist_data.empty else 0.3
        gex_data = {}
        for exp in exps[:3]:
            try:
                opt = stock.option_chain(exp)
                if opt.calls.empty and opt.puts.empty: continue
                T = max((datetime.strptime(exp, '%Y-%m-%d') - datetime.now()).days / 365, 1/365)
                for _, row in opt.calls.iterrows():
                    strike = row['strike']
                    if abs(strike - current_price) / current_price > 0.3: continue
                    oi = row['openInterest'] if not pd.isna(row['openInterest']) else 0
                    iv = row['impliedVolatility'] if not pd.isna(row['impliedVolatility']) else historical_vol
                    if oi <= 0: continue
                    gamma = black_scholes_gamma(current_price, strike, T, r, iv if iv > 0 else historical_vol)
                    gex_value = gamma * oi * 100 * current_price
                    gex_data[strike] = gex_data.get(strike, 0) + gex_value
                for _, row in opt.puts.iterrows():
                    strike = row['strike']
                    if abs(strike - current_price) / current_price > 0.3: continue
                    oi = row['openInterest'] if not pd.isna(row['openInterest']) else 0
                    iv = row['impliedVolatility'] if not pd.isna(row['impliedVolatility']) else historical_vol
                    if oi <= 0: continue
                    gamma = black_scholes_gamma(current_price, strike, T, r, iv if iv > 0 else historical_vol)
                    gex_value = -gamma * oi * 100 * current_price
                    gex_data[strike] = gex_data.get(strike, 0) + gex_value
            except Exception as e:
                logger.debug(f"⚠️ 到期日 {exp} 處理失敗: {e}")
                continue
        if not gex_data: return None
        strikes = sorted(gex_data.keys())
        gex_values = [gex_data[s] for s in strikes]
        total_positive_gex = sum(v for v in gex_values if v > 0)
        total_negative_gex = sum(v for v in gex_values if v < 0)
        gamma_flip = None
        for i in range(1, len(gex_values)):
            if gex_values[i-1] < 0 and gex_values[i] > 0:
                gamma_flip = strikes[i]; break
        pos_gex = [(strikes[i], gex_values[i]) for i in range(len(strikes)) if gex_values[i] > 0]
        neg_gex = [(strikes[i], gex_values[i]) for i in range(len(strikes)) if gex_values[i] < 0]
        call_wall = max(pos_gex, key=lambda x: x[1])[0] if pos_gex else None
        put_wall = min(neg_gex, key=lambda x: x[1])[0] if neg_gex else None
        max_val = max(abs(v) for v in gex_values) if gex_values else 1
        y_range = [-max_val * 1.2, max_val * 1.2]
        result = {
            "call_wall": round(call_wall,2) if call_wall else None,
            "put_wall": round(put_wall,2) if put_wall else None,
            "current_price": round(current_price,2),
            "gamma_flip_strike": round(gamma_flip,2) if gamma_flip else None,
            "net_total_gex": sum(gex_values),
            "total_positive_gex": total_positive_gex,
            "total_negative_gex": total_negative_gex,
            "strikes": strikes, "gex_values": gex_values, "y_range": y_range
        }
        set_cached_gex(ticker, result)
        logger.info(f"✅ {ticker} GEX 計算成功")
        return result
    except Exception as e:
        logger.error(f"❌ {ticker} GEX 計算失敗: {e}")
        return None
def get_options_signals(ticker):
    if '.TW' in ticker: return None
    pcr = calculate_put_call_ratio(ticker)
    mp = calculate_max_pain(ticker)
    iv = calculate_iv_rank(ticker)
    gex = calculate_gamma_exposure(ticker)
    if not pcr and not mp and not iv and not gex: return None
    score = 0; reasons = []
    if pcr:
        if pcr['signal']=='bullish': score+=2; reasons.append("PCR 偏多")
        elif pcr['signal']=='bearish': score-=2; reasons.append("PCR 偏空")
    if mp:
        if mp['dist_pct'] < -5: score+=1.5; reasons.append(f"低於 Max Pain {abs(mp['dist_pct']):.1f}%")
        elif mp['dist_pct'] > 5: score-=1.5; reasons.append(f"高於 Max Pain {mp['dist_pct']:.1f}%")
    if iv:
        if iv['iv'] < 30: score+=1; reasons.append("IV 偏低")
        elif iv['iv'] > 70: score-=1; reasons.append("IV 偏高")
    if gex:
        if gex['call_wall'] and gex['current_price'] > gex['call_wall']: score-=1.5; reasons.append(f"高於 Call Wall ${gex['call_wall']:.2f}")
        if gex['put_wall'] and gex['current_price'] < gex['put_wall']: score+=1.5; reasons.append(f"低於 Put Wall ${gex['put_wall']:.2f}")
        if gex['gamma_flip_strike'] and gex['current_price'] > gex['gamma_flip_strike']: score+=1; reasons.append(f"站上 Gamma Flip ${gex['gamma_flip_strike']:.2f}")
    if score >= 2: composite, color = "🔥 期權信號強烈偏多", "#00ff00"
    elif score >= 0.5: composite, color = "📈 期權信號偏多", "#88ff88"
    elif score <= -2: composite, color = "❄️ 期權信號強烈偏空", "#ff4444"
    elif score <= -0.5: composite, color = "📉 期權信號偏空", "#ff8844"
    else: composite, color = "⛅ 期權信號中性", "#cccccc"
    return {"pcr":pcr,"max_pain":mp,"iv":iv,"gex":gex, "composite":{"score":round(score,2),"text":composite,"color":color,"reasons":reasons}}
# ================== 資料處理函數 ==================
def safe_get_close(df):
    if df is None or df.empty: return pd.Series(dtype=float)
    try:
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        close = df['Close']
        if isinstance(close, pd.DataFrame): close = close.iloc[:,0]
        return pd.Series(close.values.flatten(), index=close.index, name='Close').dropna()
    except: return pd.Series(dtype=float)
def get_us_stock_data(ticker):
    df = get_cached_data(ticker, period="2y")
    if df.empty: return 0,0,0,df
    try:
        tk = yf.Ticker(ticker); fast = tk.fast_info
        curr = fast.get('last_price', None); prev = fast.get('previous_close', None)
        if curr is not None and prev is not None:
            change = curr-prev; pct = change/prev*100 if prev!=0 else 0
            return curr, change, pct, df
    except: pass
    close = safe_get_close(df)
    if len(close)<2: return 0,0,0,df
    curr = float(close.iloc[-1]); prev = float(close.iloc[-2])
    change = curr-prev; pct = change/prev*100 if prev!=0 else 0
    return curr, change, pct, df
def get_tw_stock_data(ticker):
    if not ticker.endswith('.TW'): ticker_with_suffix = ticker + '.TW'
    else: ticker_with_suffix = ticker
    curr, change, pct = get_tw_stock_realtime_shioaji(ticker_with_suffix)
    _,_,_,df_hist = get_us_stock_data(ticker_with_suffix)
    if curr is not None and curr>0 and not math.isnan(curr): return curr, change, pct, df_hist
    else:
        if df_hist.empty or len(df_hist)<2: return 0,0,0,df_hist
        curr = float(df_hist['Close'].iloc[-1]); prev = float(df_hist['Close'].iloc[-2])
        if math.isnan(curr) or math.isnan(prev): return 0,0,0,df_hist
        change = curr-prev; pct = change/prev*100 if prev!=0 else 0
        return curr, change, pct, df_hist
def get_all_tickers(market):
    file_path = STOCK_FILE_TW if market=='tw' else STOCK_FILE_US
    tickers = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().upper()
            if t:
                if market == 'tw' and not t.endswith('.TW'): t = t + '.TW'
                tickers.append(t)
    return tickers
def compute_stock_summary(ticker, market):
  
    # 原本的程式碼繼續...
    try:
        if market=='tw': curr, change, pct, df = get_tw_stock_data(ticker)
        else: curr, change, pct, df = get_us_stock_data(ticker)
        if curr==0 or df.empty: return None
        close = df['Close'].values
        if len(close)>=15:
            deltas = np.diff(close[-15:])
            gain = np.mean(deltas[deltas>0]) if any(deltas>0) else 0
            loss = -np.mean(deltas[deltas<0]) if any(deltas<0) else 0
            rs = gain/loss if loss>0 else 1
            rsi = 100 - 100/(1+rs)
        else: rsi=50
        close_series = safe_get_close(df)
        if len(close_series)>=35: _,_,_,macd_status = calculate_macd(close_series)
        else: macd_status = "-"
        vwap = calculate_vwap(df)
        vwap_status = get_vwap_status(curr, vwap) if vwap else "N/A"
        volume = df['Volume'].values
        vol_ma5 = np.mean(volume[-5:]) if len(volume)>=5 else volume[-1]
        vol_ratio = volume[-1]/vol_ma5 if vol_ma5>0 else 1
        ai_score = calculate_ai_resonance_score(rsi, macd_status, vwap_status, vol_ratio)
        ma20 = np.mean(close[-20:]) if len(close)>=20 else curr
        ma60 = np.mean(close[-60:]) if len(close)>=60 else curr
        ma200 = np.mean(close[-200:]) if len(close)>=200 else curr
        trend = "多頭" if curr>ma20 else "空頭"
        if ai_score>=70 and trend=="多頭": signal = "買進"
        elif ai_score<40 or trend=="空頭": signal = "賣出/避開"
        else: signal = "持有/觀望"
        return {'ticker':ticker,'price':round(curr,2),'change':round(change,2),'pct':round(pct,2),
                'ai_score':ai_score,'trend':trend,'signal':signal,'market':market,'rsi':round(rsi,1),
                'macd_status':macd_status,'vwap_status':vwap_status,'volume_ratio':round(vol_ratio,2),
                'ma20':round(ma20,2),'ma60':round(ma60,2),'ma200':round(ma200,2)}
    except Exception as e: 
        logger.error(f"計算 {ticker} 摘要失敗: {e}")
        return None
def update_all_ai_scores():
    logger.info("開始背景更新 AI 評分...")
    start_time = time.time()
    all_data = []
    for market in ['tw','us']:
        for ticker in get_all_tickers(market):
            s = compute_stock_summary(ticker, market)
            if s: all_data.append(s)
    with open(AI_SCORES_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    elapsed = time.time() - start_time
    logger.info(f"背景更新完成，共更新 {len(all_data)} 檔股票，耗時 {elapsed:.1f} 秒")
def update_tenbagger():
    logger.info("開始背景更新十倍股雷達...")
    candidates = []
    for market in ['tw','us']:
        for ticker in get_all_tickers(market):
            radar = tenbagger_radar(ticker, market)
            if radar and radar.get('score', 0) >= 3: candidates.append(radar)
    candidates.sort(key=lambda x: x.get('score',0), reverse=True)
    with open(TENBAGGER_FILE, 'w', encoding='utf-8') as f:
        json.dump(candidates[:20], f, indent=2, ensure_ascii=False)
    logger.info(f"十倍股雷達更新完成，共 {len(candidates)} 檔候選")
def tenbagger_radar(ticker, market):
    try:
        if market=='tw': curr,_,_,df = get_tw_stock_data(ticker)
        else: curr,_,_,df = get_us_stock_data(ticker)
        if df.empty or len(df)<60: return None
        revenue_growth = 0
        try:
            tk = yf.Ticker(ticker)
            qf = tk.quarterly_financials
            if not qf.empty and 'Total Revenue' in qf.index:
                rev = qf.loc['Total Revenue'].dropna()
                if len(rev)>=2: revenue_growth = (rev.iloc[0]-rev.iloc[1])/rev.iloc[1]*100
        except: pass
        gm_growth = 0
        try:
            inc = yf.Ticker(ticker).quarterly_income_stmt
            if not inc.empty and 'Gross Profit' in inc.index and 'Total Revenue' in inc.index:
                gp = inc.loc['Gross Profit'].dropna(); rev = inc.loc['Total Revenue'].dropna()
                if len(gp)>=2 and len(rev)>=2:
                    gm1 = gp.iloc[0]/rev.iloc[0]*100; gm2 = gp.iloc[1]/rev.iloc[1]*100
                    gm_growth = gm1 - gm2
        except: pass
        high_52w = df['High'].rolling(252).max().iloc[-1] if len(df)>=252 else df['High'].max()
        is_high = curr >= high_52w*0.98
        vol = df['Volume'].values
        vol_ma20 = np.mean(vol[-20:]) if len(vol)>=20 else vol[-1]
        vol_surge = vol[-1] > vol_ma20*1.5
        short_high = False
        if market=='us':
            try:
                short = yf.Ticker(ticker).info.get('shortPercentOfFloat',0)*100
                short_high = short > 15
            except: pass
        score = 0; cond = []
        if revenue_growth>30: score+=1; cond.append(f"營收成長 {revenue_growth:.1f}%")
        if gm_growth>5: score+=1; cond.append(f"毛利率成長 {gm_growth:.1f}%")
        if is_high: score+=1; cond.append("52週新高")
        if vol_surge: score+=1; cond.append("爆量")
        if short_high: score+=1; cond.append("高空頭比例")
        return {'ticker':ticker,'price':round(curr,2),'score':score,'conditions':cond}
    except: return None
def scan_alerts():
    alerts = []
    try:
        with open(AI_SCORES_FILE, 'r', encoding='utf-8') as f: stocks = json.load(f)
    except: return alerts
    for s in stocks[:100]:
        ticker, market = s['ticker'], s['market']
        if market=='tw': curr,_,_,df = get_tw_stock_data(ticker)
        else: curr,_,_,df = get_us_stock_data(ticker)
        if df.empty: continue
        vol = df['Volume'].values
        vol_ma20 = np.mean(vol[-20:]) if len(vol)>=20 else vol[-1]
        vol_ratio = vol[-1]/vol_ma20 if vol_ma20>0 else 1
        if vol_ratio>3: alerts.append({"ticker":ticker,"type":"volume_surge","message":f"成交量爆增 {vol_ratio:.1f}倍","importance":"high"})
        high_52w = df['High'].rolling(252).max().iloc[-1] if len(df)>=252 else df['High'].max()
        if curr >= high_52w*0.99: alerts.append({"ticker":ticker,"type":"new_high","message":"突破52週新高","importance":"high"})
    alerts.sort(key=lambda x: 0 if x['importance']=='high' else 1)
    with open(ALERTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(alerts[:20], f, indent=2, ensure_ascii=False)
    return alerts
def run_backtest_advanced(ticker, days=180, benchmark_ticker='^GSPC'):
    try:
        stock_df = get_cached_data(ticker, period="1y")
        if stock_df.empty or len(stock_df)<50: return None
        try:
            bench_df = get_cached_data(benchmark_ticker, period="1y")
            bench_returns = bench_df['Close'].pct_change().dropna()
        except: bench_returns = None
        df_test = stock_df.tail(days)
        capital, position = 10000, 0
        portfolio, dates = [], []
        for i in range(20, len(df_test)):
            close = df_test['Close'].values[:i]
            if len(close)<20: continue
            ma5, ma20 = np.mean(close[-5:]), np.mean(close[-20:])
            price = close[-1]
            deltas = np.diff(close[-15:]) if len(close)>=15 else np.diff(close)
            gain = np.mean(deltas[deltas>0]) if any(deltas>0) else 0
            loss = -np.mean(deltas[deltas<0]) if any(deltas<0) else 0
            rsi = 100 - 100/(1+ (gain/loss if loss else 1))
            buy = price > ma5 and price > ma20 and rsi < 70
            sell = price < ma5 or rsi > 70
            if buy and position==0 and capital>0:
                shares = int(capital/price)
                if shares>0: capital -= shares*price; position = shares
            elif sell and position>0:
                capital += position*price; position = 0
            portfolio.append(capital + position*price)
            dates.append(df_test.index[i])
        ps = pd.Series(portfolio, index=dates)
        rets = ps.pct_change().dropna()
        total_ret = (ps.iloc[-1]-10000)/10000*100
        years = len(rets)/252
        cagr = (ps.iloc[-1]/10000)**(1/years)-1 if years>0 else 0
        cum = (1+rets).cumprod()
        dd = (cum - cum.expanding().max()) / cum.expanding().max()
        max_dd = dd.min()*100
        rf = 0.02/252
        excess = rets - rf
        sharpe = np.sqrt(252)*excess.mean()/rets.std() if rets.std()!=0 else 0
        downside = rets[rets<0]
        sortino = np.sqrt(252)*excess.mean()/downside.std() if len(downside)>0 and downside.std()!=0 else 0
        if bench_returns is not None:
            common = rets.index.intersection(bench_returns.index)
            if len(common)>0:
                sr = rets[common]; br = bench_returns[common]
                cov = np.cov(sr, br)[0,1]
                beta = cov/np.var(br) if np.var(br)!=0 else 1
                alpha = (sr.mean()-rf) - beta*(br.mean()-rf)
                alpha_annual = alpha*252*100
            else: beta, alpha_annual = 1, 0
        else: beta, alpha_annual = 1, 0
        return {'initial_capital':10000,'final_value':round(ps.iloc[-1],2),'total_return':round(total_ret,2),
                'cagr':round(cagr*100,2),'max_drawdown':round(max_dd,2),'sharpe_ratio':round(sharpe,2),
                'sortino_ratio':round(sortino,2),'alpha':round(alpha_annual,2),'beta':round(beta,2)}
    except Exception as e:
        logger.error(f"進階回測 {ticker} 失敗: {e}")
        return None
def get_backtest_performance(ticker):
    try:
        df = get_cached_data(ticker, period="2y")
        if df.empty or len(df)<100: return None
        close = df['Close'].values
        trades = []
        for i in range(60, len(close)-5):
            seg = df.iloc[i-60:i]
            feats = calculate_features(seg)
            if feats is None: continue
            model, scaler = get_rf_model(ticker)
            if model and scaler:
                X = np.array(get_feature_vector(feats)).reshape(1,-1)
                X_scaled = scaler.transform(X)
                pred = model.predict(X_scaled)[0]
                prob = 100 / (1 + math.exp(-pred*10))
                if prob > 65:
                    ret = (close[i+5]-close[i])/close[i]*100
                    trades.append(ret)
        if not trades: return None
        wins = sum(1 for r in trades if r>0)
        win_rate = wins/len(trades)*100
        avg_profit = np.mean([r for r in trades if r>0]) if any(r>0 for r in trades) else 0
        avg_loss = abs(np.mean([r for r in trades if r<0])) if any(r<0 for r in trades) else 0
        total_profit = sum(r for r in trades if r>0)
        total_loss = abs(sum(r for r in trades if r<0))
        pf = total_profit/total_loss if total_loss>0 else 0
        return {'total_signals':len(trades),'win_rate':round(win_rate,1),'avg_profit':round(avg_profit,1),'avg_loss':round(avg_loss,1),'profit_factor':round(pf,2)}
    except Exception as e:
        logger.error(f"回測統計失敗 {ticker}: {e}")
        return None
def find_ma_bonding_stocks(tickers, threshold=0.03):
    results = []
    ma_list = [5,20,60]
    for ticker in tickers:
        try:
            df = get_cached_data(ticker, period="3mo", ttl_seconds=300)
            if df.empty or len(df) < max(ma_list): continue
            close = df['Close']; vol = df['Volume'].values
            price = close.iloc[-1]
            ma_vals = [close.rolling(ma).mean().iloc[-1] for ma in ma_list]
            ma_max, ma_min = max(ma_vals), min(ma_vals)
            spread = (ma_max - ma_min) / ma_min if ma_min>0 else 1
            if spread < threshold:
                avg_vol = np.mean(vol[-20:]) if len(vol)>=20 else vol[-1]
                vol_confirm = vol[-1] > avg_vol*1.2
                if price > ma_max: direction = "向上突破可能"
                elif price < ma_min: direction = "向下突破可能"
                else: direction = "糾結區內待方向"
                results.append({'ticker':ticker,'price':round(price,2),'spread':round(spread*100,2),
                                'ma5':round(ma_vals[0],2),'ma20':round(ma_vals[1],2),'ma60':round(ma_vals[2],2),
                                'direction':direction,'volume_confirm':vol_confirm})
        except: continue
    results.sort(key=lambda x: x['spread'])
    return results[:50]

# ================== 市場數據函數（已修復 NaN/0）==================
import requests

EOD_API_KEY = os.getenv("EOD_API_KEY", "")
EOD_BASE_URL = "https://eodhistoricaldata.com/api"
if not EOD_API_KEY:
    logger.warning("⚠️ EOD_API_KEY 未設定，EOD 歷史資料功能將無法使用")

def get_eod_stock_data(ticker, period="1y"):
    try:
        if '.TW' in ticker:
            eod_ticker = ticker.replace('.TW', '.TW')
        else:
            eod_ticker = f"{ticker}.US"
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        url = f"{EOD_BASE_URL}/eod/{eod_ticker}"
        params = {'api_token': EOD_API_KEY, 'from': start_date, 'to': end_date, 'period': 'd', 'fmt': 'json'}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.rename(columns={'adjusted_close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
            if 'Close' not in df.columns and 'adjusted_close' in df.columns:
                df['Close'] = df['adjusted_close']
            return df
        else:
            logger.warning(f"EOD API 請求失敗: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"EOD API 錯誤: {e}")
        return pd.DataFrame()

def get_eod_realtime_quote(ticker):
    try:
        if '.TW' in ticker:
            eod_ticker = ticker.replace('.TW', '.TW')
        else:
            eod_ticker = f"{ticker}.US"
        url = f"{EOD_BASE_URL}/real-time/{eod_ticker}"
        params = {'api_token': EOD_API_KEY, 'fmt': 'json'}
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            last = data.get('last', data.get('close', 0))
            prev_close = data.get('previous_close', 0)
            if last and prev_close and prev_close != 0:
                change = last - prev_close
                pct = (change / prev_close) * 100
                return last, change, pct, "即時 (EOD)"
        return None, None, None, "延遲"
    except:
        return None, None, None, "錯誤"
def get_vix():
    try: return yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
    except: return 20.0
def get_treasury_yield():
    try: return yf.Ticker("^TNX").history(period="5d")['Close'].iloc[-1]
    except: return 4.2
def get_dollar_index():
    try: return yf.Ticker("DX-Y.NYB").history(period="5d")['Close'].iloc[-1]
    except: return 104.0
def get_market_trend():
    try:
        sp = yf.Ticker("^GSPC").history(period="1mo")['Close']
        nas = yf.Ticker("^IXIC").history(period="1mo")['Close']
        if sp.empty or nas.empty: return 50
        sp_ret = (sp.iloc[-1]-sp.iloc[0])/sp.iloc[0]*100
        nas_ret = (nas.iloc[-1]-nas.iloc[0])/nas.iloc[0]*100
        return max(0, min(100, 50 + (sp_ret+nas_ret)*2))
    except: return 50
def calculate_market_temperature():
    vix = get_vix(); yield_val = get_treasury_yield(); dollar = get_dollar_index(); trend = get_market_trend()
    vix_score = 30 if vix<15 else (15 if vix<25 else -20)
    yield_score = 20 if yield_val<4 else (0 if yield_val<5 else -15)
    dollar_score = 10 if dollar<103 else (-10 if dollar>105 else 0)
    total = trend + vix_score + yield_score + dollar_score
    total = max(0,min(100,total))
    if total>=75: status = "🔴 風險偏好高，適合積極操作"
    elif total>=45: status = "⚪ 中性市場，平衡配置"
    else: status = "🟢 避險情緒濃厚，降低持股"
    return round(total), status

US_SECTOR_ETFS = {"AI":"AIQ","電力":"XLU","機器人":"ARKQ","核能":"URA","網路安全":"CIBR"}
TW_SECTOR_ETFS = {"半導體":"0050.TW","電子":"0052.TW","金融":"0055.TW"}
def get_us_sector_performance():
    results = []
    for name, etf in US_SECTOR_ETFS.items():
        try:
            df = get_cached_data(etf, period="2d")
            if not df.empty and len(df)>=2:
                ret = (df['Close'].iloc[-1]-df['Close'].iloc[0])/df['Close'].iloc[0]*100
                results.append((name, round(ret,2)))
            else: results.append((name,0))
        except: results.append((name,0))
    results.sort(key=lambda x: x[1], reverse=True)
    return results
def get_tw_sector_performance():
    results = []
    for name, etf in TW_SECTOR_ETFS.items():
        try:
            df = get_cached_data(etf, period="2d")
            if not df.empty and len(df)>=2:
                ret = (df['Close'].iloc[-1]-df['Close'].iloc[0])/df['Close'].iloc[0]*100
                results.append((name, round(ret,2)))
            else: results.append((name,0))
        except: results.append((name,0))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ================== 即時指數函數 ==================
def get_index_data(symbol):
    try:
        tk = yf.Ticker(symbol)
        try:
            fast = tk.fast_info
            curr = fast.get("lastPrice") or fast.get("last_price")
            prev = fast.get("previousClose") or fast.get("previous_close")
            if curr and prev and prev != 0:
                change = curr - prev
                pct = change / prev * 100
                return round(float(curr),2), round(float(change),2), round(float(pct),2)
        except:
            pass
        df = tk.history(period="1d", interval="1m", prepost=True)
        if df is not None and len(df) >= 2:
            curr = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[0])
            if prev != 0:
                return round(curr,2), round(curr-prev,2), round((curr-prev)/prev*100,2)
        df = tk.history(period="5d")
        if df is not None and len(df) >= 2:
            curr = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            return round(curr,2), round(curr-prev,2), round((curr-prev)/prev*100,2)
        return 0,0,0
    except Exception:
        return 0,0,0

def get_tw_index(): return get_index_data("^TWII")
def get_otc_index(): return get_index_data("^TWOII")
def get_dow_index(): return get_index_data("^DJI")
def get_nasdaq_index(): return get_index_data("^IXIC")
def get_phlx_index(): return get_index_data("^SOX")
def get_sp500_index(): return get_index_data("^GSPC")
def get_russell_index(): return get_index_data("^RUT")

def get_buffett_indicator():
    try:
        wilshire = yf.Ticker("^W5000").history(period="1d")
        if not wilshire.empty:
            market_cap_trillion = wilshire['Close'].iloc[-1] / 1000
        else:
            market_cap_trillion = 45
        gdp_trillion = 27
        ratio = (market_cap_trillion / gdp_trillion) * 100
        return round(ratio, 1), f"{ratio:.1f}% (參考值)", "https://www.macromicro.me/charts/406/us-buffet-index-gspc"
    except:
        return None, "參考 MacroMicro 數據", "https://www.macromicro.me/charts/406/us-buffet-index-gspc"

def get_yield_curve_inversion():
    try:
        tnx = yf.Ticker("^TNX"); fvx = yf.Ticker("^FVX")
        df_tnx = tnx.history(period="1d"); df_fvx = fvx.history(period="1d")
        if df_tnx.empty or df_fvx.empty:
            return None, None, "無法取得殖利率資料"
        tnx_rate = df_tnx['Close'].iloc[-1]; fvx_rate = df_fvx['Close'].iloc[-1]
        spread = tnx_rate - fvx_rate
        if spread < 0:
            inversion = True
            text = f"⚠️ 殖利率曲線倒掛 ({tnx_rate:.2f}% - {fvx_rate:.2f}% = {spread:.2f}%) → 經濟衰退警訊"
        else:
            inversion = False
            text = f"✅ 殖利率曲線正常 ({tnx_rate:.2f}% - {fvx_rate:.2f}% = {spread:.2f}%)"
        return round(spread, 2), inversion, text
    except Exception:
        return None, None, "無法取得殖利率資料"

def get_cnn_fear_greed():
    try:
        import fear_greed
        data = fear_greed.get()
        rating_map = {'extreme fear': '極度恐懼', 'fear': '恐懼', 'neutral': '中性', 'greed': '貪婪', 'extreme greed': '極度貪婪'}
        score = data['score']
        level = rating_map.get(data['rating'], '中性')
        return score, level, "https://edition.cnn.com/markets/fear-and-greed"
    except Exception as e:
        logger.warning(f"恐懼貪婪指數獲取失敗: {e}")
        return 50, "中性", "https://edition.cnn.com/markets/fear-and-greed"

def get_margin_data():
    url = "https://www.wantgoo.com/stock/margin-trading/utilization-rate-rank"
    balance = "約 3,200 億"
    maintenance_ratio = "約 160%"
    return balance, maintenance_ratio, url
import math

def clean_nan(value, default='N/A'):
    if value is None: return default
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value): return default
        return round(value, 2)
    if isinstance(value, dict):
        return {k: clean_nan(v, default) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_nan(v, default) for v in value]
    return value
# ================== 三大法人與主力籌碼數據整合 ==================
def extract_stock_number(ticker):
    """從 '2330.TW' 提取 '2330'"""
    return ticker.replace('.TW', '')

def update_institutional_data(ticker):
    """更新單一股票的三大法人資料到資料庫"""
    foreign, trust, dealer, data_date = fetch_institutional_holders(ticker)
    if foreign is None:
        logger.warning(f"{ticker} 無三大法人資料，跳過更新")
        return
    with closing(get_db_connection()) as conn:
        cur = conn.cursor()
        if DATABASE_URL:
            # PostgreSQL 語法：使用 ON CONFLICT 處理重複
            cur.execute('''
                INSERT INTO institutional_ownership (stock_id, date, foreign_investors, investment_trust, dealers)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (stock_id, date) DO UPDATE SET
                    foreign_investors = EXCLUDED.foreign_investors,
                    investment_trust = EXCLUDED.investment_trust,
                    dealers = EXCLUDED.dealers
            ''', (ticker, data_date, foreign, trust, dealer))
        else:
            # SQLite 語法
            cur.execute('''
                INSERT OR REPLACE INTO institutional_ownership (stock_id, date, foreign_investors, investment_trust, dealers)
                VALUES (?, ?, ?, ?, ?)
            ''', (ticker, data_date, foreign, trust, dealer))
        conn.commit()
    logger.info(f"更新 {ticker} ({data_date}) 三大法人: 外資={foreign}, 投信={trust}, 自營={dealer}")
def update_large_shareholders_data(ticker):
    """更新千張大戶持股比率到資料庫"""
    ratio = fetch_large_shareholders(ticker)
    if ratio is None:
        logger.warning(f"{ticker} 無千張大戶資料，跳過更新")
        return
    today = datetime.now().strftime('%Y%m%d')
    with closing(get_db_connection()) as conn:
        cur = conn.cursor()
        if DATABASE_URL:
            cur.execute('''
                INSERT INTO large_shareholders (stock_id, date, holding_ratio, shareholders_count)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (stock_id, date) DO UPDATE SET
                    holding_ratio = EXCLUDED.holding_ratio,
                    shareholders_count = EXCLUDED.shareholders_count
            ''', (ticker, today, ratio, 0))
        else:
            cur.execute('''
                INSERT OR REPLACE INTO large_shareholders (stock_id, date, holding_ratio, shareholders_count)
                VALUES (?, ?, ?, ?)
            ''', (ticker, today, ratio, 0))
        conn.commit()
    logger.info(f"更新 {ticker} 千張大戶持股比率: {ratio:.2f}%")
    # 每日更新三大法人資料（週一至週五 16:30）
    scheduler.add_job(
        func=lambda: [update_institutional_data(t) for t in get_all_tickers('tw')],
        trigger="cron",
        day_of_week='mon-fri',
        hour=16,
        minute=30,
        max_instances=1
    )
    # 每週六更新千張大戶持股（週五資料）
    scheduler.add_job(
        func=lambda: [update_large_shareholders_data(t) for t in get_all_tickers('tw')],
        trigger="cron",
        day_of_week='sat',
        hour=8,
        minute=0,
        max_instances=1
    )
# ================== 持倉函數 ==================
def load_positions():
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    return []
def save_positions(positions):
    with open(POSITIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(positions, f, indent=2, ensure_ascii=False)
def add_multiple_stocks(tickers_str, market):
    tickers = re.split(r'[,\s\n]+', tickers_str.strip().upper())
    tickers = [t for t in tickers if t]
    if market == 'tw':
        tickers = [t if t.endswith('.TW') else t + '.TW' for t in tickers]
    file_path = STOCK_FILE_TW if market=='tw' else STOCK_FILE_US
    with open(file_path, 'r', encoding='utf-8') as f:
        existing = [l.strip().upper() for l in f if l.strip()]
    new_tickers = [t for t in tickers if t not in existing]
    with open(file_path, 'a', encoding='utf-8') as f:
        for t in new_tickers: f.write(f"{t}\n")
    return len(new_tickers), new_tickers

# ================== 研究連結函數 ==================
def get_research_links(ticker):
    ticker_clean = ticker.replace('.TW', '')
    if '.TW' in ticker:
        return {"TradingView": f"https://www.tradingview.com/chart/?symbol={ticker_clean}",
                "Yahoo": f"https://tw.stock.yahoo.com/quote/{ticker_clean}.TW",
                "Google": f"https://www.google.com/search?q={ticker_clean}+台股",
                "WantGoo": f"https://www.wantgoo.com/stock/{ticker_clean}",
                "GoodInfo": f"https://goodinfo.tw/tw/stockdetail.asp?STOCK_ID={ticker_clean}"}
    else:
        return {"TradingView": f"https://www.tradingview.com/chart/?symbol={ticker_clean}",
                "Finviz": f"https://finviz.com/quote.ashx?t={ticker_clean}",
                "MarketWatch": f"https://www.marketwatch.com/investing/stock/{ticker_clean}",
                "Yahoo": f"https://finance.yahoo.com/quote/{ticker_clean}?lang=zh-Hant",
                "Reddit": f"https://www.reddit.com/search/?q={ticker_clean}%20stock",
                "Barchart": f"https://www.barchart.com/stocks/quotes/{ticker_clean}/options"}

# ================== GEX 篩選器函數 ==================
def update_all_gex_screener():
    logger.info("開始背景更新 GEX 篩選器...")
    start_time = time.time()
    tickers = get_all_tickers('us')
    results = []
    for ticker in tickers:
        try:
            gex = calculate_gamma_exposure(ticker)
            if gex is None: continue
            results.append({"ticker":ticker,"current_price":gex.get("current_price"),"net_gex":gex.get("net_total_gex"),
                            "positive_gex":gex.get("total_positive_gex"),"negative_gex":gex.get("total_negative_gex"),
                            "call_wall":gex.get("call_wall"),"put_wall":gex.get("put_wall"),"gamma_flip":gex.get("gamma_flip_strike"),
                            "last_update":time.time()})
            time.sleep(0.2)
        except Exception as e: logger.error(f"GEX 篩選器更新 {ticker} 失敗: {e}")
    results.sort(key=lambda x: abs(x.get("net_gex",0)) if x.get("net_gex") is not None else 0, reverse=True)
    with open(GEX_SCREENER_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    elapsed = time.time() - start_time
    logger.info(f"GEX 篩選器更新完成，共 {len(results)} 檔，耗時 {elapsed:.1f} 秒")

# ================== 美股財報離線快取 ==================
def update_earnings_dates():
    logger.info("開始背景更新美股財報資料...")
    tickers = get_all_tickers('us')
    earnings_data = {}
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            cal = tk.calendar
            earnings_date = None
            if cal is not None and 'Earnings Date' in cal and len(cal['Earnings Date']) > 0:
                earnings_date = cal['Earnings Date'][0]
            expected_move = None
            try:
                if tk.options:
                    opt = tk.option_chain(tk.options[0])
                    if not opt.calls.empty:
                        current_price = tk.info.get('regularMarketPrice', tk.history(period='1d')['Close'].iloc[-1])
                        idx = (opt.calls['strike'] - current_price).abs().idxmin()
                        atm_strike = opt.calls.loc[idx, 'strike']
                        call_price = opt.calls[opt.calls['strike']==atm_strike]['lastPrice'].values[0] if len(opt.calls[opt.calls['strike']==atm_strike])>0 else 0
                        put_price = opt.puts[opt.puts['strike']==atm_strike]['lastPrice'].values[0] if len(opt.puts[opt.puts['strike']==atm_strike])>0 else 0
                        straddle = call_price + put_price
                        expected_move = round((straddle / current_price) * 100, 2)
            except: pass
            earnings_data[ticker] = {"date": earnings_date.isoformat() if earnings_date else None,
                                     "expected_move": expected_move, "last_updated": time.time()}
            time.sleep(0.1)
        except Exception as e:
            logger.warning(f"獲取 {ticker} 財報資料失敗: {e}")
            earnings_data[ticker] = {"date": None, "expected_move": None, "last_updated": time.time()}
    with open(EARNINGS_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(earnings_data, f, indent=2, ensure_ascii=False)
    logger.info(f"美股財報資料快取完成，共 {len(earnings_data)} 檔")
def get_earnings_info(ticker):
    try:
        with open(EARNINGS_CACHE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get(ticker, {})
    except: return {}
# ================== 強化戰術建議函數 ==================
def generate_tactical_advice(current_price, call_wall, put_wall, gamma_flip, ma_trend, vwap_status, ai_signal=None, unusual_options=None):
    advice = []
    if current_price is None: return "⚠️ 無 GEX 數據，無法提供戰術建議"
    if call_wall is not None:
        dist = (call_wall - current_price) / current_price * 100
        if dist < 1 and dist > 0: advice.append(f"🔴 股價距離 Call Wall ${call_wall:.2f} 僅 {dist:.2f}%，面臨強壓力")
        elif dist < 0: advice.append(f"🚨 突破 Call Wall ${call_wall:.2f}，多頭強勢")
        else: advice.append(f"📈 低於 Call Wall {dist:.2f}%，上行空間充足")
    else: advice.append("📊 Call Wall 暫無")
    if put_wall is not None:
        dist = (current_price - put_wall) / current_price * 100
        if dist < 1 and dist > 0: advice.append(f"🟢 接近 Put Wall ${put_wall:.2f}，強支撐")
        elif dist < 0: advice.append(f"⚠️ 跌破 Put Wall ${put_wall:.2f}，加速下跌風險")
        else: advice.append(f"📉 高於 Put Wall {dist:.2f}%，支撐較遠")
    else: advice.append("📊 Put Wall 暫無")
    if gamma_flip is not None:
        if current_price > gamma_flip: advice.append(f"⚡ 站上 Gamma Flip ${gamma_flip:.2f}，做市商撐盤")
        else: advice.append(f"⚡ 低於 Gamma Flip ${gamma_flip:.2f}，易加速下跌")
    else: advice.append("📊 Gamma Flip 暫無")
    if ma_trend == "多頭排列": advice.append("🐂 均線多頭排列，中期強勢")
    elif ma_trend == "空頭排列": advice.append("🐻 均線空頭排列，反彈賣點")
    if vwap_status != "N/A":
        if "站上" in vwap_status: advice.append("📊 站上 VWAP")
        else: advice.append("📊 跌破 VWAP")
    if ai_signal:
        if ai_signal == "買進" or ai_signal == "強力買進":
            advice.append("🤖 AI 模型偏多，順勢操作")
        elif ai_signal == "賣出/避開":
            advice.append("🤖 AI 模型偏空，嚴控風險")
    if unusual_options:
        advice.append(f"📢 異常期權: {unusual_options}")
    if not advice: return "⚖️ 區間震盪"
    return " | ".join(advice)
def calculate_option_delta(ticker, expiration_date, strike, option_type='call'):
    """
    計算特定期權的 Delta 值
    - ticker: 股票代號（如 AAPL）
    - expiration_date: 到期日（YYYY-MM-DD）
    - strike: 履約價
    - option_type: 'call' 或 'put'
    """
    try:
        import yfinance as yf
        from datetime import datetime
        import numpy as np
        from scipy.stats import norm

        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            return None
        current_price = hist['Close'].iloc[-1]

        # 計算到期天數（年化）
        expiry = datetime.strptime(expiration_date, '%Y-%m-%d')
        T = (expiry - datetime.now()).days / 365
        if T <= 0:
            return None

        # 獲取歷史波動率（使用 60 天年化波動率）
        hist_data = stock.history(period="60d")
        if len(hist_data) < 2:
            sigma = 0.3  # 預設值
        else:
            sigma = hist_data['Close'].pct_change().dropna().std() * np.sqrt(252)

        # 無風險利率（此處使用 4.2% 近似）
        r = 0.042

        # 計算 Delta
        d1 = (np.log(current_price / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        if option_type.lower() == 'call':
            delta_val = norm.cdf(d1)
        else:  # put
            delta_val = norm.cdf(d1) - 1

        return round(delta_val, 4)
    except Exception as e:
        print(f"計算 Delta 失敗: {e}")
        return None

def analyze_unusual_options(ticker, unusual_options):
    """分析異常期權，輸出 Delta、價內/價外狀態與綜合判斷"""
    if not unusual_options:
        return []
    import re
    import yfinance as yf
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    if hist.empty:
        return [{"summary": "無法獲取股價"}]
    current_price = hist['Close'].iloc[-1]

    analysis = []
    for opt in unusual_options:
        match = re.search(r'Call \$([\d.]+)', opt)
        if not match:
            continue
        strike = float(match.group(1))
        # 需要到期日，此處用最近可用的到期日（可改進）
        exps = stock.options
        if not exps:
            continue
        expiration_date = exps[0]  # 取最近到期日
        delta_val = calculate_option_delta(ticker, expiration_date, strike, 'call')
        if delta_val is None:
            continue

        # 判斷價內/價外
        moneyness = "價內" if strike < current_price else "價外" if strike > current_price else "價平"
        # 判斷 Delta 強度
        if delta_val >= 0.7:
            strength = "高敏感（深度價內）"
        elif delta_val >= 0.4:
            strength = "中敏感（價平附近）"
        else:
            strength = "低敏感（深度價外）"
        analysis.append({
            'strike': strike,
            'delta': delta_val,
            'moneyness': moneyness,
            'strength': strength,
            'summary': f"Call ${strike:.1f} → Delta={delta_val:.2f} ({moneyness}，{strength})"
        })
    return analysis

def get_unusual_options(ticker):
    try:
        tk = yf.Ticker(ticker)
        exps = tk.options
        if not exps: return []
        opt = tk.option_chain(exps[0])
        unusual = []
        for _, row in opt.calls.iterrows():
            vol = row['volume'] if not pd.isna(row['volume']) else 0
            oi = row['openInterest'] if not pd.isna(row['openInterest']) else 0
            if oi > 0 and vol / oi > 3:
                unusual.append(f"Call ${row['strike']} 爆量 (Vol/OI={vol/oi:.1f})")
        for _, row in opt.puts.iterrows():
            vol = row['volume'] if not pd.isna(row['volume']) else 0
            oi = row['openInterest'] if not pd.isna(row['openInterest']) else 0
            if oi > 0 and vol / oi > 3:
                unusual.append(f"Put ${row['strike']} 爆量 (Vol/OI={vol/oi:.1f})")
        return unusual[:3]
    except: return []
# ================== 三大法人與主力籌碼數據整合 ==================
def init_institutional_db():
    """初始化三大法人資料庫，支援 SQLite 和 PostgreSQL"""
    with closing(get_db_connection()) as conn:
        cur = conn.cursor()
        if DATABASE_URL:
            # PostgreSQL 語法
            cur.execute("""
                CREATE TABLE IF NOT EXISTS institutional_ownership (
                    id SERIAL PRIMARY KEY,
                    stock_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    foreign_investors INTEGER,
                    investment_trust INTEGER,
                    dealers INTEGER,
                    UNIQUE(stock_id, date)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS large_shareholders (
                    id SERIAL PRIMARY KEY,
                    stock_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    holding_ratio REAL,
                    shareholders_count INTEGER,
                    UNIQUE(stock_id, date)
                )
            """)
        else:
            # SQLite 語法
            cur.execute("""
                CREATE TABLE IF NOT EXISTS institutional_ownership (
                    stock_id TEXT,
                    date TEXT,
                    foreign_investors INTEGER,
                    investment_trust INTEGER,
                    dealers INTEGER,
                    PRIMARY KEY (stock_id, date)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS large_shareholders (
                    stock_id TEXT,
                    date TEXT,
                    holding_ratio REAL,
                    shareholders_count INTEGER,
                    PRIMARY KEY (stock_id, date)
                )
            """)
        conn.commit()
    logger.info("三大法人資料庫初始化完成")

# 啟動時初始化（如果尚未執行，可在此呼叫）
# init_institutional_db()  # 我們稍後會在程式啟動時呼叫一次，此處先註解

def extract_stock_number(ticker):
    """從 '2330.TW' 提取 '2330'"""
    return ticker.replace('.TW', '')

def fetch_institutional_holders(ticker, date=None):
    import requests
    from datetime import datetime, timedelta

    stock_id = extract_stock_number(ticker)
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    else:
        date = datetime.strptime(date, '%Y%m%d').strftime('%Y%m%d')

    for i in range(5):
        check_date = (datetime.strptime(date, '%Y%m%d') - timedelta(days=i)).strftime('%Y%m%d')
        # 使用 T86 全體資料端點，再過濾代號
        url = f"https://www.twse.com.tw/fund/T86?response=json&date={check_date}&selectType=ALLBUT0999"
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
                            logger.info(f"成功抓取 {ticker} 在 {check_date} 的三大法人資料")
                            return foreign, trust, dealer, check_date
                    continue  # 沒有找到該股票代號
                else:
                    continue
        except Exception as e:
            logger.error(f"抓取 {ticker} 在 {check_date} 失敗: {e}")
            continue

    logger.warning(f"{ticker} 連續5天都找不到三大法人資料")
    return None, None, None, None

def fetch_large_shareholders(ticker):
    """從集保結算所抓取千張大戶持股比率（使用新版 API）"""
    import requests
    import ssl
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

    stock_id = extract_stock_number(ticker)
    # 新版 API 網址（集保中心開放資料）
    url = f"https://opendata.tdcc.com.tw/api/opendata/{stock_id}"
    # 若上面仍然 redirect，可以試試用 requests 的 allow_redirects=False 並處理 Location
    try:
        # 禁用自動重定向，手動跟進（但最多一次）
        resp = requests.get(url, timeout=10, verify=False, allow_redirects=False)
        if resp.status_code == 302 or resp.status_code == 301:
            # 獲取重定向後的網址（可能是 https://tdcc.com.tw/...）
            new_url = resp.headers.get('Location')
            if new_url:
                resp = requests.get(new_url, timeout=10, verify=False)
        if resp.status_code == 200:
            data = resp.json()
            if data and 'data' in data and data['data']:
                total_shares = None
                large_shares = 0
                for item in data['data']:
                    range_str = item.get('持股分級', '')
                    shares_str = item.get('股數', '0')
                    try:
                        shares = int(shares_str.replace(',', ''))
                    except:
                        shares = 0
                    if range_str == "合計":
                        total_shares = shares
                    elif any(level in range_str for level in ["1,000", "5,000", "10,000", "50,000", "100,000"]):
                        large_shares += shares
                if total_shares and total_shares > 0:
                    ratio = large_shares / total_shares * 100
                    return ratio
    except Exception as e:
        logger.error(f"抓取 {ticker} 千張大戶資料失敗: {e}")
    return None
def update_large_shareholders_data(ticker):
    """更新千張大戶持股比率到資料庫"""
    ratio = fetch_large_shareholders(ticker)
    if ratio is None:
        logger.warning(f"{ticker} 無千張大戶資料，跳過更新")
        return
    today = datetime.now().strftime('%Y%m%d')
    with closing(sqlite3.connect(INSTITUTIONAL_DB)) as conn:
        conn.execute('''
            INSERT OR REPLACE INTO large_shareholders (stock_id, date, holding_ratio, shareholders_count)
            VALUES (?, ?, ?, ?)
        ''', (ticker, today, ratio, 0))
        conn.commit()
    logger.info(f"更新 {ticker} 千張大戶持股比率: {ratio:.2f}%")
def get_institutional_data(ticker, date=None):
    """取得最新三大法人買賣超，回傳 (外資, 投信, 自營商, 資料日期)"""
    with closing(get_db_connection()) as conn:
        cur = conn.cursor()
        if DATABASE_URL:
            # PostgreSQL 使用 %s 佔位符
            if date is None:
                cur.execute('''
                    SELECT foreign_investors, investment_trust, dealers, date
                    FROM institutional_ownership
                    WHERE stock_id = %s
                    ORDER BY date DESC LIMIT 1
                ''', (ticker,))
            else:
                cur.execute('''
                    SELECT foreign_investors, investment_trust, dealers, date
                    FROM institutional_ownership
                    WHERE stock_id = %s AND date = %s
                ''', (ticker, date))
        else:
            # SQLite 使用 ? 佔位符
            if date is None:
                cur.execute('''
                    SELECT foreign_investors, investment_trust, dealers, date
                    FROM institutional_ownership
                    WHERE stock_id = ?
                    ORDER BY date DESC LIMIT 1
                ''', (ticker,))
            else:
                cur.execute('''
                    SELECT foreign_investors, investment_trust, dealers, date
                    FROM institutional_ownership
                    WHERE stock_id = ? AND date = ?
                ''', (ticker, date))
        row = cur.fetchone()
        if row:
            return row[0], row[1], row[2], row[3]
    return None, None, None, None
def calculate_institutional_score(foreign, trust, dealer):
    """將三大法人買賣超轉換為 0-100 分，買超為正向貢獻"""
    score = 50  # 基準分
    # 外資、投信、自營商各貢獻 0~16.67 分，總和 50 分
    if foreign is not None:
        foreign_abs = abs(foreign)
        if foreign > 0:
            foreign_score = min(16.67, foreign_abs / 1_000_000)
        elif foreign < 0:
            foreign_score = -min(16.67, foreign_abs / 1_000_000)
        else:
            foreign_score = 0
        score += foreign_score
    if trust is not None:
        trust_abs = abs(trust)
        if trust > 0:
            trust_score = min(16.67, trust_abs / 1_000_000)
        elif trust < 0:
            trust_score = -min(16.67, trust_abs / 1_000_000)
        else:
            trust_score = 0
        score += trust_score
    if dealer is not None:
        dealer_abs = abs(dealer)
        if dealer > 0:
            dealer_score = min(16.67, dealer_abs / 1_000_000)
        elif dealer < 0:
            dealer_score = -min(16.67, dealer_abs / 1_000_000)
        else:
            dealer_score = 0
        score += dealer_score
    return max(0, min(100, score))
# ================== 路由 ==================
@app.route('/gex_plot/<ticker>')
def gex_plot(ticker):
    if not PLOTLY_AVAILABLE:
        return "請安裝 plotly: pip install plotly"
    try:
        import urllib.parse
        ticker = urllib.parse.unquote(ticker).upper()
        gex_result = calculate_gamma_exposure(ticker)
        if gex_result is None or gex_result.get('strikes') is None:
            barchart_url = f"https://www.barchart.com/stocks/quotes/{ticker}/options"
            return f"""<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head><body style="background:#121214;color:#fff;padding:20px;"><h3>❌ {ticker} 無期權數據</h3><a href="{barchart_url}" target="_blank">📊 Barchart 期權鏈</a><br><a href="/indicators/{ticker}">🔙 返回</a></body></html>"""
        strikes = gex_result['strikes']
        gex_values = gex_result['gex_values']
        current_price = gex_result.get('current_price')
        call_wall = gex_result.get('call_wall')
        put_wall = gex_result.get('put_wall')
        gamma_flip = gex_result.get('gamma_flip_strike')
        net_gex = gex_result.get('net_total_gex',0)
        total_pos = gex_result.get('total_positive_gex',0)
        total_neg = gex_result.get('total_negative_gex',0)
        cache_time = None
        with _gex_lock:
            if ticker in _gex_cache:
                cache_time = _gex_cache[ticker]['time']
        update_time_str = datetime.fromtimestamp(cache_time).strftime('%Y-%m-%d %H:%M:%S') if cache_time else "剛剛"
        current_price_str = f"${current_price:.2f}" if current_price else "N/A"
        call_wall_str = f"${call_wall:.2f}" if call_wall else "N/A"
        put_wall_str = f"${put_wall:.2f}" if put_wall else "N/A"
        gamma_flip_str = f"${gamma_flip:.2f}" if gamma_flip else "N/A"
        net_gex_str = f"{net_gex:,.0f}"; total_pos_str = f"{total_pos:,.0f}"; total_neg_str = f"{total_neg:,.0f}"
        same_wall_flip = (call_wall and gamma_flip and abs(call_wall - gamma_flip) < 0.01)
        flip_highlight = "⭐️ 關鍵轉折點" if same_wall_flip else ""
        if current_price:
            call_dist_pct = ((call_wall - current_price) / current_price * 100) if call_wall else None
            put_dist_pct = ((put_wall - current_price) / current_price * 100) if put_wall else None
            flip_dist_pct = ((gamma_flip - current_price) / current_price * 100) if gamma_flip else None
        else:
            call_dist_pct = put_dist_pct = flip_dist_pct = None
        bar_colors = ['#00cc66' if v>0 else '#ff4444' for v in gex_values]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=strikes, y=gex_values, name='GEX 數值', marker_color=bar_colors, marker_line_width=1, marker_line_color='black',
                             hovertemplate='履約價: $%{x}<br>GEX: %{y:,.0f}<extra></extra>'))
        cumulative_gex = np.cumsum(gex_values)
        fig.add_trace(go.Scatter(x=strikes, y=cumulative_gex, mode='lines+markers',
                                 name='累積 GEX (Aggregate)', 
                                 line=dict(color='#ffaa00', width=3, dash='solid'),
                                 marker=dict(size=6, color='#ffaa00', symbol='circle'),
                                 hovertemplate='履約價: $%{x}<br>累積 GEX: %{y:,.0f}<extra></extra>'))
        fig.add_hline(y=0, line_dash="solid", line_color="#ffffff", line_width=3,
                      annotation_text="⚖️ 多空分界 (GEX=0)", annotation_position="right",
                      annotation_font_size=11, annotation_font_color="white")
        used_pos = {'bottom': False, 'top': False, 'left': False, 'right': False}
        if current_price:
            fig.add_vline(x=current_price, line_dash="solid", line_color="#ffffff", line_width=2,
                          annotation_text=f"📍 現價 ${current_price:.2f}", annotation_position="bottom")
            used_pos['bottom'] = True
        if call_wall and call_dist_pct is not None:
            if not used_pos['top']:
                pos = "top"
                used_pos['top'] = True
            else:
                pos = "bottom" if not used_pos['bottom'] else "top right"
            fig.add_vline(x=call_wall, line_dash="dashdot", line_color="#00ff00", line_width=2,
                          annotation_text=f"🧱 Call Wall ${call_wall:.2f}<br>(+{call_dist_pct:.2f}%)", annotation_position=pos)
        if put_wall and put_dist_pct is not None:
            if not used_pos['bottom']:
                pos = "bottom"
                used_pos['bottom'] = True
            elif not used_pos['top']:
                pos = "top"
                used_pos['top'] = True
            else:
                pos = "bottom right"
            fig.add_vline(x=put_wall, line_dash="dashdot", line_color="#ff6666", line_width=2,
                          annotation_text=f"🛡️ Put Wall ${put_wall:.2f}<br>({put_dist_pct:.2f}%)", annotation_position=pos)
        if gamma_flip and flip_dist_pct is not None:
            if not used_pos.get('top right', False):
                pos = "top right"
            elif not used_pos.get('bottom right', False):
                pos = "bottom right"
            else:
                pos = "top left"
            anno = f"⚡ Gamma Flip ${gamma_flip:.2f}<br>({flip_dist_pct:.2f}%)"
            if same_wall_flip:
                anno = f"🌟 關鍵轉折點 ${gamma_flip:.2f}<br>({flip_dist_pct:.2f}%) (Call Wall)"
            fig.add_vline(x=gamma_flip, line_dash="dot", line_color="#ffaa00", line_width=2.5,
                          annotation_text=anno, annotation_position=pos)
        
        if gex_values:
            max_abs = max(abs(max(gex_values)), abs(min(gex_values))) * 1.2
        else:
            max_abs = 1
        fig.update_layout(
            title=dict(text=f"📊 {ticker} 期權 Gamma 曝險分析", x=0),
            xaxis_title="履約價",
            yaxis_title="Gamma Exposure (GEX) 單位: USD",
            template='plotly_dark',
            autosize=True,
            height=700,
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='x unified',
            plot_bgcolor='#0e0e12',
            paper_bgcolor='#0e0e12',
            xaxis=dict(tickangle=-45, tickfont=dict(size=10), nticks=15),
            yaxis=dict(tickfont=dict(size=10), range=[-max_abs, max_abs], zeroline=True, zerolinecolor='#ffffff', zerolinewidth=2)
        )
        
        table_rows = ""
        for i, strike in enumerate(strikes):
            gex_val = gex_values[i]
            gex_class = "positive" if gex_val > 0 else "negative" if gex_val < 0 else ""
            gex_str = f"{gex_val:,.0f}"
            table_rows += f"<tr><td style='text-align:center'>{strike:.2f}</td><td class='{gex_class}' style='text-align:right'>{gex_str}</td></tr>"
        detail_table = f"""
        <div class="card" style="margin-top:20px; overflow-x:auto;">
            <h3>📋 GEX 明細表 (每履約價)</h3>
            <table style="width:100%; border-collapse:collapse;">
                <thead><tr><th style="width:50%">履約價 ($)</th><th style="width:50%">Gamma Exposure (USD)</th></tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
        </div>
        """
        dashboard_html = f"""<div style="background:#1e1e1e; border-radius:12px; padding:15px; margin-bottom:20px; display:flex; flex-wrap:wrap; gap:20px; justify-content:space-between;"><div style="display:flex; gap:30px; flex-wrap:wrap;"><div><strong>📌 當前股價</strong><br>{current_price_str}</div><div><strong>🧱 Call Wall</strong><br>{call_wall_str} {flip_highlight}</div><div><strong>🛡️ Put Wall</strong><br>{put_wall_str}</div><div><strong>⚡ Gamma Flip</strong><br>{gamma_flip_str} {flip_highlight}</div><div><strong>📊 淨 GEX</strong><br>{net_gex_str}</div><div><strong>📈 正 GEX (Call)</strong><br>{total_pos_str}</div><div><strong>📉 負 GEX (Put)</strong><br>{total_neg_str}</div></div><div style="color:#aaa;">🕒 {update_time_str}</div></div>"""
        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        full_html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{ticker} Gamma Exposure</title><style>body{{background:#121214;color:#fff;padding:20px; max-width:1400px; margin:0 auto;}} .back-link{{display:inline-block;margin-bottom:20px;background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;}} .positive{{color:#00ff00;}} .negative{{color:#ff4444;}} table{{width:100%; border-collapse:collapse;}} th,td{{padding:8px; text-align:left; border-bottom:1px solid #333;}} th{{background:#2a2a35;}}</style></head><body><a href="/indicators/{ticker}" class="back-link">🔙 返回個股頁面</a>{dashboard_html}{plot_html}{detail_table}</body></html>"""
        return full_html
    except Exception as e:
        logger.exception(f"GEX 繪圖錯誤: {e}")
        return f"<h3>錯誤: {e}</h3><a href='/indicators/{ticker}'>返回個股頁面</a>"   

@app.route("/add_stock_tw", methods=["POST"])
def add_stock_tw():
    tickers = request.form.get("ticker","").strip().upper()
    if not tickers: return redirect(url_for('index'))
    added, new = add_multiple_stocks(tickers, 'tw')
    if added==0: return "⚠️ 所有台股代號都已存在<br><a href='/tw'>返回</a>"
    return f"✅ 新增 {added} 檔台股：{', '.join(new)}<br><a href='/tw'>返回</a>"
@app.route("/add_stock_us", methods=["POST"])
def add_stock_us():
    tickers = request.form.get("ticker","").strip().upper()
    if not tickers: return redirect(url_for('index'))
    added, new = add_multiple_stocks(tickers, 'us')
    if added==0: return "⚠️ 所有美股代號都已存在<br><a href='/us'>返回</a>"
    return f"✅ 新增 {added} 檔美股：{', '.join(new)}<br><a href='/us'>返回</a>"
@app.route("/add_position", methods=["POST"])
def add_position():
    try:
        ticker = request.form.get("pos_ticker","").strip().upper()
        cost = float(request.form.get("cost",0)); shares = int(request.form.get("shares",0))
        if not ticker or cost<=0 or shares<=0: return "❌ 請填寫完整資訊",400
        pos = load_positions()
        found=False
        for p in pos:
            if p['ticker']==ticker: p['cost'],p['shares']=cost,shares; found=True; break
        if not found: pos.append({"ticker":ticker,"cost":cost,"shares":shares})
        save_positions(pos)
        return redirect(url_for('positions_page'))
    except: return "❌ 輸入錯誤",400
@app.route("/add_batch_us", methods=["POST"])
def add_batch_us():
    """批量新增美股（熱門清單）"""
    # 預設熱門美股（可依需求修改）
    default_tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "META", "AMD", "INTC", "QCOM", "AMZN"]
    # 從表單獲取（允許自訂，若無則使用預設）
    tickers_str = request.form.get("tickers", ",".join(default_tickers))
    # 解析多種分隔符（逗號、空格、換行）
    tickers = [t.strip().upper() for t in tickers_str.replace('\n', ',').replace(' ', ',').split(',') if t.strip()]
    added, new = add_multiple_stocks(tickers, 'us')
    if added == 0:
        return "⚠️ 所有美股代號都已存在<br><a href='/us'>返回</a>"
    return f"✅ 成功新增 {added} 檔美股：{', '.join(new)}<br><a href='/us'>返回</a>"

@app.route("/add_batch_tw", methods=["POST"])
def add_batch_tw():
    """批量新增台股（熱門權值股）"""
    default_tickers = ["2330", "2317", "2454", "2308", "2412", "2881", "2882", "2891", "1303", "1326"]
    tickers_str = request.form.get("tickers", ",".join(default_tickers))
    tickers = [t.strip().upper() for t in tickers_str.replace('\n', ',').replace(' ', ',').split(',') if t.strip()]
    # 確保台股代號加上 .TW 後綴（add_multiple_stocks 會自動處理）
    added, new = add_multiple_stocks(tickers, 'tw')
    if added == 0:
        return "⚠️ 所有台股代號都已存在<br><a href='/tw'>返回</a>"
    return f"✅ 成功新增 {added} 檔台股：{', '.join(new)}<br><a href='/tw'>返回</a>"
@app.route("/clear_position/<ticker>")
def clear_position(ticker):
    pos = load_positions()
    save_positions([p for p in pos if p['ticker']!=ticker.upper()])
    return redirect(url_for('positions_page'))
@app.route("/delete/tw/<ticker>")
def delete_tw(ticker):
    with open(STOCK_FILE_TW,'r', encoding='utf-8') as f: lines = f.readlines()
    with open(STOCK_FILE_TW,'w', encoding='utf-8') as f:
        for line in lines:
            if line.strip().upper() != ticker.upper(): f.write(line)
    return redirect(url_for('tw_page'))
@app.route("/delete/us/<ticker>")
def delete_us(ticker):
    with open(STOCK_FILE_US,'r', encoding='utf-8') as f: lines = f.readlines()
    with open(STOCK_FILE_US,'w', encoding='utf-8') as f:
        for line in lines:
            if line.strip().upper() != ticker.upper(): f.write(line)
    return redirect(url_for('us_page'))
@app.route("/retrain_all")
def retrain_all_models():
    tickers = get_all_tickers('tw')+get_all_tickers('us')
    results = []
    for t in tickers:
        try:
            train_single_model_rf(t, force_retrain=True)
            if XGB_AVAILABLE: train_xgboost_model(t, force_retrain=True)
            if LGB_AVAILABLE: train_lightgbm_model(t, force_retrain=True)
            results.append(f"{t}: ✅ 成功")
        except Exception as e: results.append(f"{t}: ❌ {str(e)}")
    with _model_lock_rf: _model_cache_rf.clear()
    with _model_lock_xgb: _model_cache_xgb.clear()
    with _model_lock_lgb: _model_cache_lgb.clear()
    return "<html><body><h1>重新訓練完成</h1><pre>"+"\n".join(results)+"</pre><a href='/'>返回</a></body></html>"
@app.route("/clear_model_cache")
def clear_model_cache(): clear_cache(); return "✅ 所有快取已清除"
@app.route("/force_update")
def force_update():
    threading.Thread(target=update_all_ai_scores, daemon=True).start()
    threading.Thread(target=update_tenbagger, daemon=True).start()
    threading.Thread(target=scan_alerts, daemon=True).start()
    return "✅ 已觸發背景更新，請稍後刷新頁面"
@app.route("/force_update_gex")
def force_update_gex():
    threading.Thread(target=update_all_gex_screener, daemon=True).start()
    return "✅ 已觸發 GEX 篩選器背景更新，請稍後刷新頁面"
@app.route("/force_update_earnings")
def force_update_earnings():
    threading.Thread(target=update_earnings_dates, daemon=True).start()
    return "✅ 已觸發美股財報資料背景更新，請稍後刷新頁面"
@app.route('/gex_screener')
def gex_screener():
    try:
        with open(GEX_SCREENER_FILE, 'r', encoding='utf-8') as f: data = json.load(f)
    except: data = []
    for item in data:
        if "last_update" in item:
            item["last_update_str"] = datetime.fromtimestamp(item["last_update"]).strftime('%Y-%m-%d %H:%M:%S')
        else: item["last_update_str"] = "未知"
    return render_template('gex_screener.html', stocks=data)

@app.route('/battle_map')
def battle_map():
    try:
        with open(GEX_SCREENER_FILE, 'r', encoding='utf-8') as f:
            gex_items = json.load(f); gex_data = {item['ticker']:item for item in gex_items}
    except: gex_data = {}
    try:
        with open(AI_SCORES_FILE, 'r', encoding='utf-8') as f:
            ai_items = json.load(f); ai_data = {item['ticker']:item for item in ai_items if item.get('market')=='us'}
    except: ai_data = {}
    stocks_info = []
    for ticker in get_all_tickers('us'):
        gex = gex_data.get(ticker, {}); ai = ai_data.get(ticker, {})
        curr_price = gex.get('current_price') or ai.get('price')
        if curr_price is None: continue
        call_wall = gex.get('call_wall'); put_wall = gex.get('put_wall'); gamma_flip = gex.get('gamma_flip')
        net_gex = gex.get('net_gex')
        ai_trend = ai.get('trend','N/A')
        ma_trend = "多頭排列" if ai_trend=="多頭" else "空頭排列" if ai_trend=="空頭" else "混亂"
        vwap_status = "站上 VWAP" if ai.get('vwap_status') and "站上" in ai.get('vwap_status','') else "跌破 VWAP" if ai.get('vwap_status') else "N/A"
        advice = generate_tactical_advice(curr_price, call_wall, put_wall, gamma_flip, ma_trend, vwap_status)
        stocks_info.append({"ticker":ticker,"current_price":curr_price,"call_wall":call_wall,"put_wall":put_wall,
                            "gamma_flip":gamma_flip,"net_gex":net_gex,"ai_score":ai.get('ai_score',0),
                            "trend":ai_trend,"advice":advice,"last_update":gex.get('last_update_str','未知')})
    stocks_info.sort(key=lambda x: x.get('net_gex',0) if x.get('net_gex') is not None else 0, reverse=True)
    return render_template('battle_map.html', stocks=stocks_info)

@app.route('/simulator/<ticker>')
def simulator(ticker):
    ticker = ticker.upper()
    try: sim_price = float(request.args.get('price',0))
    except: sim_price = 0
    gex_result = calculate_gamma_exposure(ticker)
    if gex_result is None or gex_result.get('strikes') is None:
        return jsonify({"error": f"{ticker} 無期權數據"})
    current_price = gex_result.get('current_price')
    call_wall = gex_result.get('call_wall'); put_wall = gex_result.get('put_wall'); gamma_flip = gex_result.get('gamma_flip_strike')
    def calc_analysis(price):
        data = {"price": round(price,2)}
        if call_wall:
            data["call_dist_pct"] = round((call_wall - price)/price*100,2)
            data["call_status"] = "壓力區" if data["call_dist_pct"]<1 and data["call_dist_pct"]>0 else "突破壓力" if data["call_dist_pct"]<0 else "距離尚遠"
        else: data["call_dist_pct"] = None; data["call_status"]="N/A"
        if put_wall:
            data["put_dist_pct"] = round((price - put_wall)/price*100,2)
            data["put_status"] = "支撐區" if data["put_dist_pct"]<1 and data["put_dist_pct"]>0 else "跌破支撐" if data["put_dist_pct"]<0 else "距離尚遠"
        else: data["put_dist_pct"] = None; data["put_status"]="N/A"
        if gamma_flip:
            data["flip_dist_pct"] = round((price - gamma_flip)/price*100,2)
            data["flip_status"] = "站上 Gamma Flip (偏多)" if price>gamma_flip else "低於 Gamma Flip (偏空)"
        else: data["flip_dist_pct"] = None; data["flip_status"]="N/A"
        return data
    current = calc_analysis(current_price) if current_price else {}
    simulated = calc_analysis(sim_price) if sim_price>0 else None
    return render_template('simulator.html', ticker=ticker, current=current, simulated=simulated,
                                   current_price=current_price, call_wall=call_wall, put_wall=put_wall, gamma_flip=gamma_flip)

@app.route('/')
def index():
    try: all_scores = json.load(open(AI_SCORES_FILE, 'r', encoding='utf-8'))
    except: all_scores = []
    tw_stocks = [s for s in all_scores if s.get('market')=='tw'][:10]
    us_stocks = [s for s in all_scores if s.get('market')=='us'][:10]
    try: alerts = json.load(open(ALERTS_FILE, 'r', encoding='utf-8'))[:10]
    except: alerts = []
    market_temp, market_status = calculate_market_temperature()
    us_sector = get_us_sector_performance(); tw_sector = get_tw_sector_performance()
    vix = get_vix(); treasury = get_treasury_yield(); dollar = get_dollar_index(); shioaji_status = get_shioaji_status()
    buffett_val, buffett_desc, buffett_url = get_buffett_indicator()
    yield_spread, yield_inverted, yield_text = get_yield_curve_inversion()
    fear_greed_score, fear_greed_level, fear_greed_url = get_cnn_fear_greed()
    margin_balance, margin_ratio, margin_url = get_margin_data()
    return render_template('index.html', tw_stocks=tw_stocks, us_stocks=us_stocks, alerts=alerts,
                                  market_temp=market_temp, market_status=market_status,
                                  us_sector_perf=us_sector, tw_sector_perf=tw_sector,
                                  vix=vix, treasury=treasury, dollar=dollar, shioaji_status=shioaji_status,
                                  buffett_val=buffett_val, buffett_desc=buffett_desc, buffett_url=buffett_url,
                                  yield_spread=yield_spread, yield_inverted=yield_inverted, yield_text=yield_text,
                                  fear_greed_score=fear_greed_score, fear_greed_level=fear_greed_level, fear_greed_url=fear_greed_url,
                                  margin_balance=margin_balance, margin_ratio=margin_ratio, margin_url=margin_url)

@app.route('/tw')
def tw_page():
    tw_curr, tw_change, tw_pct = get_tw_index()
    otc_curr, otc_change, otc_pct = get_otc_index()
    dow_curr, dow_change, dow_pct = get_dow_index()
    nas_curr, nas_change, nas_pct = get_nasdaq_index()
    sox_curr, sox_change, sox_pct = get_phlx_index()
    try:
        all_stocks = json.load(open(AI_SCORES_FILE, 'r', encoding='utf-8'))
        stocks = [s for s in all_stocks if s.get('market')=='tw']
    except: stocks = []
    stocks.sort(key=lambda x: x.get('ai_score',0), reverse=True)
    return render_template('tw_us.html', market='tw', stocks=stocks, title="台股監控 - 所有自選股",
                                  tw_curr=tw_curr, tw_change=tw_change, tw_pct=tw_pct,
                                  otc_curr=otc_curr, otc_change=otc_change, otc_pct=otc_pct,
                                  dow_curr=dow_curr, dow_change=dow_change, dow_pct=dow_pct,
                                  nas_curr=nas_curr, nas_change=nas_change, nas_pct=nas_pct,
                                  sox_curr=sox_curr, sox_change=sox_change, sox_pct=sox_pct,
                                  shioaji_status=get_shioaji_status())

@app.route('/us')
def us_page():
    tw_curr, tw_change, tw_pct = get_tw_index()
    otc_curr, otc_change, otc_pct = get_otc_index()
    dow_curr, dow_change, dow_pct = get_dow_index()
    nas_curr, nas_change, nas_pct = get_nasdaq_index()
    sox_curr, sox_change, sox_pct = get_phlx_index()
    try:
        all_stocks = json.load(open(AI_SCORES_FILE, 'r', encoding='utf-8'))
        stocks = [s for s in all_stocks if s.get('market')=='us']
    except: stocks = []
    stocks.sort(key=lambda x: x.get('ai_score',0), reverse=True)
    return render_template('tw_us.html', market='us', stocks=stocks, title="美股監控 - 所有自選股",
                                  tw_curr=tw_curr, tw_change=tw_change, tw_pct=tw_pct,
                                  otc_curr=otc_curr, otc_change=otc_change, otc_pct=otc_pct,
                                  dow_curr=dow_curr, dow_change=dow_change, dow_pct=dow_pct,
                                  nas_curr=nas_curr, nas_change=nas_change, nas_pct=nas_pct,
                                  sox_curr=sox_curr, sox_change=sox_change, sox_pct=sox_pct,
                                  shioaji_status=get_shioaji_status())

@app.route('/ai_ranking')
def ai_ranking():
    try:
        stocks = json.load(open(AI_SCORES_FILE, 'r', encoding='utf-8'))
        stocks.sort(key=lambda x: x.get('ai_score',0), reverse=True)
    except: stocks = []
    return render_template('ai_ranking.html', stocks=stocks)

@app.route('/tenbagger')
def tenbagger_radar_page():
    try: candidates = json.load(open(TENBAGGER_FILE, 'r', encoding='utf-8'))
    except: candidates = []
    return render_template('tenbagger.html', candidates=candidates)

@app.route('/backtest/<ticker>')
def backtest_page(ticker):
    import urllib.parse
    ticker = urllib.parse.unquote(ticker).upper()
    benchmark = '^GSPC' if '.TW' not in ticker else '^TWII'
    result = run_backtest_advanced(ticker, days=180, benchmark_ticker=benchmark)
    if result is None: return render_template('backtest.html', ticker=ticker, error="資料不足，無法回測")
    return render_template('backtest.html', ticker=ticker, result=result)

@app.route('/positions')
def positions_page():
    pos = load_positions()
    enriched = []
    for p in pos:
        ticker = p['ticker']
        market = 'tw' if '.TW' in ticker else 'us'
        def indicators_page(ticker):
    import urllib.parse
    ticker = urllib.parse.unquote(ticker).upper()
    market = 'tw' if '.TW' in ticker else 'us'
    unusual_opt = []   # 初始化，防止未定義錯誤
    # ... 其餘程式碼 ...
        if market=='tw': curr,_,_,_ = get_tw_stock_data(ticker)
        else: curr,_,_,_ = get_us_stock_data(ticker)
        if curr==0: curr = p['cost']
        value = curr * p['shares']; cost_val = p['cost'] * p['shares']
        profit = value - cost_val; profit_pct = profit/cost_val*100 if cost_val else 0
        stop = p['cost'] * (1 - settings.get('max_loss_per_trade',5)/100)
        enriched.append({'ticker':ticker,'shares':p['shares'],'cost':round(p['cost'],2),
                         'current_price':round(curr,2),'value':round(value,2),'cost_value':round(cost_val,2),
                         'profit':round(profit,2),'profit_pct':round(profit_pct,2),'stop_loss':round(stop,2)})
    return render_template('positions.html', positions=enriched)
@app.route('/indicators/<ticker>')
def indicators_page(ticker):
    import urllib.parse
    ticker = urllib.parse.unquote(ticker).upper()
    market = 'tw' if '.TW' in ticker else 'us'
    if market == 'tw':
        curr, change, pct, df = get_tw_stock_data(ticker)
    else:
        curr, change, pct, df = get_us_stock_data(ticker)
    if df.empty or curr == 0 or math.isnan(curr):
        return render_template('indicators.html', ticker=ticker, error="無法取得有效股價資料")

    close = df['Close'].dropna().values
    if len(close) < 2:
        return render_template('indicators.html', ticker=ticker, error="歷史收盤價不足 (少於2天)")

    if len(close) >= 5:
        ma5 = np.mean(close[-5:])
        ma10 = np.mean(close[-10:]) if len(close) >= 10 else close[-1]
        ma20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
        ma60 = np.mean(close[-60:]) if len(close) >= 60 else close[-1]
        ma120 = np.mean(close[-120:]) if len(close) >= 120 else close[-1]
        ma240 = np.mean(close[-240:]) if len(close) >= 240 else close[-1]
    else:
        ma5 = ma10 = ma20 = ma60 = ma120 = ma240 = curr

    for var in ['ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma240']:
        if np.isnan(locals()[var]):
            locals()[var] = curr

    close_series = safe_get_close(df)
    if len(close_series) >= 15:
        deltas = np.diff(close_series[-15:])
        gain = np.mean(deltas[deltas > 0]) if any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if any(deltas < 0) else 0
        rsi = 100 - 100 / (1 + (gain / loss if loss else 1))
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

    if len(close_series) >= 35:
        _, _, hist, macd_status = calculate_macd(close_series)
    else:
        macd_status, hist = "-", 0

    vwap = calculate_vwap(df)
    vwap_status = get_vwap_status(curr, vwap) if vwap else "N/A"

    def get_ma_trend(ma_name):
        if len(close) < 2:
            return "flat", 0
        if ma_name == 'ma5':
            yesterday = np.mean(close[-6:-1]) if len(close) >= 6 else ma5
            diff = ma5 - yesterday
        elif ma_name == 'ma10':
            yesterday = np.mean(close[-11:-1]) if len(close) >= 11 else ma10
            diff = ma10 - yesterday
        elif ma_name == 'ma20':
            yesterday = np.mean(close[-21:-1]) if len(close) >= 21 else ma20
            diff = ma20 - yesterday
        elif ma_name == 'ma60':
            yesterday = np.mean(close[-61:-1]) if len(close) >= 61 else ma60
            diff = ma60 - yesterday
        elif ma_name == 'ma120':
            yesterday = np.mean(close[-121:-1]) if len(close) >= 121 else ma120
            diff = ma120 - yesterday
        elif ma_name == 'ma240':
            yesterday = np.mean(close[-241:-1]) if len(close) >= 241 else ma240
            diff = ma240 - yesterday
        else:
            diff = 0
        if diff > 0:
            return "up", diff
        elif diff < 0:
            return "down", diff
        else:
            return "flat", 0

    ma5_trend, _ = get_ma_trend('ma5')
    ma10_trend, _ = get_ma_trend('ma10')
    ma20_trend, _ = get_ma_trend('ma20')
    ma60_trend, _ = get_ma_trend('ma60')
    ma120_trend, _ = get_ma_trend('ma120')
    ma240_trend, _ = get_ma_trend('ma240')

    def price_to_ma_ratio(price, ma):
        if ma == 0:
            return 0
        return (price / ma - 1) * 100

    ma5_ratio = price_to_ma_ratio(curr, ma5)
    ma10_ratio = price_to_ma_ratio(curr, ma10)
    ma20_ratio = price_to_ma_ratio(curr, ma20)
    ma60_ratio = price_to_ma_ratio(curr, ma60)
    ma120_ratio = price_to_ma_ratio(curr, ma120)
    ma240_ratio = price_to_ma_ratio(curr, ma240)

    if curr > ma60 and ma60 > ma120 and ma120 > ma240:
        ma_trend = "多頭排列"
    elif curr < ma60 and ma60 < ma120 and ma120 < ma240:
        ma_trend = "空頭排列"
    else:
        ma_trend = "混亂"

    vol = df['Volume'].values
    vol_ma20 = np.mean(vol[-20:]) if len(vol) >= 20 else vol[-1]
    vol_ratio = vol[-1] / vol_ma20 if vol_ma20 > 0 else 1
    sr = calculate_support_resistance(df, curr)
    ai_score = calculate_ai_resonance_score(rsi, macd_status, vwap_status, vol_ratio)

    # AI 模型預測
    rf_pred_val = xgb_pred_val = lgb_pred_val = None
    rf_model, rf_scaler = get_rf_model(ticker)
    if rf_model and rf_scaler:
        feats = calculate_features(df)
        if feats:
            X_pred = np.array(get_feature_vector(feats)).reshape(1, -1)
            X_scaled = rf_scaler.transform(X_pred)
            rf_pred_val = rf_model.predict(X_scaled)[0] * 100
    if XGB_AVAILABLE:
        xgb_model, xgb_scaler = get_xgb_model(ticker)
        if xgb_model and xgb_scaler:
            feats = calculate_features(df)
            if feats:
                X_pred = np.array(get_feature_vector(feats)).reshape(1, -1)
                X_scaled = xgb_scaler.transform(X_pred)
                xgb_pred_val = xgb_model.predict(X_scaled)[0] * 100
    if LGB_AVAILABLE:
        lgb_model, lgb_scaler = get_lgb_model(ticker)
        if lgb_model and lgb_scaler:
            feats = calculate_features(df)
            if feats:
                X_pred = np.array(get_feature_vector(feats)).reshape(1, -1)
                X_scaled = lgb_scaler.transform(X_pred)
                lgb_pred_val = lgb_model.predict(X_scaled)[0] * 100

    ensemble = ensemble_predict(ticker, df)
    options = None
    if market == 'us':
        options = get_options_signals(ticker)
    research_links = get_research_links(ticker)
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
    if vol_ratio > 1.5:
        reasons.append(f"量能放大 {vol_ratio:.1f}倍")
    reasons.append("AI綜合評分")

    gex_data = calculate_gamma_exposure(ticker)
    if gex_data:
        gex_call = gex_data.get('call_wall')
        gex_put = gex_data.get('put_wall')
        gex_flip = gex_data.get('gamma_flip_strike')
    else:
        gex_call = gex_put = gex_flip = None
    # 分析異常期權的 Delta 值
    delta_analysis = analyze_unusual_options(ticker, unusual_opt)

  
                # 取得異常期權（僅美股有）
    if market == 'us':
        try:
            unusual_opt = get_unusual_options(ticker)
        except Exception as e:
            logger.error(f"取得異常期權失敗: {e}")
            unusual_opt = []
    else:
        unusual_opt = []
    
    ai_signal = ensemble[1] if ensemble[1] else None
    tactical_advice = generate_tactical_advice(curr, gex_call, gex_put, gex_flip, ma_trend, vwap_status, ai_signal,
                                               unusual_opt[:2] if unusual_opt else None)
    
    # 分析異常期權的 Delta 值（僅美股）
    if market == 'us' and unusual_opt:
        try:
            delta_analysis = analyze_unusual_options(ticker, unusual_opt)
        except Exception as e:
            logger.error(f"Delta 分析失敗: {e}")
            delta_analysis = []
    else:
        delta_analysis = []

    # 取得三大法人資料（用於顯示）
    foreign, trust, dealer, inst_date = get_institutional_data(ticker)
    if foreign is not None:
        inst_display = f"外資:{foreign:,} 投信:{trust:,} 自營:{dealer:,} (日期:{inst_date})"
        inst_score = calculate_institutional_score(foreign, trust, dealer)
    else:
        inst_display = "無資料"
        inst_score = 50

    return render_template('indicators.html', ticker=ticker, price=round(curr, 2), change=round(change, 2),
                           pct=round(pct, 2), rsi=round(rsi, 1), rsi_status=rsi_status, macd_status=macd_status,
                           macd_hist=round(hist, 3) if hist else 0, vwap_status=vwap_status,
                           ma5=round(ma5, 2), ma10=round(ma10, 2), ma20=round(ma20, 2), ma60=round(ma60, 2),
                           ma120=round(ma120, 2), ma240=round(ma240, 2), ma_trend=ma_trend,
                           volume_ratio=round(vol_ratio, 2), ai_score=ai_score, reasons=reasons,
                           resistance=sr['resistance'], target=sr['target'], stop_loss=sr['stop_loss'],
                           rf_pred=f"{rf_pred_val:+.1f}%" if rf_pred_val else None,
                           xgb_pred=f"{xgb_pred_val:+.1f}%" if xgb_pred_val else None,
                           lgb_pred=f"{lgb_pred_val:+.1f}%" if lgb_pred_val else None,
                           ensemble_score=ensemble[0] if ensemble[0] else None,
                           ensemble_signal=ensemble[1] if ensemble[1] else None,
                           ensemble_details=ensemble[2] if ensemble[2] else {}, agreement=ensemble[3] if ensemble[3] else 0,
                           smart_score=ensemble[4] if ensemble[4] else 50, hype_score=ensemble[5] if ensemble[5] else 50,
                           trend_score=ensemble[6] if ensemble[6] else 50, growth_score=ensemble[7] if ensemble[7] else 50,
                           ten_bagger_score=ensemble[8] if ensemble[8] else 50, consensus_score=ensemble[10] if ensemble[10] else 0,
                           options=options, market=market, research_links=research_links, shioaji_status=get_shioaji_status(),
                           ma5_trend=ma5_trend, ma10_trend=ma10_trend, ma20_trend=ma20_trend, ma60_trend=ma60_trend,
                           ma120_trend=ma120_trend, ma240_trend=ma240_trend,
                           ma5_ratio=ma5_ratio, ma10_ratio=ma10_ratio, ma20_ratio=ma20_ratio, ma60_ratio=ma60_ratio,
                           ma120_ratio=ma120_ratio, ma240_ratio=ma240_ratio,
                           gex_call=gex_call, gex_put=gex_put, gex_flip=gex_flip, tactical_advice=tactical_advice,
                           unusual_options=unusual_opt, delta_analysis=delta_analysis, institutional_data=inst_display, institutional_score=inst_score)
@app.route('/bonding')
def bonding_page():
    threshold = settings.get('bonding_threshold',3.0)/100
    tw_tickers = get_all_tickers('tw'); us_tickers = get_all_tickers('us')
    tw_results = find_ma_bonding_stocks(tw_tickers, threshold=threshold)
    us_results = find_ma_bonding_stocks(us_tickers, threshold=threshold)
    for r in tw_results:
        r['price'] = clean_nan(r.get('price'), 'N/A')
        r['spread'] = clean_nan(r.get('spread'), 'N/A')
        r['ma5'] = clean_nan(r.get('ma5'), 'N/A')
        r['ma20'] = clean_nan(r.get('ma20'), 'N/A')
        r['ma60'] = clean_nan(r.get('ma60'), 'N/A')
    for r in us_results:
        r['price'] = clean_nan(r.get('price'), 'N/A')
        r['spread'] = clean_nan(r.get('spread'), 'N/A')
        r['ma5'] = clean_nan(r.get('ma5'), 'N/A')
        r['ma20'] = clean_nan(r.get('ma20'), 'N/A')
        r['ma60'] = clean_nan(r.get('ma60'), 'N/A')
    return render_template('bonding.html', tw_results=tw_results, us_results=us_results,
                                  tw_count=len(tw_results), us_count=len(us_results),
                                  tw_total=len(tw_tickers), us_total=len(us_tickers),
                                  threshold=settings.get('bonding_threshold',3.0))

@app.route('/settings', methods=['GET','POST'])
def settings_page():
    if request.method == 'POST':
        settings['max_loss_per_trade'] = float(request.form.get('max_loss_per_trade',5))
        settings['max_concentration'] = float(request.form.get('max_concentration',40))
        settings['cash_balance'] = float(request.form.get('cash_balance',0))
        settings['bonding_threshold'] = float(request.form.get('bonding_threshold',3))
        settings['background_update_minutes'] = int(request.form.get('background_update_minutes',15))
        settings['cache_ttl_minutes'] = int(request.form.get('cache_ttl_minutes',5))
        save_settings(settings)
        watchlist = request.form.get('watchlist','').upper()
        if watchlist:
            with open(WATCHLIST_FILE, 'w', encoding='utf-8') as f:
                json.dump([t.strip() for t in watchlist.split(',') if t.strip()], f)
        return redirect(url_for('settings_page'))
    watchlist = []
    try:
        watchlist = json.load(open(WATCHLIST_FILE, 'r', encoding='utf-8'))
    except:
        pass
    return render_template('settings.html', settings=settings, watchlist=','.join(watchlist))

@app.route('/backtest_stats/<ticker>')
def backtest_stats(ticker):
    stats = get_backtest_performance(ticker)
    if stats:
        return jsonify(stats)
    return jsonify({'error':'資料不足'})

@app.route('/earnings_calendar')
def earnings_calendar():
    try:
        data = json.load(open(EARNINGS_CACHE_FILE, 'r', encoding='utf-8'))
    except:
        data = {}
    stocks_info = []
    for ticker in get_all_tickers('us'):
        info = data.get(ticker, {})
        date_str = info.get('date')
        expected_move = info.get('expected_move')
        if expected_move is not None:
            try:
                expected_move = float(expected_move)
            except (ValueError, TypeError):
                expected_move = None
        if date_str:
            try:
                dt = datetime.fromisoformat(date_str)
                days_until = (dt - datetime.now()).days
                if days_until < -1:
                    status = "已過"
                elif days_until == -1:
                    status = "昨日"
                elif days_until == 0:
                    status = "🔥 今日"
                elif days_until <= 3:
                    status = f"⚠️ {days_until} 天後 (高風險)"
                else:
                    status = f"{days_until} 天後"
            except:
                status = "未知"
        else:
            date_str = "N/A"
            status = "無資料"
        stocks_info.append({
            "ticker": ticker,
            "date": date_str if date_str else "N/A",
            "status": status,
            "expected_move": expected_move
        })
    stocks_info.sort(key=lambda x: (x['date'] == "N/A" or x['date'] is None, x['date']))
    return render_template('earnings_calendar.html', stocks=stocks_info)

@app.route('/alert_stream')
def alert_stream():
    def generate():
        global _last_alerts_hash
        try:
            with open(ALERTS_FILE, 'r', encoding='utf-8') as f:
                alerts = json.load(f)
                _last_alerts_hash = hash(json.dumps(alerts, sort_keys=True))
        except:
            _last_alerts_hash = None

        while True:
            try:
                time.sleep(3)
                if not os.path.exists(ALERTS_FILE):
                    continue
                with open(ALERTS_FILE, 'r', encoding='utf-8') as f:
                    new_alerts = json.load(f)
                new_hash = hash(json.dumps(new_alerts, sort_keys=True))
                if new_hash != _last_alerts_hash:
                    _last_alerts_hash = new_hash
                    high_alerts = [a for a in new_alerts if a.get('importance') == 'high'][:3]
                    if high_alerts:
                        msg = " | ".join([f"{a['ticker']}: {a['message']}" for a in high_alerts])
                        yield f"data: 🚨 {msg}\n\n"
                    else:
                        if new_alerts:
                            msg = " | ".join([f"{a['ticker']}: {a['message']}" for a in new_alerts[:3]])
                            yield f"data: 📢 {msg}\n\n"
            except GeneratorExit:
                break
            except Exception as e:
                logger.error(f"SSE 推送错误: {e}")
                time.sleep(5)

    return Response(stream_with_context(generate()), mimetype="text/event-stream")
@app.route('/quote/<ticker>')
def get_quote(ticker):
    """取得指定股票的最新即時報價（從 WebSocket）"""
    stock_id = ticker.upper().replace('.TW', '')
    with quote_lock:
        data = latest_quote.get(stock_id, {})
    if data:
        return jsonify(data)
    else:
        return jsonify({"error": "尚無報價資料"}), 404


@app.route('/institutional')
def institutional_view():
    with closing(get_db_connection()) as conn:
        cur = conn.cursor()
        if DATABASE_URL:
            # PostgreSQL
            cur.execute('''
                SELECT stock_id, date, foreign_investors, investment_trust, dealers
                FROM institutional_ownership
                WHERE date = (SELECT MAX(date) FROM institutional_ownership)
                ORDER BY stock_id
            ''')
            rows = [dict(zip([desc[0] for desc in cur.description], row)) for row in cur.fetchall()]
        else:
            # SQLite
            conn.row_factory = sqlite3.Row
            cur = conn.execute('''
                SELECT stock_id, date, foreign_investors, investment_trust, dealers
                FROM institutional_ownership
                WHERE date = (SELECT MAX(date) FROM institutional_ownership)
                ORDER BY stock_id
            ''')
            rows = cur.fetchall()
    return render_template('institutional.html', data=rows)   
    # 產生 HTML 表格
    html = '''
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"><title>三大法人與大戶持股</title>
    <style>
        body{font-family:sans-serif;background:#121214;color:#fff;padding:20px;}
        table{border-collapse:collapse;width:100%;}
        th,td{border:1px solid #444;padding:8px;text-align:right;}
        th{background:#2a2a35;text-align:center;}
        td:first-child{text-align:center;font-weight:bold;}
        .btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;margin:10px 0;display:inline-block;}
    </style>
    </head>
    <body>
    <a href="/" class="btn">🔙 返回首頁</a>
    <h1>📊 三大法人買賣超 & 千張大戶持股比率（最新日期）</h1>
    <table>
        <thead><tr><th>股票代號</th><th>日期</th><th>外資買賣超</th><th>投信買賣超</th><th>自營商買賣超</th><th>千張大戶持股(%)</th></tr></thead>
        <tbody>
    '''
    for row in rows:
        html += f'''
            <tr>
                <td>{row["stock_id"]}</td>
                <td>{row["date"]}</td>
                <td style="color:{"#ff4444" if row["foreign_investors"]>0 else "#00ff00"}">{row["foreign_investors"]:,}</td>
                <td style="color:{"#ff4444" if row["investment_trust"]>0 else "#00ff00"}">{row["investment_trust"]:,}</td>
                <td style="color:{"#ff4444" if row["dealers"]>0 else "#00ff00"}">{row["dealers"]:,}</td>
                <td>{f"{row['holding_ratio']:.2f}%" if row['holding_ratio'] else "-"}</td>
            </tr>
        '''
    html += '''
        </tbody>
    </table>
    </body>
    </html>
    '''
    return html


def get_crypto_price_ccxt(symbol, exchange='binance'):
    try:
        ex = getattr(ccxt, exchange)()
        ticker = ex.fetch_ticker(symbol.upper())
        return ticker.get('last') or ticker.get('ask')
    except Exception as e:
        print(f"ccxt error: {e}")
        return None



# ================== Shioaji WebSocket 即時報價 ==================
import shioaji as sj
import threading
import time

latest_quote = {}
quote_lock = threading.Lock()

def start_websocket_stream():
    """啟動 Shioaji WebSocket 訂閱即時報價"""
    global latest_quote
    try:
        api = sj.Shioaji(simulation=False)
        api.login(
            api_key=os.getenv('SHIOAJI_API_KEY'),
            secret_key=os.getenv('SHIOAJI_SECRET_KEY')
        )
        api.activate_ca(
            ca_path="Sinopac.pfx",
            ca_passwd=os.getenv('CA_PASSWD', '你的身份證字號'),
            person_id=os.getenv('CA_PASSWD', '你的身份證字號')
        )
        print("Shioaji 已登入並啟用憑證")

        @api.quote.on_tick_stk_v1
        def quote_callback(exchange, tick):
            with quote_lock:
                latest_quote[tick.code] = {
                    'price': float(tick.close),
                    'volume': tick.volume,
                    'total_volume': tick.total_volume,
                    'datetime': str(tick.datetime)
                }

        tickers = get_all_tickers('tw')
        for ticker in tickers:
            stock_id = ticker.replace('.TW', '')
            try:
                contract = api.Contracts.Stocks[stock_id]
                api.quote.subscribe(contract, quote_type=sj.constant.QuoteType.Tick)
                print(f"已訂閱 {ticker}")
            except Exception as e:
                print(f"訂閱 {ticker} 失敗: {e}")

        # 保持執行（WebSocket 在背景執行）
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"WebSocket 啟動失敗: {e}")


# ================== 主程式 ==================
if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print("🚀 啟動 AI 量化監控中心 v16_final")
    print(f"📊 訪問地址：0.0.0.0:{os.environ.get('PORT', 5000)}  (debug={debug_mode})")

    # 初始化三大法人資料庫
    init_institutional_db()

    scheduler = BackgroundScheduler()
    
    scheduler.add_job(
        func=update_all_ai_scores,
        trigger="interval",
        minutes=settings.get('background_update_minutes', 15),
        next_run_time=datetime.now(),
        max_instances=1
    )
    scheduler.add_job(
        func=update_tenbagger,
        trigger="interval",
        minutes=30,
        next_run_time=datetime.now(),
        max_instances=1
    )
    scheduler.add_job(
        func=scan_alerts,
        trigger="interval",
        seconds=60,
        next_run_time=datetime.now(),
        max_instances=1
    )
    scheduler.add_job(
        func=update_all_gex_screener,
        trigger="interval",
        hours=1,
        next_run_time=datetime.now(),
        max_instances=1
    )
    scheduler.add_job(
        func=update_earnings_dates,
        trigger="interval",
        hours=24,
        next_run_time=datetime.now(),
        max_instances=1
    )
    
    # ========== 手動立即測試三大法人更新（僅測試，完成後請註解） ==========
    print("開始手動更新三大法人資料...")
    for t in get_all_tickers('tw'):
        update_institutional_data(t)
        # 千張大戶暫時停用，避免重定向錯誤
        # update_large_shareholders_data(t)
    print("手動更新完成")
    # ========== 手動測試結束 ==========
             
    # 啟動 WebSocket 即時訂閱（獨立執行緒）
    # threading.Thread(target=start_websocket_stream, daemon=True).start()   # 暫時禁用

    scheduler.start()
    
    # 關鍵：從環境變數讀取 PORT，若無則使用 5000；監聽所有外部連線
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug_mode, threaded=True)