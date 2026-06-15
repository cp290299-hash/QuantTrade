# -*- coding: utf-8 -*-
"""
完整功能 v16_final (徹底修正表格標籤、多餘字元、CSS 固定寬度)
"""
import os, sys, json, math, time, logging, random, re, threading
from datetime import datetime, timedelta
from collections import defaultdict
from flask import Flask, render_template_string, request, redirect, url_for, jsonify, Response
from dotenv import load_dotenv
load_dotenv()

import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib, numpy as np, pandas as pd

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

def get_cached_data(ticker, period="2y", ttl_seconds=None):
    if ttl_seconds is None:
        ttl_seconds = settings.get('cache_ttl_minutes',5)*60
    key = f"{ticker}_{period}"
    now = time.time()
    with _cache_lock:
        if key in _data_cache and now - _data_cache[key]['time'] < ttl_seconds:
            df = _data_cache[key]['data']
        else:
            try:
                df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
                if df.empty:
                    df = pd.DataFrame()
                else:
                    df = df.ffill().dropna()
                _data_cache[key] = {'data': df, 'time': now}
            except Exception as e:
                logger.warning(f"取得 {ticker} 資料失敗: {e}")
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
def get_tw_stock_realtime_shioaji(ticker):
    sj_api = get_shioaji()
    if sj_api is None: return None, None, None
    stock_id = ticker.replace('.TW', '').upper()
    try:
        contract = sj_api.Contracts.Stocks[stock_id]
        snapshot = sj_api.snapshot([contract])
        if not snapshot: return None, None, None
        tick = snapshot[0]
        curr = tick.close if tick.close else tick.last_price
        if curr is None: return None, None, None
        prev = tick.reference
        if prev is None or prev == 0: return None, None, None
        change = curr - prev
        pct = (change / prev) * 100
        return curr, change, pct
    except Exception: return None, None, None

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
    if feats is None: return None, None, {}, None, None, None, None, None, None, None, None
    X = np.array(get_feature_vector(feats)).reshape(1,-1)
    scores, details = {}, {}
    rf_model, rf_scaler = get_rf_model(ticker)
    if rf_model and rf_scaler:
        X_scaled = rf_scaler.transform(X)
        rf_pred = rf_model.predict(X_scaled)[0]
        rf_prob = 100 / (1 + math.exp(-rf_pred*10))
        scores['RF'] = rf_prob; details['RF'] = f"預期報酬 {rf_pred*100:+.1f}% → 上漲機率 {rf_prob:.0f}%"
    xgb_model, xgb_scaler = get_xgb_model(ticker)
    if xgb_model and xgb_scaler:
        X_scaled = xgb_scaler.transform(X)
        xgb_pred = xgb_model.predict(X_scaled)[0]
        xgb_prob = 100 / (1 + math.exp(-xgb_pred*10))
        scores['XGB'] = xgb_prob; details['XGB'] = f"預期報酬 {xgb_pred*100:+.1f}% → 上漲機率 {xgb_prob:.0f}%"
    lgb_model, lgb_scaler = get_lgb_model(ticker)
    if lgb_model and lgb_scaler:
        X_scaled = lgb_scaler.transform(X)
        lgb_pred = lgb_model.predict(X_scaled)[0]
        lgb_prob = 100 / (1 + math.exp(-lgb_pred*10))
        scores['LGB'] = lgb_prob; details['LGB'] = f"預期報酬 {lgb_pred*100:+.1f}% → 上漲機率 {lgb_prob:.0f}%"
    smart = get_smart_money_score(ticker); scores['SMART'] = smart; details['SMART'] = f"籌碼 {smart:.0f}"
    hype = 50.0; scores['HYPE'] = hype; details['HYPE'] = f"熱度 {hype:.0f}"
    trend = get_trend_score(df); scores['TREND'] = trend; details['TREND'] = f"趨勢 {trend:.0f}"
    growth, _ = get_growth_score(ticker, df); scores['GROWTH'] = growth; details['GROWTH'] = f"成長 {growth:.0f}"
    if not scores: return None, None, {}, None, None, None, None, None, None, None, None
    weights = {'RF':0.25,'XGB':0.25,'LGB':0.20,'SMART':0.10,'TREND':0.10,'GROWTH':0.10}
    total_w = sum(weights[m] for m in scores if m in weights)
    if total_w == 0: return None, None, {}, None, None, None, None, None, None, None, None
    final = sum(scores[m]*weights[m] for m in scores if m in weights) / total_w
    final = round(final,1)
    if final >= 85: signal = "強力買進"
    elif final >= 75: signal = "買進"
    elif final >= 65: signal = "試單買進"
    elif final >= 55: signal = "持有"
    elif final >= 45: signal = "觀望"
    else: signal = "賣出/避開"
    return final, signal, details, 0.5, smart, hype, trend, growth, get_ten_bagger_score(ticker), [], final

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
        except Exception:
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
        vix = get_vix()
        if vix > 30: score = 20
        elif vix > 25: score = 35
        elif vix > 20: score = 50
        elif vix > 15: score = 65
        else: score = 80
        if score <= 25: level = "極度恐懼"
        elif score <= 45: level = "恐懼"
        elif score <= 55: level = "中性"
        elif score <= 75: level = "貪婪"
        else: level = "極度貪婪"
        return score, level, "https://edition.cnn.com/markets/fear-and-greed"
    except:
        return 50, "中性", "https://edition.cnn.com/markets/fear-and-greed"

def get_margin_data():
    url = "https://www.wantgoo.com/stock/margin-trading/utilization-rate-rank"
    balance = "約 3,200 億"
    maintenance_ratio = "約 160%"
    return balance, maintenance_ratio, url

import math


def clean_nan(value, default='N/A'):
    """將 NaN, None, inf 轉為預設值，浮點數四捨五入至小數2位"""
    if value is None:
        return default
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return default
        return round(value, 2)
    if isinstance(value, dict):
        return {k: clean_nan(v, default) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_nan(v, default) for v in value]
    return value

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

# ================== 路由 ==================
@app.route('/gex_plot/<ticker>')
def gex_plot(ticker):
    if not PLOTLY_AVAILABLE: return "請安裝 plotly: pip install plotly"
    try:
        import urllib.parse
        ticker = urllib.parse.unquote(ticker).upper()
        gex_result = calculate_gamma_exposure(ticker)
        if gex_result is None or gex_result.get('strikes') is None:
            barchart_url = f"https://www.barchart.com/stocks/quotes/{ticker}/options"
            return f"""<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head><body style="background:#121214;color:#fff;padding:20px;"><h3>❌ {ticker} 無期權數據</h3><a href="{barchart_url}" target="_blank">📊 Barchart 期權鏈</a><br><a href="/indicators/{ticker}">🔙 返回</a></body></html>"""
        strikes = gex_result['strikes']; gex_values = gex_result['gex_values']
        current_price = gex_result.get('current_price')
        call_wall = gex_result.get('call_wall'); put_wall = gex_result.get('put_wall'); gamma_flip = gex_result.get('gamma_flip_strike')
        net_gex = gex_result.get('net_total_gex',0); total_pos = gex_result.get('total_positive_gex',0); total_neg = gex_result.get('total_negative_gex',0)
        cache_time = None
        with _gex_lock:
            if ticker in _gex_cache: cache_time = _gex_cache[ticker]['time']
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
        else: call_dist_pct = put_dist_pct = flip_dist_pct = None
        bar_colors = ['#00cc66' if v>0 else '#ff4444' for v in gex_values]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=strikes, y=gex_values, name='GEX 數值', marker_color=bar_colors, marker_line_width=1, marker_line_color='black',
                             hovertemplate='履約價: $%{x}<br>GEX: %{y:,.0f}<extra></extra>'))
        fig.add_hline(y=0, line_dash="solid", line_color="#ffffff", line_width=3,
                      annotation_text="⚖️ 多空分界", annotation_position="right",
                      annotation_font_size=12, annotation_font_color="white")
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
        fig.update_layout(title=dict(text=f"📊 {ticker} 期權 Gamma 曝險分析", x=0), xaxis_title="履約價",
                          yaxis_title="Gamma Exposure (GEX) 單位: USD", template='plotly_dark',
                          autosize=True, height=700, margin=dict(l=50, r=50, t=80, b=50),
                          hovermode='x unified', plot_bgcolor='#0e0e12', paper_bgcolor='#0e0e12')
        
        # 建立 GEX 明細表格
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
        full_html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{ticker} Gamma Exposure</title><style>body{{background:#121214;color:#fff;padding:20px;}} .back-link{{display:inline-block;margin-bottom:20px;background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;}} .positive{{color:#00ff00;}} .negative{{color:#ff4444;}} table{{width:100%; border-collapse:collapse;}} th,td{{padding:8px; text-align:left; border-bottom:1px solid #333;}} th{{background:#2a2a35;}}</style></head><body><a href="/indicators/{ticker}" class="back-link">🔙 返回個股頁面</a>{dashboard_html}{plot_html}{detail_table}</body></html>"""
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
    return render_template_string(GEX_SCREENER_HTML, stocks=data)

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
    return render_template_string(BATTLE_MAP_HTML, stocks=stocks_info)

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
    return render_template_string(SIMULATOR_HTML, ticker=ticker, current=current, simulated=simulated,
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
    return render_template_string(INDEX_HTML, tw_stocks=tw_stocks, us_stocks=us_stocks, alerts=alerts,
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
    return render_template_string(TW_US_HTML, market='tw', stocks=stocks, title="台股監控 - 所有自選股",
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
    return render_template_string(TW_US_HTML, market='us', stocks=stocks, title="美股監控 - 所有自選股",
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
    return render_template_string(AI_RANKING_HTML, stocks=stocks)

@app.route('/tenbagger')
def tenbagger_radar_page():
    try: candidates = json.load(open(TENBAGGER_FILE, 'r', encoding='utf-8'))
    except: candidates = []
    return render_template_string(TENBAGGER_HTML, candidates=candidates)

@app.route('/backtest/<ticker>')
def backtest_page(ticker):
    import urllib.parse
    ticker = urllib.parse.unquote(ticker).upper()
    benchmark = '^GSPC' if '.TW' not in ticker else '^TWII'
    result = run_backtest_advanced(ticker, days=180, benchmark_ticker=benchmark)
    if result is None: return render_template_string(BACKTEST_HTML, ticker=ticker, error="資料不足，無法回測")
    return render_template_string(BACKTEST_HTML, ticker=ticker, result=result)

@app.route('/positions')
def positions_page():
    pos = load_positions()
    enriched = []
    for p in pos:
        ticker = p['ticker']
        market = 'tw' if '.TW' in ticker else 'us'
        if market=='tw': curr,_,_,_ = get_tw_stock_data(ticker)
        else: curr,_,_,_ = get_us_stock_data(ticker)
        if curr==0: curr = p['cost']
        value = curr * p['shares']; cost_val = p['cost'] * p['shares']
        profit = value - cost_val; profit_pct = profit/cost_val*100 if cost_val else 0
        stop = p['cost'] * (1 - settings.get('max_loss_per_trade',5)/100)
        enriched.append({'ticker':ticker,'shares':p['shares'],'cost':round(p['cost'],2),
                         'current_price':round(curr,2),'value':round(value,2),'cost_value':round(cost_val,2),
                         'profit':round(profit,2),'profit_pct':round(profit_pct,2),'stop_loss':round(stop,2)})
    return render_template_string(POSITIONS_HTML, positions=enriched)

@app.route('/indicators/<ticker>')
def indicators_page(ticker):
    import urllib.parse
    ticker = urllib.parse.unquote(ticker).upper()
    market = 'tw' if '.TW' in ticker else 'us'
    if market=='tw': curr, change, pct, df = get_tw_stock_data(ticker)
    else: curr, change, pct, df = get_us_stock_data(ticker)
    if df.empty or curr==0 or math.isnan(curr):
        return render_template_string(INDICATORS_HTML, ticker=ticker, error="無法取得有效股價資料")
    
    # ---------- 均線計算（防 NaN）----------
    close = df['Close'].dropna().values
    if len(close) < 2:
        return render_template_string(INDICATORS_HTML, ticker=ticker, error="歷史收盤價不足 (少於2天)")
    
    if len(close) >= 5:
        ma5 = np.mean(close[-5:])
        ma10 = np.mean(close[-10:]) if len(close) >= 10 else close[-1]
        ma20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
        ma60 = np.mean(close[-60:]) if len(close) >= 60 else close[-1]
        ma120 = np.mean(close[-120:]) if len(close) >= 120 else close[-1]
        ma240 = np.mean(close[-240:]) if len(close) >= 240 else close[-1]
    else:
        ma5 = ma10 = ma20 = ma60 = ma120 = ma240 = curr
    
    # 強制取代任何 NaN
    for var in ['ma5','ma10','ma20','ma60','ma120','ma240']:
        if np.isnan(locals()[var]):
            locals()[var] = curr
    
    # 其餘技術指標計算（與原始相同，略過重複，但確保不會因為 close 長度不足而錯）
    close_series = safe_get_close(df)
    if len(close_series)>=15:
        deltas = np.diff(close_series[-15:])
        gain = np.mean(deltas[deltas>0]) if any(deltas>0) else 0
        loss = -np.mean(deltas[deltas<0]) if any(deltas<0) else 0
        rsi = 100 - 100/(1+ (gain/loss if loss else 1))
    else:
        rsi = 50
    if rsi<30: rsi_status = "🔴 超賣 (可能反彈)"
    elif rsi<50: rsi_status = "🟡 弱勢"
    elif rsi<70: rsi_status = "🟢 強勢"
    else: rsi_status = "⚠️ 超買 (注意回檔)"
    
    if len(close_series)>=35:
        _,_,hist,macd_status = calculate_macd(close_series)
    else:
        macd_status, hist = "-", 0
    
    vwap = calculate_vwap(df)
    vwap_status = get_vwap_status(curr, vwap) if vwap else "N/A"
    
    # 計算均線趨勢與乖離
    def get_ma_trend(ma_name):
        if len(close)<2: return "flat",0
        if ma_name=='ma5':
            yesterday = np.mean(close[-6:-1]) if len(close)>=6 else ma5
            diff = ma5 - yesterday
        elif ma_name=='ma10':
            yesterday = np.mean(close[-11:-1]) if len(close)>=11 else ma10
            diff = ma10 - yesterday
        elif ma_name=='ma20':
            yesterday = np.mean(close[-21:-1]) if len(close)>=21 else ma20
            diff = ma20 - yesterday
        elif ma_name=='ma60':
            yesterday = np.mean(close[-61:-1]) if len(close)>=61 else ma60
            diff = ma60 - yesterday
        elif ma_name=='ma120':
            yesterday = np.mean(close[-121:-1]) if len(close)>=121 else ma120
            diff = ma120 - yesterday
        elif ma_name=='ma240':
            yesterday = np.mean(close[-241:-1]) if len(close)>=241 else ma240
            diff = ma240 - yesterday
        else:
            diff=0
        if diff>0: return "up",diff
        elif diff<0: return "down",diff
        else: return "flat",0
    ma5_trend,_ = get_ma_trend('ma5'); ma10_trend,_ = get_ma_trend('ma10'); ma20_trend,_ = get_ma_trend('ma20')
    ma60_trend,_ = get_ma_trend('ma60'); ma120_trend,_ = get_ma_trend('ma120'); ma240_trend,_ = get_ma_trend('ma240')
    
    def price_to_ma_ratio(price, ma):
        if ma==0: return 0
        return (price/ma -1)*100
    ma5_ratio = price_to_ma_ratio(curr, ma5)
    ma10_ratio = price_to_ma_ratio(curr, ma10)
    ma20_ratio = price_to_ma_ratio(curr, ma20)
    ma60_ratio = price_to_ma_ratio(curr, ma60)
    ma120_ratio = price_to_ma_ratio(curr, ma120)
    ma240_ratio = price_to_ma_ratio(curr, ma240)
    
    if curr>ma60 and ma60>ma120 and ma120>ma240: ma_trend = "多頭排列"
    elif curr<ma60 and ma60<ma120 and ma120<ma240: ma_trend = "空頭排列"
    else: ma_trend = "混亂"
    
    vol = df['Volume'].values
    vol_ma20 = np.mean(vol[-20:]) if len(vol)>=20 else vol[-1]
    vol_ratio = vol[-1]/vol_ma20 if vol_ma20>0 else 1
    sr = calculate_support_resistance(df, curr)
    ai_score = calculate_ai_resonance_score(rsi, macd_status, vwap_status, vol_ratio)
    
    # AI 模型預測
    rf_pred_val = xgb_pred_val = lgb_pred_val = None
    rf_model, rf_scaler = get_rf_model(ticker)
    if rf_model and rf_scaler:
        feats = calculate_features(df)
        if feats:
            X_pred = np.array(get_feature_vector(feats)).reshape(1,-1)
            X_scaled = rf_scaler.transform(X_pred)
            rf_pred_val = rf_model.predict(X_scaled)[0]*100
    if XGB_AVAILABLE:
        xgb_model, xgb_scaler = get_xgb_model(ticker)
        if xgb_model and xgb_scaler:
            feats = calculate_features(df)
            if feats:
                X_pred = np.array(get_feature_vector(feats)).reshape(1,-1)
                X_scaled = xgb_scaler.transform(X_pred)
                xgb_pred_val = xgb_model.predict(X_scaled)[0]*100
    if LGB_AVAILABLE:
        lgb_model, lgb_scaler = get_lgb_model(ticker)
        if lgb_model and lgb_scaler:
            feats = calculate_features(df)
            if feats:
                X_pred = np.array(get_feature_vector(feats)).reshape(1,-1)
                X_scaled = lgb_scaler.transform(X_pred)
                lgb_pred_val = lgb_model.predict(X_scaled)[0]*100
    ensemble = ensemble_predict(ticker, df)
    options = None
    if market == 'us': options = get_options_signals(ticker)
    research_links = get_research_links(ticker)
    reasons = []
    if rsi<30: reasons.append("RSI超賣區")
    elif rsi>70: reasons.append("RSI超買區")
    else: reasons.append("RSI中性")
    if "金叉" in macd_status: reasons.append("MACD黃金交叉")
    elif "死叉" in macd_status: reasons.append("MACD死亡交叉")
    else: reasons.append("MACD平穩")
    if "站上" in vwap_status: reasons.append("站上VWAP")
    else: reasons.append("跌破VWAP")
    if vol_ratio>1.5: reasons.append(f"量能放大 {vol_ratio:.1f}倍")
    reasons.append("AI綜合評分")
    gex_data = calculate_gamma_exposure(ticker)
    if gex_data:
        gex_call = gex_data.get('call_wall'); gex_put = gex_data.get('put_wall'); gex_flip = gex_data.get('gamma_flip_strike')
    else: gex_call = gex_put = gex_flip = None
    unusual_opt = get_unusual_options(ticker)
    ai_signal = ensemble[1] if ensemble[1] else None
    tactical_advice = generate_tactical_advice(curr, gex_call, gex_put, gex_flip, ma_trend, vwap_status, ai_signal, unusual_opt[:2] if unusual_opt else None)
    return render_template_string(INDICATORS_HTML, ticker=ticker, price=round(curr,2), change=round(change,2), pct=round(pct,2),
                                  rsi=round(rsi,1), rsi_status=rsi_status, macd_status=macd_status, macd_hist=round(hist,3) if hist else 0,
                                  vwap_status=vwap_status, ma5=round(ma5,2), ma10=round(ma10,2), ma20=round(ma20,2),
                                  ma60=round(ma60,2), ma120=round(ma120,2), ma240=round(ma240,2), ma_trend=ma_trend,
                                  volume_ratio=round(vol_ratio,2), ai_score=ai_score, reasons=reasons,
                                  resistance=sr['resistance'], target=sr['target'], stop_loss=sr['stop_loss'],
                                  rf_pred=f"{rf_pred_val:+.1f}%" if rf_pred_val else None,
                                  xgb_pred=f"{xgb_pred_val:+.1f}%" if xgb_pred_val else None,
                                  lgb_pred=f"{lgb_pred_val:+.1f}%" if lgb_pred_val else None,
                                  ensemble_score=ensemble[0] if ensemble[0] else None, ensemble_signal=ensemble[1] if ensemble[1] else None,
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
                                  unusual_options=unusual_opt)

@app.route('/bonding')
def bonding_page():
    threshold = settings.get('bonding_threshold',3.0)/100
    tw_tickers = get_all_tickers('tw'); us_tickers = get_all_tickers('us')
    tw_results = find_ma_bonding_stocks(tw_tickers, threshold=threshold)
    us_results = find_ma_bonding_stocks(us_tickers, threshold=threshold)
    return render_template_string(BONDING_HTML, tw_results=tw_results, us_results=us_results,
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
            with open(WATCHLIST_FILE, 'w', encoding='utf-8') as f: json.dump([t.strip() for t in watchlist.split(',') if t.strip()], f)
        return redirect(url_for('settings_page'))
    watchlist = []
    try: watchlist = json.load(open(WATCHLIST_FILE, 'r', encoding='utf-8'))
    except: pass
    return render_template_string(SETTINGS_HTML, settings=settings, watchlist=','.join(watchlist))

@app.route('/backtest_stats/<ticker>')
def backtest_stats(ticker):
    stats = get_backtest_performance(ticker)
    if stats: return jsonify(stats)
    return jsonify({'error':'資料不足'})

@app.route('/earnings_calendar')
def earnings_calendar():
    try: data = json.load(open(EARNINGS_CACHE_FILE, 'r', encoding='utf-8'))
    except: data = {}
    stocks_info = []
    for ticker in get_all_tickers('us'):
        info = data.get(ticker, {})
        date_str = info.get('date')
        expected_move = info.get('expected_move')
        if date_str:
            try:
                dt = datetime.fromisoformat(date_str)
                days_until = (dt - datetime.now()).days
                if days_until < -1: status = "已過"
                elif days_until == -1: status = "昨日"
                elif days_until == 0: status = "🔥 今日"
                elif days_until <= 3: status = f"⚠️ {days_until} 天後 (高風險)"
                else: status = f"{days_until} 天後"
            except: status = "未知"
        else:
            date_str = "N/A"; status = "無資料"
        stocks_info.append({"ticker":ticker,"date":date_str,"status":status,"expected_move":expected_move})
    stocks_info.sort(key=lambda x: (x['date']=="N/A" or x['date'] is None, x['date']))
    return render_template_string(EARNINGS_CALENDAR_HTML, stocks=stocks_info)

@app.route('/earnings_simulator/<ticker>')
def earnings_simulator(ticker):
    ticker = ticker.upper()
    gex_result = calculate_gamma_exposure(ticker)
    if gex_result is None: return jsonify({"error":f"{ticker} 無期權數據"})
    current_price = gex_result.get('current_price')
    call_wall = gex_result.get('call_wall'); put_wall = gex_result.get('put_wall'); gamma_flip = gex_result.get('gamma_flip_strike')
    earnings_info = get_earnings_info(ticker); expected_move = earnings_info.get('expected_move')
    try: pct = float(request.args.get('pct',0))
    except: pct = 0
    sim_price = current_price * (1 + pct/100) if current_price else None
    if sim_price:
        def calc_dist(price, level):
            if level is None or price is None: return None
            return (level - price)/price*100
        call_dist = calc_dist(sim_price, call_wall); put_dist = calc_dist(sim_price, put_wall); flip_dist = calc_dist(sim_price, gamma_flip)
        call_cross = (current_price < call_wall and sim_price > call_wall) or (current_price > call_wall and sim_price < call_wall) if call_wall else None
        put_cross = (current_price > put_wall and sim_price < put_wall) or (current_price < put_wall and sim_price > put_wall) if put_wall else None
        flip_cross = (current_price < gamma_flip and sim_price > gamma_flip) or (current_price > gamma_flip and sim_price < gamma_flip) if gamma_flip else None
        result = {"sim_price":round(sim_price,2),"pct_change":pct,"call_dist":round(call_dist,2) if call_dist else None,
                  "put_dist":round(put_dist,2) if put_dist else None,"flip_dist":round(flip_dist,2) if flip_dist else None,
                  "call_cross":call_cross,"put_cross":put_cross,"flip_cross":flip_cross}
    else: result = None
    return render_template_string(EARNINGS_SIMULATOR_HTML, ticker=ticker, current_price=current_price,
                                  call_wall=call_wall, put_wall=put_wall, gamma_flip=gamma_flip,
                                  expected_move=expected_move, sim_result=result, pct=pct)

@app.route('/alert_stream')
def alert_stream():
    def event_stream():
        last_state = {}
        while True:
            for ticker in get_all_tickers('us'):
                try:
                    curr = get_us_stock_data(ticker)[0]
                    if curr == 0: continue
                    gex = get_cached_gex(ticker)
                    if gex:
                        call = gex.get('call_wall'); put = gex.get('put_wall'); flip = gex.get('gamma_flip_strike')
                        state = (curr > call if call else False, curr > flip if flip else False, curr < put if put else False)
                        if state != last_state.get(ticker):
                            last_state[ticker] = state
                            if call and curr > call: yield f"data: {ticker} 突破 Call Wall ${call:.2f}\n\n"
                            if flip and curr > flip: yield f"data: {ticker} 站上 Gamma Flip ${flip:.2f}\n\n"
                            if put and curr < put: yield f"data: {ticker} 跌破 Put Wall ${put:.2f}\n\n"
                except: pass
            time.sleep(5)
    return Response(event_stream(), mimetype="text/event-stream")

# ================== HTML 模板（徹底修正表格標籤，無多餘字元，CSS 固定寬度）==================
INDEX_HTML = '''<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>AI量化監控 - 儀表板</title>
<style>
    body{background:#121214;color:#fff;font-family:sans-serif;padding:20px;}
    .card{background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:20px;}
    .metric-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:15px;}
    .metric{background:#2a2a35;padding:10px;border-radius:8px;}
    table{width:100%;border-collapse:collapse;margin-top:20px;table-layout:fixed;}
    th,td{padding:8px 6px;text-align:left;border-bottom:1px solid #333;word-break:break-word;}
    th{background:#2a2a35;}
    .positive{color:#ff4444;}.negative{color:#00ff00;}
    .btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;margin:5px;display:inline-block;}
    .stock-link{color:#ffcc00;text-decoration:none;font-weight:bold;background:#333;padding:2px 6px;border-radius:4px;}
    .alert{background:#2a1a1a;border-left:4px solid #ff4444;padding:8px;margin:5px 0;}
    .toast{position:fixed;bottom:20px;right:20px;background:#333;color:#fff;padding:12px 20px;border-radius:8px;z-index:1000;opacity:0;transition:opacity 0.3s;}
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
    <a href="/gex_screener" class="btn">📊 GEX 篩選器</a>
    <a href="/battle_map" class="btn">🗺️ 盤前戰術板</a>
    <a href="/earnings_calendar" class="btn">📅 財報日曆</a>
    <a href="/settings" class="btn">⚙️ 設定</a>
    <a href="/retrain_all" class="btn" style="background:#ff9800;">🔄 重新訓練所有模型</a>
    <a href="/clear_model_cache" class="btn" style="background:#ff9800;">🗑️ 清除快取</a>
    <a href="/force_update" class="btn" style="background:#2196f3;">🔄 強制更新</a>
</div>
<div class="card">
    <h2>📊 市場溫度計</h2>
    <div>市場綜合評分: <strong>{{ market_temp }}/100</strong> - {{ market_status }}</div>
    <div>VIX: {{ "%.2f"|format(vix) }} | 美債: {{ "%.2f"|format(treasury) }}% | 美元: {{ "%.2f"|format(dollar) }}</div>
    <div>🇹🇼 台股報價: {{ shioaji_status }}</div>
</div>
<div class="card">
    <h2>📉 崩盤風險監控</h2>
    <div class="metric-grid">
        <div class="metric"><strong>📊 巴菲特指標</strong><br>{% if buffett_val %}{{ "%.1f"|format(buffett_val) }}%{% else %}{{ buffett_desc }}{% endif %}<br><a href="{{ buffett_url }}" target="_blank">詳細</a></div>
        <div class="metric"><strong>😨 VIX</strong><br>{{ "%.2f"|format(vix) }}<br>{% if vix > 30 %}🔴極度恐慌{% elif vix > 20 %}🟡恐慌升溫{% else %}🟢平靜{% endif %}</div>
        <div class="metric"><strong>😱 恐懼貪婪</strong><br>{{ fear_greed_score }}/100 - {{ fear_greed_level }}<br><a href="{{ fear_greed_url }}" target="_blank">走勢</a></div>
        <div class="metric"><strong>📉 殖利率曲線</strong><br>{% if yield_spread %}{{ "%.2f"|format(yield_spread) }}%<br>{{ yield_text }}{% else %}N/A{% endif %}</div>
        <div class="metric"><strong>💰 融資餘額</strong><br>{{ margin_balance }}<br>{{ margin_ratio }}<br><a href="{{ margin_url }}" target="_blank">查詢</a></div>
    </div>
</div>
<div class="card">
    <h2>🏭 產業輪動</h2>
    美股: {% for n,r in us_sector_perf %}{{ n }}:{{ r }}% {% endfor %}<br>
    台股: {% for n,r in tw_sector_perf %}{{ n }}:{{ r }}% {% endfor %}
</div>
<div class="card">
    <h2>🚨 警報</h2>
    {% for a in alerts %}<div class="alert">🚨 {{ a.ticker }} - {{ a.message }}</div>{% else %}無{% endfor %}
</div>

<!-- 台股 AI 前10 表格 -->
<div class="card">
    <h2>🇹🇼 台股 AI前10</h2>
    <table>
        <thead>
            <tr><th>代號</th><th>價格</th><th>漲跌</th><th>AI分數</th><th>訊號</th></tr>
        </thead>
        <tbody>
            {% for s in tw_stocks %}
            <tr>
                <td><a href="/indicators/{{ s.ticker }}" class="stock-link">{{ s.ticker }}</a></td>
                <td>{{ s.price }}</td>
                <td class="{{ 'positive' if s.pct>=0 else 'negative' }}">{{ s.pct }}%</td>
                <td style="color: {{ '#00ff00' if s.ai_score>=70 else '#ffcc00' if s.ai_score>=50 else '#ff4444' }}">{{ s.ai_score }}</td>
                <td>{{ s.signal }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>

<!-- 美股 AI 前10 表格 -->
<div class="card">
    <h2>🇺🇸 美股 AI前10</h2>
    <table>
        <thead>
            <tr><th>代號</th><th>價格</th><th>漲跌</th><th>AI分數</th><th>訊號</th></tr>
        </thead>
        <tbody>
            {% for s in us_stocks %}
            <tr>
                <td><a href="/indicators/{{ s.ticker }}" class="stock-link">{{ s.ticker }}</a></td>
                <td>{{ s.price }}</td>
                <td class="{{ 'positive' if s.pct>=0 else 'negative' }}">{{ s.pct }}%</td>
                <td style="color: {{ '#00ff00' if s.ai_score>=70 else '#ffcc00' if s.ai_score>=50 else '#ff4444' }}">{{ s.ai_score }}</td>
                <td>{{ s.signal }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>

<div id="toast" class="toast"></div>
<script>
if(typeof(EventSource)!=='undefined'){
    var source=new EventSource('/alert_stream');
    source.onmessage=function(event){
        var toast=document.getElementById('toast');
        toast.innerHTML=event.data;
        toast.style.opacity=1;
        setTimeout(function(){toast.style.opacity=0;},5000);
    };
}
</script>
</body>
</html>'''

TW_US_HTML = '''<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>{{ title }}</title>
<style>
    body{background:#121214;color:#fff;font-family:sans-serif;padding:20px;}
    .card{background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:20px;}
    .index-grid{display:flex;gap:15px;flex-wrap:wrap;margin-bottom:20px;}
    .index-item{background:#2a2a35;padding:8px 15px;border-radius:8px;}
    .positive{color:#ff4444;}.negative{color:#00ff00;}
    table{width:100%;border-collapse:collapse;table-layout:fixed;}
    th,td{padding:10px 8px;text-align:left;border-bottom:1px solid #333;word-break:break-word;}
    th{background:#2a2a35;}
    .btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;margin:5px;display:inline-block;}
    .stock-link{color:#ffcc00;text-decoration:none;font-weight:bold;background:#333;padding:2px 6px;border-radius:4px;}
</style>
</head>
<body>
<h1>{{ title }}</h1>
<div><a href="/" class="btn">返回儀表板</a> <a href="/ai_ranking" class="btn">🏆 AI排行榜</a></div>
<div class="card">
    <div class="index-grid">
        <div class="index-item">📊 加權: {{ tw_curr }} <span class="{{ 'positive' if tw_pct>=0 else 'negative' }}">{{ tw_change }}點, {{ tw_pct }}%</span></div>
        <div class="index-item">📊 櫃買: {{ otc_curr }} <span class="{{ 'positive' if otc_pct>=0 else 'negative' }}">{{ otc_change }}點, {{ otc_pct }}%</span></div>
        <div class="index-item">📊 道瓊: {{ dow_curr }} <span class="{{ 'positive' if dow_pct>=0 else 'negative' }}">{{ dow_change }}點, {{ dow_pct }}%</span></div>
        <div class="index-item">📊 那斯達克: {{ nas_curr }} <span class="{{ 'positive' if nas_pct>=0 else 'negative' }}">{{ nas_change }}點, {{ nas_pct }}%</span></div>
        <div class="index-item">📊 費半: {{ sox_curr }} <span class="{{ 'positive' if sox_pct>=0 else 'negative' }}">{{ sox_change }}點, {{ sox_pct }}%</span></div>
    </div>
</div>
<div class="card">
    <form method="POST" action="/add_stock_{{ market }}">
        <input name="ticker" placeholder="股票代號" required>
        <button type="submit" class="btn">➕ 加入自選</button>
    </form>
</div>
<div class="card">
    <table>
        <thead>
            <tr>
                <th>代號</th><th>價格</th><th>漲跌</th>
                <th>類型</th><th>AI分數</th><th>趨勢</th>
                <th>訊號</th><th>操作</th>
            </tr>
        </thead>
        <tbody>
            {% for s in stocks %}
            <tr>
                <td><a href="/indicators/{{ s.ticker }}" class="stock-link">{{ s.ticker }}</a></td>
                <td>{{ s.price }}</td>
                <td class="{{ 'positive' if s.pct>=0 else 'negative' }}">{{ s.pct }}%</td>
                <td>{% if market == 'tw' %}{% if shioaji_status == '🟢 即時模式' %}🟢 即時{% else %}🔴 延遲{% endif %}{% else %}🔵 延遲 15分{% endif %}</td>
                <td>{{ s.ai_score }}</td>
                <td>{{ s.trend }}</td>
                <td>{{ s.signal }}</td>
                <td><a href="/delete/{{ market }}/{{ s.ticker }}" class="btn" style="background:#ff4444;">刪除</a></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
</body>
</html>'''
AI_RANKING_HTML = '''<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>AI選股排行榜</title>
<style>
    body{background:#121214;color:#fff;font-family:sans-serif;padding:20px;}
    table{width:100%;border-collapse:collapse;table-layout:fixed;}
    th,td{padding:10px;text-align:left;border-bottom:1px solid #333;word-break:break-word;}
    th{background:#2a2a35;}
    .btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;margin:5px;}
    .stock-link{color:#ffcc00;text-decoration:none;font-weight:bold;background:#333;padding:2px 6px;border-radius:4px;}
</style>
</head>
<body>
<h1>🏆 AI 共振評分排行榜</h1>
<a href="/" class="btn">返回首頁</a>
<div class="card">
    <table>
        <thead>
            <tr><th>排名</th><th>代號</th><th>市場</th><th>價格</th><th>AI分數</th><th>趨勢</th><th>買賣訊號</th></tr>
        </thead>
        <tbody>
            {% for s in stocks %}
            <tr>
                <td>{{ loop.index }}</td>
                <td><a href="/indicators/{{ s.ticker }}" class="stock-link">{{ s.ticker }}</a></td>
                <td>{{ '台股' if s.market=='tw' else '美股' }}</td>
                <td>{{ s.price }}</td>
                <td style="color: {{ '#00ff00' if s.ai_score>=70 else '#ffcc00' if s.ai_score>=50 else '#ff4444' }}">{{ s.ai_score }}/100</td>
                <td>{{ s.trend }}</td>
                <td>{{ s.signal }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
</body>
</html>'''

POSITIONS_HTML = '''<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>持倉中心</title>
<style>
    body{background:#121214;color:#fff;font-family:sans-serif;padding:20px;}
    .card{background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:20px;}
    table{width:100%;border-collapse:collapse;table-layout:fixed;}
    th,td{padding:8px;text-align:left;border-bottom:1px solid #333;word-break:break-word;}
    th{background:#2a2a35;}
    .positive{color:#00ff00;}.negative{color:#ff4444;}
    .btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;}
    .stock-link{color:#ffcc00;text-decoration:none;font-weight:bold;background:#333;padding:2px 6px;border-radius:4px;}
</style>
</head>
<body>
<h1>📊 持倉管理</h1>
<a href="/" class="btn">首頁</a>
<div class="card">
    <form method="POST" action="/add_position">
        <input name="pos_ticker" placeholder="代號" required>
        <input name="cost" placeholder="成本" step="0.01" required>
        <input name="shares" placeholder="股數" required>
        <button type="submit" class="btn">新增/更新</button>
    </form>
</div>
<div class="card">
    <table>
        <thead>
            <tr><th>代號</th><th>股數</th><th>成本</th><th>現價</th><th>市值</th><th>損益</th><th>報酬率</th><th>停損價</th><th>操作</th></tr>
        </thead>
        <tbody>
            {% for p in positions %}
            <tr>
                <td><a href="/indicators/{{ p.ticker }}" class="stock-link">{{ p.ticker }}</a></td>
                <td>{{ p.shares }}</td>
                <td>{{ p.cost }}</td>
                <td>{{ p.current_price }}</td>
                <td>{{ p.value }}</td>
                <td class="{{ 'positive' if p.profit>=0 else 'negative' }}">{{ p.profit }}</td>
                <td>{{ p.profit_pct }}%</td>
                <td>{{ p.stop_loss }}</td>
                <td><a href="/clear_position/{{ p.ticker }}" class="btn" style="background:#ff4444;">歸零</a></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
</body>
</html>'''
INDICATORS_HTML = '''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>{{ ticker }} 技術指標</title><style>body{background:#121214;color:#fff;padding:20px;font-family:sans-serif;}.card{background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:20px;}.metric-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:15px;}.metric{background:#2a2a35;padding:10px;border-radius:8px;}.btn{background:#2a2a35;padding:8px 14px;border-radius:20px;color:white;text-decoration:none;display:inline-block;margin:5px;}.btn-green{background:#2e7d32;}.btn-blue{background:#1565c0;}.btn-orange{background:#e65100;}.btn-red{background:#c62828;}.link-buttons{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;}.ma-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;}.ma-metric{position:relative;overflow:hidden;transition:transform 0.2s;}.ma-metric:hover{transform:translateY(-2px);background:#353540;}.ma-label{font-size:12px;color:#aaa;margin-bottom:5px;display:flex;justify-content:space-between;}.ma-value{font-size:18px;font-weight:bold;display:flex;align-items:center;gap:6px;margin-bottom:4px;}.trend-up{color:#ff4444;font-size:14px;}.trend-down{color:#00ff00;font-size:14px;}.trend-flat{color:#ffaa00;font-size:12px;}.progress-bar-container{background-color:#333;border-radius:10px;height:6px;margin:6px 0 4px;overflow:hidden;}.progress-bar-fill{height:100%;border-radius:10px;transition:width 0.3s;}.ratio-text{font-size:10px;text-align:right;color:#ccc;}.badge{padding:4px 12px;border-radius:20px;font-size:14px;font-weight:bold;display:inline-flex;align-items:center;gap:6px;}.badge.bull{background-color:rgba(255,68,68,0.2);color:#ff4444;border:1px solid #ff4444;}.badge.bear{background-color:rgba(0,255,0,0.2);color:#00ff00;border:1px solid #00ff00;}.badge.neutral{background-color:rgba(255,170,0,0.2);color:#ffaa00;border:1px solid #ffaa00;}.pulse{display:inline-block;width:10px;height:10px;border-radius:50%;background-color:currentColor;animation:pulse 1.5s infinite;}@keyframes pulse{0%{opacity:0.4;transform:scale(0.8);}70%{opacity:1;transform:scale(1.2);}100%{opacity:0.4;transform:scale(0.8);}}.tooltip-hover{cursor:help;border-bottom:1px dotted #888;}.alert-box{background:#2a2a35;border-left:4px solid #ffaa00;padding:10px;margin-bottom:15px;border-radius:8px;}.unusual-box{background:#1e2a1e;border-left:4px solid #ffaa00;padding:8px;margin-top:10px;font-size:12px;}</style></head><body><h1>📈 {{ ticker }} 技術指標</h1><div><a href="/" class="btn">返回首頁</a> <a href="/backtest/{{ ticker }}" class="btn">📊 回測</a> <a href="/simulator/{{ ticker }}" class="btn btn-orange">🎯 情境模擬器</a> <a href="/earnings_simulator/{{ ticker }}" class="btn btn-blue">📊 財報模擬器</a></div><div class="link-buttons"><a href="/gex_plot/{{ ticker }}" target="_blank" class="btn btn-green">📊 本地 GEX</a>{% if market == 'us' %}<a href="{{ research_links.MarketWatch }}" target="_blank" class="btn btn-blue">📈 MarketWatch</a><a href="{{ research_links.Barchart }}" target="_blank" class="btn btn-orange">📊 Barchart 期權</a><a href="{{ research_links.Finviz }}" target="_blank" class="btn">🔍 Finviz</a><a href="{{ research_links.Reddit }}" target="_blank" class="btn btn-red">💬 Reddit</a>{% else %}<a href="{{ research_links.WantGoo }}" target="_blank" class="btn btn-blue">🇹🇼 玩股網</a><a href="{{ research_links.GoodInfo }}" target="_blank" class="btn btn-orange">📊 GoodInfo</a><a href="{{ research_links.Google }}" target="_blank" class="btn">🔍 Google 新聞</a>{% endif %}<a href="{{ research_links.TradingView }}" target="_blank" class="btn btn-blue">📈 TradingView</a><a href="{{ research_links.Yahoo }}" target="_blank" class="btn">📰 Yahoo 財經</a></div>{% if error %}<div class="card">{{ error }}</div>{% else %}<div class="card"><h3>即時報價</h3><div class="metric-grid"><div class="metric">價格: {{ price }}</div><div class="metric">漲跌: {{ change }} ({{ pct }}%)</div><div class="metric">AI評分: {{ ai_score }}/100</div><div class="metric">報價類型: {% if market == 'tw' %}{% if shioaji_status == '🟢 即時模式' %}🟢 即時 (Shioaji){% else %}🔴 延遲 (Yahoo){% endif %}{% else %}🔵 延遲 15分鐘 (Yahoo){% endif %}</div></div></div><div class="card"><h3>⚔️ 戰術建議</h3><div class="alert-box">{{ tactical_advice }}</div>{% if unusual_options %}<div class="unusual-box">📢 異常期權活動: {{ unusual_options|join(' | ') }}</div>{% endif %}</div><div class="card"><h3>核心指標</h3><div class="metric-grid"><div class="metric">RSI: {{ rsi }} {{ rsi_status }}</div><div class="metric">MACD: {{ macd_status }}</div><div class="metric">VWAP: {{ vwap_status }}</div><div class="metric">量比: {{ volume_ratio }}</div></div></div><div class="card"><h3>🤖 AI預測</h3><div class="metric-grid"><div class="metric"><strong>RandomForest</strong><br>預期漲跌: {{ rf_pred if rf_pred else 'N/A' }}</div><div class="metric"><strong>XGBoost</strong><br>預期漲跌: {{ xgb_pred if xgb_pred else 'N/A' }}</div><div class="metric"><strong>LightGBM</strong><br>預期漲跌: {{ lgb_pred if lgb_pred else 'N/A' }}</div><div class="metric"><strong>綜合評分</strong><br>{% if ensemble_score %}{{ ensemble_score }}/100<br><strong>{{ ensemble_signal }}</strong>{% else %}訓練中...{% endif %}</div></div></div><div class="card"><h3>📊 因子</h3><div class="metric-grid"><div class="metric">💰 籌碼: {{ smart_score }}</div><div class="metric">📈 趨勢: {{ trend_score }}</div><div class="metric">📈 成長: {{ growth_score }}</div><div class="metric">🚀 十倍股: {{ ten_bagger_score }}</div></div></div><div class="card"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;"><h3 style="margin:0;" class="tooltip-hover" title="均線趨勢與乖離率分析。紅色箭頭=均線上漲；進度條紅色=股價站上均線。">📈 均線分析</h3><div class="badge {% if ma_trend == '多頭排列' %}bull{% elif ma_trend == '空頭排列' %}bear{% else %}neutral{% endif %}" title="{% if ma_trend == '多頭排列' %}短期均線 > 中期 > 長期，技術強勢{% elif ma_trend == '空頭排列' %}長期均線壓制，技術弱勢{% else %}均線交錯，方向不明{% endif %}"><span class="pulse"></span>{% if ma_trend == '多頭排列' %}🐂 多頭排列{% elif ma_trend == '空頭排列' %}🐻 空頭排列{% else %}⚖️ 混亂{% endif %}</div></div><div class="ma-grid"><div class="metric ma-metric"><div class="ma-label">5MA</div><div class="ma-value">{{ "%.2f"|format(ma5) }}{% if ma5_trend == 'up' %}<span class="trend-up">▲</span>{% elif ma5_trend == 'down' %}<span class="trend-down">▼</span>{% else %}<span class="trend-flat">●</span>{% endif %}</div><div class="progress-bar-container"><div class="progress-bar-fill" style="width:{{ 50+ma5_ratio }}%;background-color:{% if ma5_ratio>0 %}#ff4444{% else %}#00ff00{% endif %};"></div></div><div class="ratio-text">{{ "%.1f"|format(ma5_ratio) }}%</div></div><div class="metric ma-metric"><div class="ma-label">10MA</div><div class="ma-value">{{ "%.2f"|format(ma10) }}{% if ma10_trend == 'up' %}<span class="trend-up">▲</span>{% elif ma10_trend == 'down' %}<span class="trend-down">▼</span>{% else %}<span class="trend-flat">●</span>{% endif %}</div><div class="progress-bar-container"><div class="progress-bar-fill" style="width:{{ 50+ma10_ratio }}%;background-color:{% if ma10_ratio>0 %}#ff4444{% else %}#00ff00{% endif %};"></div></div><div class="ratio-text">{{ "%.1f"|format(ma10_ratio) }}%</div></div><div class="metric ma-metric"><div class="ma-label">20MA</div><div class="ma-value">{{ "%.2f"|format(ma20) }}{% if ma20_trend == 'up' %}<span class="trend-up">▲</span>{% elif ma20_trend == 'down' %}<span class="trend-down">▼</span>{% else %}<span class="trend-flat">●</span>{% endif %}</div><div class="progress-bar-container"><div class="progress-bar-fill" style="width:{{ 50+ma20_ratio }}%;background-color:{% if ma20_ratio>0 %}#ff4444{% else %}#00ff00{% endif %};"></div></div><div class="ratio-text">{{ "%.1f"|format(ma20_ratio) }}%</div></div><div class="metric ma-metric"><div class="ma-label">60MA</div><div class="ma-value">{{ "%.2f"|format(ma60) }}{% if ma60_trend == 'up' %}<span class="trend-up">▲</span>{% elif ma60_trend == 'down' %}<span class="trend-down">▼</span>{% else %}<span class="trend-flat">●</span>{% endif %}</div><div class="progress-bar-container"><div class="progress-bar-fill" style="width:{{ 50+ma60_ratio }}%;background-color:{% if ma60_ratio>0 %}#ff4444{% else %}#00ff00{% endif %};"></div></div><div class="ratio-text">{{ "%.1f"|format(ma60_ratio) }}%</div></div><div class="metric ma-metric"><div class="ma-label">120MA</div><div class="ma-value">{{ "%.2f"|format(ma120) }}{% if ma120_trend == 'up' %}<span class="trend-up">▲</span>{% elif ma120_trend == 'down' %}<span class="trend-down">▼</span>{% else %}<span class="trend-flat">●</span>{% endif %}</div><div class="progress-bar-container"><div class="progress-bar-fill" style="width:{{ 50+ma120_ratio }}%;background-color:{% if ma120_ratio>0 %}#ff4444{% else %}#00ff00{% endif %};"></div></div><div class="ratio-text">{{ "%.1f"|format(ma120_ratio) }}%</div></div><div class="metric ma-metric"><div class="ma-label">240MA</div><div class="ma-value">{{ "%.2f"|format(ma240) }}{% if ma240_trend == 'up' %}<span class="trend-up">▲</span>{% elif ma240_trend == 'down' %}<span class="trend-down">▼</span>{% else %}<span class="trend-flat">●</span>{% endif %}</div><div class="progress-bar-container"><div class="progress-bar-fill" style="width:{{ 50+ma240_ratio }}%;background-color:{% if ma240_ratio>0 %}#ff4444{% else %}#00ff00{% endif %};"></div></div><div class="ratio-text">{{ "%.1f"|format(ma240_ratio) }}%</div></div></div></div><div class="card"><h3>壓力支撐</h3><div class="metric-grid"><div class="metric">🔴 壓力: {{ resistance }}</div><div class="metric">🟢 目標: {{ target }}</div><div class="metric">⚪ 停損: {{ stop_loss }}</div></div></div><div class="card"><h3>📌 價格警戒線 (GEX)</h3><div class="metric-grid"><div class="metric">🧱 Call Wall: {{ gex_call if gex_call else 'N/A' }}</div><div class="metric">🛡️ Put Wall: {{ gex_put if gex_put else 'N/A' }}</div><div class="metric">⚡ Gamma Flip: {{ gex_flip if gex_flip else 'N/A' }}</div></div></div>{% if options %}<div class="card"><h3>📊 期權信號</h3><div class="metric-grid">{% if options.pcr %}<div class="metric">PCR: {{ options.pcr.pcr }} {{ options.pcr.text }}</div>{% endif %}{% if options.max_pain %}<div class="metric">Max Pain: {{ options.max_pain.max_pain }}<br>{{ options.max_pain.signal }}</div>{% endif %}{% if options.iv %}<div class="metric">IV Rank: {{ options.iv.iv }}%<br>{{ options.iv.text }}</div>{% endif %}{% if options.gex %}<div class="metric">Call Wall: {{ options.gex.call_wall }}<br>Put Wall: {{ options.gex.put_wall }}<br>Gamma Flip: {{ options.gex.gamma_flip_strike }}</div>{% endif %}</div><div class="metric" style="text-align:center;"><strong style="color:{{ options.composite.color }};">{{ options.composite.text }}</strong><br>{{ options.composite.reasons|join(' ｜ ') }}</div></div>{% endif %}<div class="card"><h3>📖 評分解釋</h3><ul>{% for r in reasons %}<li>{{ r }}</li>{% endfor %}</ul></div><div class="card"><h3>📊 回測績效</h3><div class="metric-grid" id="backtest_stats"><div class="metric">載入中...</div></div></div>{% endif %}<script>fetch('/backtest_stats/{{ ticker }}').then(r=>r.json()).then(d=>{if(d.error)document.getElementById('backtest_stats').innerHTML='<div class="metric">無資料</div>';else document.getElementById('backtest_stats').innerHTML=`<div class="metric">總訊號: ${d.total_signals}次</div><div class="metric">勝率: ${d.win_rate}%</div><div class="metric">平均獲利: +${d.avg_profit}%</div><div class="metric">平均虧損: ${d.avg_loss}%</div><div class="metric">Profit Factor: ${d.profit_factor}</div>`;})</script></body></html>'''

BACKTEST_HTML = '''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>回測報告 - {{ ticker }}</title><style>body{background:#121214;color:#fff;padding:20px;font-family:sans-serif;}.card{background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:20px;}.metric-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:15px;}.metric{background:#2a2a35;padding:10px;border-radius:8px;}.btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;display:inline-block;margin:5px;}</style></head><body><h1>📊 回測報告 - {{ ticker }}</h1><a href="/" class="btn">返回儀表板</a> <a href="/indicators/{{ ticker }}" class="btn">技術指標</a>{% if error %}<div class="card error">{{ error }}</div>{% elif result %}<div class="card"><div class="metric-grid"><div class="metric">💰 初始資金<br><strong>$10,000</strong></div><div class="metric">💰 最終價值<br><strong>${{ result.final_value }}</strong></div><div class="metric">📈 總報酬<br><strong style="color:{{ '#00ff00' if result.total_return>0 else '#ff4444' }}">{{ result.total_return }}%</strong></div><div class="metric">📊 CAGR<br><strong>{{ result.cagr }}%</strong></div><div class="metric">📉 最大回撤<br><strong style="color:#ff4444">{{ result.max_drawdown }}%</strong></div><div class="metric">📐 夏普比率<br><strong>{{ result.sharpe_ratio }}</strong></div><div class="metric">📏 索提諾比率<br><strong>{{ result.sortino_ratio }}</strong></div><div class="metric">⚖️ Beta<br><strong>{{ result.beta }}</strong></div><div class="metric">📊 Alpha<br><strong>{{ result.alpha }}%</strong></div></div></div>{% else %}<div class="card">計算中...</div>{% endif %}</body></html>'''

TENBAGGER_HTML = '''<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>十倍股雷達</title>
<style>
    body{background:#121214;color:#fff;font-family:sans-serif;padding:20px;}
    .card{background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:20px;}
    table{width:100%;border-collapse:collapse;table-layout:fixed;}
    th,td{padding:8px;text-align:left;border-bottom:1px solid #333;word-break:break-word;}
    th{background:#2a2a35;}
    .score-high{color:#00ff00;font-weight:bold;}
    .btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;}
    .stock-link{color:#ffcc00;text-decoration:none;font-weight:bold;background:#333;padding:2px 6px;border-radius:4px;}
</style>
</head>
<body>
<h1>🚀 十倍股候選雷達</h1>
<a href="/" class="btn">首頁</a>
<div class="card"><p>⚡ 條件：營收成長>30% / 毛利率成長 / 52週新高 / 爆量 / 空頭比例</p></div>
<div class="card">
    <table>
        <thead>
            <tr><th>排名</th><th>代號</th><th>價格</th><th>分數</th><th>條件</th></tr>
        </thead>
        <tbody>
            {% for c in candidates %}
            <tr>
                <td>{{ loop.index }}</td>
                <td><a href="/indicators/{{ c.ticker }}" class="stock-link">{{ c.ticker }}</a></td>
                <td>{{ c.price }}</td>
                <td class="score-high">{{ c.score }}/6</td>
                <td>{{ c.conditions|join(', ') }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
</body>
</html>'''
BONDING_HTML = '''<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>均線糾結選股</title>
<style>
    body{background:#121214;color:#fff;font-family:sans-serif;padding:20px;}
    .card{background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:20px;}
    table{width:100%;border-collapse:collapse;table-layout:fixed;}
    th,td{border:1px solid #444;padding:8px;text-align:left;word-break:break-word;}
    th{background:#2a2a35;}
    .stock-link{color:#ffcc00;text-decoration:none;font-weight:bold;background:#333;padding:2px 6px;border-radius:4px;}
    .btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;}
</style>
</head>
<body>
<h1>📈 均線糾結選股</h1>
<div class="note">📊 糾結閾值: {{ threshold }}%</div>
<div class="card"><h2>🇹🇼 台股 ({{ tw_count }})</h2>
{% if tw_results %}
<table>
    <thead>32<th>代號</th><th>價格</th><th>糾結度%</th><th>MA5</th><th>MA20</th><th>MA60</th><th>方向</th></tr></thead>
    <tbody>
    {% for s in tw_results %}
    <tr>
        <td><a href="/indicators/{{ s.ticker }}" class="stock-link">{{ s.ticker }}</a></td>
        <td>{{ s.price }}</td>
        <td>{{ s.spread }}%</td>
        <td>{{ s.ma5 }}</td>
        <td>{{ s.ma20 }}</td>
        <td>{{ s.ma60 }}</td>
        <td>{{ s.direction }}{% if s.volume_confirm %}📊{% endif %}</td>
    </tr>
    {% endfor %}
    </tbody>
</table>
{% else %}<p>無</p>{% endif %}
</div>
<div class="card"><h2>🇺🇸 美股 ({{ us_count }})</h2>
{% if us_results %}
<table>
    <thead>32<th>代號</th><th>價格</th><th>糾結度%</th><th>MA5</th><th>MA20</th><th>MA60</th><th>方向</th></tr></thead>
    <tbody>
    {% for s in us_results %}
    <tr>
        <td><a href="/indicators/{{ s.ticker }}" class="stock-link">{{ s.ticker }}</a></td>
        <td>{{ s.price }}</td>
        <td>{{ s.spread }}%</td>
        <td>{{ s.ma5 }}</td>
        <td>{{ s.ma20 }}</td>
        <td>{{ s.ma60 }}</td>
        <td>{{ s.direction }}{% if s.volume_confirm %}📊{% endif %}</td>
    </tr>
    {% endfor %}
    </tbody>
</table>
{% else %}<p>無</p>{% endif %}
</div>
<a href="/" class="btn">返回首頁</a>
</body>
</html>'''
SETTINGS_HTML = '''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>系統設定</title><style>body{background:#121214;color:#fff;font-family:sans-serif;padding:20px;}.card{background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:20px;}input,button{padding:8px;margin:5px;border-radius:20px;border:none;}button{background:#4caf50;color:white;}.btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;display:inline-block;}</style></head><body><h1>⚙️ 策略設定</h1><div><a href="/" class="btn">返回首頁</a></div><form method="POST"><div class="card"><h3>風險控制</h3><label>單筆最大虧損(%): <input type="number" step="0.5" name="max_loss_per_trade" value="{{ settings.max_loss_per_trade }}"></label><br><label>單一持股最大佔比(%): <input type="number" step="5" name="max_concentration" value="{{ settings.max_concentration }}"></label><br><label>現金餘額: <input type="number" step="1000" name="cash_balance" value="{{ settings.cash_balance }}"></label><br><label>均線糾結閾值(%): <input type="number" step="0.5" name="bonding_threshold" value="{{ settings.bonding_threshold }}"></label><br></div><div class="card"><h3>系統效能</h3><label>背景更新間隔(分鐘): <input type="number" name="background_update_minutes" value="{{ settings.background_update_minutes }}"></label><br><label>資料快取時間(分鐘): <input type="number" name="cache_ttl_minutes" value="{{ settings.cache_ttl_minutes }}"></label><br><label>觀察名單: <input type="text" name="watchlist" value="{{ watchlist }}" style="width:80%;"></label><br></div><button type="submit">儲存設定</button></form></body></html>'''

GEX_SCREENER_HTML = '''<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>GEX 篩選器 - 美股淨 Gamma Exposure</title>
<style>
    body{background:#121214;color:#fff;font-family:sans-serif;padding:20px;}
    .card{background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:20px;}
    table{width:100%;border-collapse:collapse;table-layout:fixed;}
    th,td{padding:10px;text-align:left;border-bottom:1px solid #333;word-break:break-word;}
    th{background:#2a2a35;}
    .positive{color:#00ff00;}.negative{color:#ff4444;}
    .btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;margin:5px;display:inline-block;}
    .stock-link{color:#ffcc00;text-decoration:none;font-weight:bold;background:#333;padding:2px 6px;border-radius:4px;}
</style>
</head>
<body>
<h1>📊 美股 Gamma Exposure (GEX) 篩選器</h1>
<div><a href="/" class="btn">返回儀表板</a> <a href="/force_update_gex" class="btn" style="background:#2196f3;">🔄 手動更新</a></div>
<div class="card"><p>⭐ 淨 GEX = 正 GEX (Call) + 負 GEX (Put)，數值越大代表做市商越可能在該價位提供流動性。</p></div>
<div class="card">
    <table>
        <thead>
            <tr>
                <th>代號</th><th>現價</th><th>淨 GEX</th><th>正 GEX</th>
                <th>負 GEX</th><th>Call Wall</th><th>Put Wall</th><th>Gamma Flip</th><th>更新時間</th>
            </tr>
        </thead>
        <tbody>
            {% for s in stocks %}
            <tr>
                <td><a href="/indicators/{{ s.ticker }}" class="stock-link">{{ s.ticker }}</a></td>
                <td>{{ "${:.2f}".format(s.current_price) if s.current_price else "N/A" }}</td>
                <td class="{{ 'positive' if s.net_gex and s.net_gex > 0 else 'negative' if s.net_gex and s.net_gex < 0 else '' }}">{{ "{:,.0f}".format(s.net_gex) if s.net_gex is not none else "N/A" }}</td>
                <td>{{ "{:,.0f}".format(s.positive_gex) if s.positive_gex is not none else "N/A" }}</td>
                <td>{{ "{:,.0f}".format(s.negative_gex) if s.negative_gex is not none else "N/A" }}</td>
                <td>{{ "${:.2f}".format(s.call_wall) if s.call_wall else "N/A" }}</td>
                <td>{{ "${:.2f}".format(s.put_wall) if s.put_wall else "N/A" }}</td>
                <td>{{ "${:.2f}".format(s.gamma_flip) if s.gamma_flip else "N/A" }}</td>
                <td>{{ s.last_update_str }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
</body>
</html>'''
BATTLE_MAP_HTML = '''<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>盤前戰術板</title>
<style>
    body{background:#121214;color:#fff;font-family:sans-serif;padding:20px;}
    .card{background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:20px;}
    .table-wrapper{overflow-x:auto;}
    table{width:100%;border-collapse:collapse;table-layout:fixed;min-width:800px;}
    th,td{padding:10px 8px;text-align:left;border-bottom:1px solid #333;vertical-align:top;word-break:break-word;white-space:normal;}
    th{background:#2a2a35;font-weight:bold;}
    .positive{color:#00ff00;}.negative{color:#ff4444;}
    .btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;margin:5px;display:inline-block;}
    .stock-link{color:#ffcc00;text-decoration:none;font-weight:bold;background:#333;padding:2px 6px;border-radius:4px;}
    .advice{font-size:12px;color:#ffaa66;line-height:1.4;}
</style>
</head>
<body>
<h1>🗺️ 盤前戰術板 - 美股作戰地圖</h1>
<div><a href="/" class="btn">返回儀表板</a> <a href="/force_update_gex" class="btn" style="background:#2196f3;">🔄 更新 GEX</a></div>
<div class="card"><p>⚡ 使用最新 GEX 數據（每小時更新），標示關鍵壓力/支撐區域。操作建議僅供參考。</p></div>
<div class="table-wrapper">
    <table>
        <thead>
            <tr>
                <th>代號</th><th>現價</th><th>Call Wall</th><th>Put Wall</th><th>Gamma Flip</th>
                <th>淨 GEX</th><th>均線趨勢</th><th>AI分數</th><th>戰術建議</th>
            </tr>
        </thead>
        <tbody>
            {% for s in stocks %}
            <tr>
                <td><a href="/indicators/{{ s.ticker }}" class="stock-link">{{ s.ticker }}</a></td>
                <td>{{ "${:.2f}".format(s.current_price) if s.current_price else "N/A" }}</td>
                <td>{{ "${:.2f}".format(s.call_wall) if s.call_wall else "N/A" }}</td>
                <td>{{ "${:.2f}".format(s.put_wall) if s.put_wall else "N/A" }}</td>
                <td>{{ "${:.2f}".format(s.gamma_flip) if s.gamma_flip else "N/A" }}</td>
                <td class="{{ 'positive' if s.net_gex and s.net_gex > 0 else 'negative' if s.net_gex and s.net_gex < 0 else '' }}">{{ "{:,.0f}".format(s.net_gex) if s.net_gex is not none else "N/A" }}</td>
                <td>{{ s.trend }}</td>
                <td>{{ s.ai_score }}</td>
                <td class="advice">{{ s.advice }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
</body>
</html>'''
SIMULATOR_HTML = '''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>情境模擬器 - {{ ticker }}</title><style>body{background:#121214;color:#fff;font-family:sans-serif;padding:20px;}.card{background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:20px;}.metric-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:15px;}.metric{background:#2a2a35;padding:10px;border-radius:8px;}.btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;margin:5px;display:inline-block;}</style></head><body><h1>🎯 情境模擬器 - {{ ticker }}</h1><div><a href="/" class="btn">返回儀表板</a> <a href="/indicators/{{ ticker }}" class="btn">技術指標</a></div><div class="card"><h3>📌 目前數據 (GEX 快取)</h3><div class="metric-grid"><div class="metric">當前股價: ${{ current_price }}</div><div class="metric">Call Wall: ${{ call_wall if call_wall else "N/A" }}</div><div class="metric">Put Wall: ${{ put_wall if put_wall else "N/A" }}</div><div class="metric">Gamma Flip: ${{ gamma_flip if gamma_flip else "N/A" }}</div></div><h4>📊 目前分析</h4><div class="metric-grid"><div class="metric">距離 Call Wall: {{ current.call_dist_pct if current.call_dist_pct is not none else "N/A" }}%<br>狀態: {{ current.call_status if current.call_status else "N/A" }}</div><div class="metric">距離 Put Wall: {{ current.put_dist_pct if current.put_dist_pct is not none else "N/A" }}%<br>狀態: {{ current.put_status if current.put_status else "N/A" }}</div><div class="metric">距離 Gamma Flip: {{ current.flip_dist_pct if current.flip_dist_pct is not none else "N/A" }}%<br>{{ current.flip_status if current.flip_status else "N/A" }}</div></div></div><div class="card"><h3>🔮 輸入模擬價格</h3><form method="GET" action="/simulator/{{ ticker }}"><input type="number" step="0.01" name="price" placeholder="輸入模擬股價" required><button type="submit" class="btn">模擬</button></form>{% if simulated %}<h4>📈 模擬結果 (股價 = ${{ simulated.price }})</h4><div class="metric-grid"><div class="metric">距離 Call Wall: {{ simulated.call_dist_pct }}%<br>狀態: {{ simulated.call_status }}</div><div class="metric">距離 Put Wall: {{ simulated.put_dist_pct }}%<br>狀態: {{ simulated.put_status }}</div><div class="metric">距離 Gamma Flip: {{ simulated.flip_dist_pct }}%<br>{{ simulated.flip_status }}</div></div><div class="metric"><strong>💡 戰術建議</strong><br>{% if simulated.call_dist_pct < 1 and simulated.call_dist_pct > 0 %}⚠️ 接近 Call Wall 壓力區，短線遇壓<br>{% elif simulated.call_dist_pct < 0 %}✅ 突破 Call Wall，空翻多<br>{% endif %}{% if simulated.put_dist_pct < 1 and simulated.put_dist_pct > 0 %}🟢 接近 Put Wall 支撐區，止跌反彈機率高<br>{% elif simulated.put_dist_pct < 0 %}❌ 跌破 Put Wall，加速下跌風險<br>{% endif %}{% if simulated.flip_dist_pct > 0 %}⚡ 站上 Gamma Flip，做市商轉為撐盤<br>{% else %}⚡ 低於 Gamma Flip，易遭追殺<br>{% endif %}</div>{% endif %}</div></body></html>'''

EARNINGS_CALENDAR_HTML = '''<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>美股財報日曆</title>
<style>
    body{background:#121214;color:#fff;font-family:sans-serif;padding:20px;}
    .card{background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:20px;}
    table{width:100%;border-collapse:collapse;table-layout:fixed;}
    th,td{padding:10px;text-align:left;border-bottom:1px solid #333;word-break:break-word;}
    th{background:#2a2a35;}
    .stock-link{color:#ffcc00;text-decoration:none;font-weight:bold;background:#333;padding:2px 6px;border-radius:4px;}
    .btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;margin:5px;display:inline-block;}
    .warning{color:#ff4444;}.caution{color:#ffaa00;}
</style>
</head>
<body>
<h1>📅 美股財報時間表</h1>
<div><a href="/" class="btn">返回儀表板</a> <a href="/force_update_earnings" class="btn" style="background:#4caf50;">🔄 更新財報快取</a></div>
<div class="card"><p>⚡ 財報日期來自 Yahoo Finance，可能略有誤差。預期波動為 ATM Straddle 估算百分比。</p></div>
<div class="card">
    <table>
        <thead>
            <tr><th>代號</th><th>財報日期</th><th>狀態</th><th>預期波動 (%)</th></tr>
        </thead>
        <tbody>
            {% for s in stocks %}
            <tr>
                <td><a href="/indicators/{{ s.ticker }}" class="stock-link">{{ s.ticker }}</a></td>
                <td>{{ s.date }}</td>
                <td class="{% if '今日' in s.status or ('天後' in s.status and '高風險' in s.status) %}warning{% elif '天後' in s.status %}caution{% endif %}">{{ s.status }}</div>
                <td>{{ "%.2f"|format(s.expected_move) if s.expected_move else "N/A" }}%</div>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
</body>
</html>'''
EARNINGS_SIMULATOR_HTML = '''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>財報模擬器 - {{ ticker }}</title><style>body{background:#121214;color:#fff;font-family:sans-serif;padding:20px;}.card{background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:20px;}.metric-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:15px;}.metric{background:#2a2a35;padding:10px;border-radius:8px;}.btn{background:#2a2a35;padding:6px 14px;border-radius:20px;color:white;text-decoration:none;margin:5px;display:inline-block;}.slider{width:100%;}</style></head><body><h1>🎯 財報模擬器 - {{ ticker }}</h1><div><a href="/" class="btn">返回儀表板</a> <a href="/indicators/{{ ticker }}" class="btn">技術指標</a> <a href="/gex_plot/{{ ticker }}" class="btn">GEX 圖表</a></div><div class="card"><h3>📌 目前 GEX 數據</h3><div class="metric-grid"><div class="metric">當前股價: ${{ "%.2f"|format(current_price) if current_price else "N/A" }}</div><div class="metric">Call Wall: ${{ "%.2f"|format(call_wall) if call_wall else "N/A" }}</div><div class="metric">Put Wall: ${{ "%.2f"|format(put_wall) if put_wall else "N/A" }}</div><div class="metric">Gamma Flip: ${{ "%.2f"|format(gamma_flip) if gamma_flip else "N/A" }}</div><div class="metric">預期波動: {{ "%.2f"|format(expected_move) if expected_move else "N/A" }}%</div></div></div><div class="card"><h3>🔮 財報後股價模擬</h3><form method="GET" action="/earnings_simulator/{{ ticker }}"><label>股價變動百分比: <input type="range" class="slider" name="pct" min="-30" max="30" step="1" value="{{ pct }}" onchange="this.form.submit()"> {{ pct }}%</label><noscript><button type="submit">模擬</button></noscript></form>{% if sim_result %}<h4>📈 模擬結果 (股價 = ${{ sim_result.sim_price }}, 變動 {{ sim_result.pct_change }}%)</h4><div class="metric-grid"><div class="metric">距離 Call Wall: {{ sim_result.call_dist }}%<br>{% if sim_result.call_cross %}🚨 已穿越 Call Wall！{% endif %}</div><div class="metric">距離 Put Wall: {{ sim_result.put_dist }}%<br>{% if sim_result.put_cross %}🚨 已穿越 Put Wall！{% endif %}</div><div class="metric">距離 Gamma Flip: {{ sim_result.flip_dist }}%<br>{% if sim_result.flip_cross %}⚡ Gamma Flip 翻轉！{% endif %}</div></div><div class="metric"><strong>💡 戰術提示</strong><br>{% if sim_result.call_dist < 1 and sim_result.call_dist > 0 %}⚠️ 接近 Call Wall 強壓力區，做市商賣壓可能出現。<br>{% elif sim_result.call_dist < 0 %}✅ 站上 Call Wall，突破壓力轉為支撐。<br>{% endif %}{% if sim_result.put_dist < 1 and sim_result.put_dist > 0 %}🟢 接近 Put Wall 強支撐區，下跌空間有限。<br>{% elif sim_result.put_dist < 0 %}❌ 跌破 Put Wall，下方無支撐，加速下跌。<br>{% endif %}{% if sim_result.flip_dist > 0 %}⚡ 位於 Gamma Flip 上方，做市商 Long Gamma 撐盤。<br>{% else %}⚡ 位於 Gamma Flip 下方，做市商 Short Gamma 追殺風險高。<br>{% endif %}</div>{% else %}<p>請使用滑桿調整股價變動百分比</p>{% endif %}</div></body></html>'''

if __name__ == "__main__":
    print("🚀 啟動 AI 量化監控中心 v16_final (徹底修正表格標籤)")
    print("📊 訪問地址：http://127.0.0.1:5005")
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=update_all_ai_scores, trigger="interval", minutes=settings.get('background_update_minutes',15))
    scheduler.add_job(func=update_tenbagger, trigger="interval", minutes=30)
    scheduler.add_job(func=scan_alerts, trigger="interval", minutes=10)
    scheduler.add_job(func=update_all_gex_screener, trigger="interval", hours=1)
    scheduler.add_job(func=update_earnings_dates, trigger="interval", hours=24)
    scheduler.start()
    threading.Thread(target=update_all_ai_scores, daemon=True).start()
    threading.Thread(target=update_all_gex_screener, daemon=True).start()
    threading.Thread(target=update_earnings_dates, daemon=True).start()
    app.run(host="127.0.0.1", port=5005, debug=True, threaded=True)