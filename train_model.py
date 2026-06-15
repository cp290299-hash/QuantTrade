import yfinance as yf
import xgboost as xgb
import pandas as pd

def train_and_save():
    # 這裡可以用代表性的數據進行預訓練
    data = yf.download("2330.TW", period="1y", progress=False)
    df = data.copy()
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
    df['MA_Ratio'] = df['Close'] / df['Close'].rolling(20).mean()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    
    model = xgb.XGBClassifier(n_estimators=50, max_depth=3)
    model.fit(df[['RSI', 'MA_Ratio']], df['Target'])
    model.save_model("model.json")
    print("模型已儲存為 model.json")

if __name__ == "__main__":
    train_and_save()