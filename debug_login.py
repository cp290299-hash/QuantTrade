# -*- coding: utf-8 -*-
import shioaji as sj

print("準備登入...")
try:
    api = sj.Shioaji()
    # 使用正確的參數名稱：api_key, secret_key
    api.login(
        api_key="HnsXBpU7LL6WayAvTgoM8vaWDEDsDduoPd6CApYuXVRx",
        secret_key="Eob5rnN48x3Y7cK2vra3ddfXHjQQMfVgeueFjWLR1gnD",
        ca_path="Sinopac.pfx",
        ca_passwd="B120492622"
    )
    print("✅ 登入成功！")
except Exception as e:
    print(f"❌ 登入失敗，錯誤訊息: {e}")