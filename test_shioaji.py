import os
from dotenv import load_dotenv
load_dotenv()
import shioaji as sj

api_key = os.getenv("SHIOAJI_API_KEY")
secret_key = os.getenv("SHIOAJI_SECRET_KEY")
person_id = os.getenv("SHIOAJI_PERSON_ID")
passwd = os.getenv("SHIOAJI_PASSWD")

print(f"API Key: {api_key[:5]}...")
print(f"Person ID: {person_id}")

try:
    sj_api = sj.Shioaji(simulation=False)
    sj_api.login(api_key=api_key, secret_key=secret_key)
    print("✅ 登入成功")
    sj_api.logout()
except Exception as e:
    print(f"❌ 登入失敗: {e}")
    import traceback
    traceback.print_exc()