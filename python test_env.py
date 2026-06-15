from dotenv import load_dotenv
import os

# 強制載入
load_dotenv(override=True)

# 測試讀取
person_id = os.getenv("SHIOAJI_PERSON_ID")
print(f"讀取到的身分證字號為: {person_id}")

if person_id == "B120492622":
    print("成功：環境變數讀取正常！")
else:
    print("失敗：讀取內容有誤，請檢查 .env 檔案是否有亂碼。")