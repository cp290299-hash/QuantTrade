import os
import sys
import logging
from dotenv import load_dotenv

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    load_dotenv()
    required = ["SHIOAJI_API_KEY", "SHIOAJI_SECRET_KEY", "SHIOAJI_CA_PATH", "SHIOAJI_CA_PASS", "SHIOAJI_PERSON_ID"]
    for key in required:
        if not os.getenv(key):
            logging.error(f"缺少環境變數: {key}")
            sys.exit(1)
    return {k: os.getenv(k) for k in required}