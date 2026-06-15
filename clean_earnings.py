import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 刪除任何可能單獨出現的 EARNINGS_ 變數（非函數呼叫）
# 例如：EARNINGS_ 或 EARNINGS_SIMULATOR_HTML
content = re.sub(r'\bEARNINGS_\w*\b', '', content)

# 清除可能因此產生的多餘逗號或空括號
content = re.sub(r',\s*,', ',', content)
content = re.sub(r',\s*\)', ')', content)
content = re.sub(r'\(\s*,', '(', content)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 已清理 EARNINGS_ 相關殘留")