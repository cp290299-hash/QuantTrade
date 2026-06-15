import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修正兩個連續逗號
content = re.sub(r',\s*,', ',', content)

# 修正 render_template 中可能殘留的尾隨逗號（例如 ... ,) ）
content = re.sub(r',\s*\)', ')', content)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 已修正多餘逗號")