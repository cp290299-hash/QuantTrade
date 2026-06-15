import re

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 過濾掉只包含空白和一個右括號的行（例如 ")\n" 或 "  )\n"）
new_lines = []
for line in lines:
    if re.match(r'^\s*\)\s*$', line):
        print(f"刪除孤立右括號行: {line.strip()}")
        continue  # 跳過這行
    new_lines.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✅ 已清理孤立的右括號，請重新執行 app.py")