import re
import os
import shutil

# 1. 備份原始 app.py
shutil.copy('app.py', 'app.py.backup')
print("✅ 已備份 app.py -> app.py.backup")

# 2. 讀取原始內容
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 3. 修正 import：確保 render_template 被導入
# 尋找 from flask import ... 這一行
import_line_match = re.search(r'from flask import (.*)', content)
if import_line_match:
    imports = import_line_match.group(1)
    if 'render_template' not in imports:
        # 加入 render_template
        new_imports = imports.rstrip() + ', render_template'
        content = content.replace(import_line_match.group(0), f'from flask import {new_imports}')
        print("✅ 已加入 render_template 到 import")
    else:
        print("ℹ️ render_template 已存在 import 中")
else:
    print("⚠️ 未找到 from flask import 行，請手動檢查")

# 4. 定義變數與檔名的對應
mapping = {
    'INDEX_HTML': 'index.html',
    'TW_US_HTML': 'tw_us.html',
    'AI_RANKING_HTML': 'ai_ranking.html',
    'POSITIONS_HTML': 'positions.html',
    'INDICATORS_HTML': 'indicators.html',
    'BACKTEST_HTML': 'backtest.html',
    'TENBAGGER_HTML': 'tenbagger.html',
    'BONDING_HTML': 'bonding.html',
    'SETTINGS_HTML': 'settings.html',
    'GEX_SCREENER_HTML': 'gex_screener.html',
    'BATTLE_MAP_HTML': 'battle_map.html',
    'SIMULATOR_HTML': 'simulator.html',
    'EARNINGS_CALENDAR_HTML': 'earnings_calendar.html',
    'EARNINGS_SIMULATOR_HTML': 'earnings_simulator.html',
}

# 5. 替換 render_template_string 為 render_template
for var_name, filename in mapping.items():
    # 匹配 render_template_string(XXX_HTML, 其他參數...)
    # 注意：可能跨多行，使用 re.DOTALL
    pattern = rf'render_template_string\(\s*{var_name}\s*,\s*(.*?)\)'
    # 替換為 render_template('filename.html', 其他參數)
    repl = rf"render_template('{filename}', \1)"
    content = re.sub(pattern, repl, content, flags=re.DOTALL)
    print(f"✅ 已替換 {var_name} -> {filename}")

# 6. 刪除所有 HTML 變數的定義（INDEX_HTML = '''...''' 等）
# 這些變數定義可能是多行字符串，需要小心處理
# 方法：對每個變數，刪除從 var_name = ''' 到對應的 ''' 的區塊
for var_name in mapping.keys():
    # 匹配 var_name = ''' ... ''' （非貪婪，跨行）
    pattern = rf'{var_name}\s*=\s*\'\'\'(.*?)\'\'\''
    content = re.sub(pattern, '', content, flags=re.DOTALL)
    print(f"🗑️ 已刪除 {var_name} 定義")

# 清除可能殘留的空行（連續兩個換行變成一個）
content = re.sub(r'\n\s*\n', '\n\n', content)

# 7. 寫回 app.py
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("🎉 所有改寫完成！請檢查 app.py 並執行測試。")
print("💡 原始檔案已備份為 app.py.backup，若有問題可還原。")