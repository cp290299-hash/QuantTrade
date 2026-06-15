import re
import os

# 讀取 app.py
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 變數名稱 → 目標檔名
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

# 建立 templates 資料夾（如果不存在）
os.makedirs('templates', exist_ok=True)

for var_name, filename in mapping.items():
    # 用正則找出 var_name = ''' ... '''
    pattern = rf"{var_name}\s*=\s*'''(.+?)'''"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        html_content = match.group(1)
        filepath = os.path.join('templates', filename)
        with open(filepath, 'w', encoding='utf-8') as out:
            out.write(html_content)
        print(f"✅ 已建立 {filepath}")
    else:
        print(f"⚠️ 找不到 {var_name}，可能已經被刪除或格式不同")

print("🎉 全部完成！現在請手動修改路由中的 render_template_string 為 render_template")