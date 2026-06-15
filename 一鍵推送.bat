@echo off
chcp 65001 >nul
echo ========================================
echo     一鍵推送量化系統更新到 GitHub
echo ========================================
echo.

:: 切換到專案目錄
cd /d "C:\Users\cp290\Desktop\QuantTrade"

:: 檢查是否為 Git 倉庫
if not exist ".git" (
    echo 錯誤：找不到 .git 資料夾，請確認路徑是否正確。
    pause
    exit /b 1
)

:: 顯示目前修改狀態
echo 正在檢查變更的檔案...
git status

:: 詢問是否繼續
set /p confirm="確定要將所有變更推送到 GitHub 嗎？(y/n): "
if /i not "%confirm%"=="y" (
    echo 已取消。
    pause
    exit /b 0
)

:: 加入所有變更（包括新增、修改、刪除）
git add .

:: 產生提交訊息（包含日期時間）
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set DATE=%%a/%%b/%%c
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set TIME=%%a:%%b
set COMMIT_MSG="自動推送 %DATE% %TIME%"

:: 提交
echo 正在提交...
git commit -m %COMMIT_MSG%

:: 推送到 GitHub
echo 正在推送到 GitHub...
git push

:: 完成
echo.
echo ========================================
echo 推送完成！Render 將自動部署，請稍後查看日誌。
echo ========================================
pause