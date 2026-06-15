@echo off
cd /d "C:\Users\cp290\Desktop\QuantTrade"
git add .
git commit -m "自動更新 %date% %time%"
git push
echo 已上傳，Render 將自動部署。
pause