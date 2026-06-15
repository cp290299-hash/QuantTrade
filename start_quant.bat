@echo off
chcp 65001 >nul
cd /d "C:\Users\cp290\Desktop\QuantTrade"

title Quant Monitor Starter

if not exist "logs" mkdir logs

echo Checking and stopping old Quant Monitor...
call stop_quant.bat >nul 2>&1

echo Starting AI Quant Monitor...
powershell -Command "$process = Start-Process -FilePath 'python' -ArgumentList 'app.py' -WindowStyle Hidden -RedirectStandardOutput 'logs\quant_out.txt' -RedirectStandardError 'logs\quant_err.txt' -PassThru; $process.Id | Out-File -FilePath 'quant.pid' -Encoding ascii; Write-Host 'Started successfully, PID =' $process.Id"

echo Waiting for server to start (max 60 seconds)...
set WAIT_COUNT=0
:CHECK_PORT
netstat -ano | findstr :5005 | findstr LISTENING >nul
if %errorlevel% equ 0 goto SERVER_READY
timeout /t 1 /nobreak >nul
set /a WAIT_COUNT+=1
if %WAIT_COUNT% geq 60 goto TIMEOUT
goto CHECK_PORT

:TIMEOUT
echo Timeout reached. Check logs\quant_err.txt for errors.
start notepad logs\quant_err.txt
pause
exit /b 1

:SERVER_READY
echo Server ready, opening dashboard...
start http://127.0.0.1:5005/
echo.
echo ==============================================
echo Quant Monitor started successfully!
echo Dashboard: http://127.0.0.1:5005/
echo PID saved to quant.pid
echo Stop with stop_quant.bat
echo ==============================================
exit /b 0