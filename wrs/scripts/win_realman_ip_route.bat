@echo off
:: Check for admin rights
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo Requesting administrative privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:: Function to check and add route
call :AddRouteIfNotExist 192.168.0.0 255.255.255.0 192.168.124.104
call :AddRouteIfNotExist 192.168.3.0 255.255.255.0 192.168.124.104

pause
exit /b.

:AddRouteIfNotExist
setlocal
set DEST=%1
set MASK=%2
set GATEWAY=%3

route print | findstr /C:"%DEST%" >nul
if %errorlevel%==0 (
    echo Route to %DEST% already exists.
) else (
    echo Adding route to %DEST% via %GATEWAY%
    route add %DEST% mask %MASK% %GATEWAY%
)
endlocal
exit /b
