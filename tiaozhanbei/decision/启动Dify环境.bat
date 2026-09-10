@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo  Start Dify  (Docker / port 8080)
echo ========================================
echo.

echo [CHECK] Docker ...
where docker >nul 2>&1
if errorlevel 1 (
  echo.
  echo [STOP] Docker is not installed.
  echo Install Docker Desktop and wait until the tray icon is GREEN.
  echo https://www.docker.com/products/docker-desktop/
  echo See the guide markdown in this folder, section 1.2
  echo.
  pause
  exit /b 1
)

echo [CHECK] Git ...
where git >nul 2>&1
if errorlevel 1 (
  echo.
  echo [STOP] Git is not installed.
  echo https://git-scm.com/download/win
  echo See the guide markdown in this folder, section 1.3
  echo.
  pause
  exit /b 1
)

set "DIFY_HOME=%USERPROFILE%\dify"
if exist "%DIFY_HOME%\docker\compose.yaml" goto :compose
if exist "%DIFY_HOME%\docker\docker-compose.yaml" goto :compose

echo Cloning Dify to %DIFY_HOME% ...
git clone --depth 1 https://github.com/langgenius/dify.git "%DIFY_HOME%"
if errorlevel 1 (
  echo [ERROR] git clone failed
  pause
  exit /b 1
)

:compose
cd /d "%DIFY_HOME%\docker"
if not exist ".env" copy /Y ".env.example" ".env" >nul

python -c "from pathlib import Path; p=Path('.env'); t=p.read_text(encoding='utf-8', errors='replace'); t=t.replace('EXPOSE_NGINX_PORT=80','EXPOSE_NGINX_PORT=8080').replace('EXPOSE_NGINX_SSL_PORT=443','EXPOSE_NGINX_SSL_PORT=8443'); p.write_text(t, encoding='utf-8')"
if errorlevel 1 (
  echo [WARN] Could not patch .env ports with Python. Edit .env: EXPOSE_NGINX_PORT=8080
)

echo docker compose up -d
echo First run downloads images and can take several minutes ...
docker compose up -d
if errorlevel 1 (
  echo [ERROR] docker compose failed. Start Docker Desktop first.
  pause
  exit /b 1
)

echo.
echo ========================================
echo  Done.
echo  Open a browser:
echo      http://localhost:8080
echo  Next steps: guide markdown in this folder, section 3 and 4
echo      1. register admin
echo      2. set chat model
echo      3. import workflows\workflow.yml and Publish
echo      4. create API key into dify_api_key.txt
echo      5. python agent_demo_ui.py
echo ========================================
echo.
pause
