@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo  XH-202607 决策模块 - 本地生成任务 JSON
echo  不需要 Dify / Docker / 显卡（见上一级「使用说明.md」未连接智能体时）
echo ========================================
echo.

where python >nul 2>&1
if %errorlevel%==0 (
  set PY=python
  goto :run
)

where py >nul 2>&1
if %errorlevel%==0 (
  set PY=py -3
  goto :run
)

echo [错误] 未检测到 Python。
echo 请先安装：https://www.python.org/downloads/
echo 安装时务必勾选 Add python.exe to PATH，然后重新打开本窗口。
echo.
pause
exit /b 1

:run
echo 使用解释器: %PY%
echo 正在生成 delivery_for_simulation.json ...
echo.
%PY% build_tasks_local.py
if %errorlevel% neq 0 (
  echo.
  echo [失败] 请确认上一级目录有 simple_dataset.json 与 workflow_code
  pause
  exit /b 1
)

echo.
echo 完成。请查看上一级目录：
echo   delivery_for_simulation.json
echo   delivery_scenario_35.json
echo.
pause
