@echo off
REM ===========================================================================
REM  dashboard.bat -- live web dashboard for the translation pipeline
REM
REM  Usage:
REM     dashboard.bat                      2023 run  (config_2023.json), port 8765
REM     dashboard.bat -c config_2022.json  any other year
REM     dashboard.bat --port 8766          run two dashboards side by side
REM     dashboard.bat --no-browser         don't open a browser window
REM
REM  Read-only: safe to start, stop and restart while a translation is running.
REM ===========================================================================

setlocal

set "PY=C:\darsh\AI_MODELS\translator_env\Scripts\python.exe"
set "HERE=%~dp0"

if not exist "%PY%" (
    echo [ERROR] Python not found at "%PY%".
    exit /b 1
)

if "%~1"=="" (
    "%PY%" "%HERE%dashboard.py" -c "%HERE%config_2023.json"
) else (
    "%PY%" "%HERE%dashboard.py" %*
)

exit /b %ERRORLEVEL%
