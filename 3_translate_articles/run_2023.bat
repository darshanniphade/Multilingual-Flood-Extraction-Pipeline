@echo off
REM ===========================================================================
REM  run_2023.bat -- translate the 2023 flood-event articles (offline, CUDA)
REM
REM  Same as run.bat but uses config_2023.json:
REM     events : C:\darsh\pipeline\data\flood_events\2023
REM     data   : C:\darsh\pipeline\data\ssd\2023   (no junction staging needed --
REM              article ids are "2023_MM/article_..." and the source tree
REM              already uses "2023_MM" month folders)
REM     output : C:\darsh\pipeline\data\translated_articles\2023
REM     state  : C:\darsh\pipeline\3_translate_articles\state_2023
REM
REM  Usage:
REM     run_2023.bat                 translate everything (resumes automatically)
REM     run_2023.bat --limit 200     smoke test on 200 articles
REM     run_2023.bat --dry-run       resolve ids + build index, no model load
REM     run_2023.bat --no-resume     ignore checkpoint, start over
REM
REM  Safe to interrupt with Ctrl+C: progress is checkpointed and the next run
REM  resumes where this one stopped.
REM ===========================================================================

setlocal

set "PY=C:\darsh\AI_MODELS\translator_env\Scripts\python.exe"
set "HERE=%~dp0"
set "CFG=%HERE%config_2023.json"

REM -- Force fully-offline operation: never contact the Hugging Face hub. ----
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"

REM -- Workers do their own batching; per-worker thread pools would oversubscribe.
set "TOKENIZERS_PARALLELISM=false"

if not exist "%PY%" (
    echo [ERROR] Python not found at "%PY%".
    exit /b 1
)

if not exist "C:\darsh\pipeline\data\ssd\2023\2023_01\articles" (
    echo [ERROR] Source tree missing: C:\darsh\pipeline\data\ssd\2023\2023_01\articles
    exit /b 1
)

echo ============================================================
echo  Flood article translator  ^|  NLLB-200-distilled-1.3B  ^|  2023
echo  Python : %PY%
echo  Config : %CFG%
echo  Args   : %*
echo ============================================================

pushd "%HERE%"
"%PY%" "%HERE%translate_flood_articles.py" -c "%CFG%" %*
set "RC=%ERRORLEVEL%"
popd

if "%RC%"=="0" (
    echo.
    echo [OK] Finished. See translation_2023.log for details.
) else if "%RC%"=="130" (
    echo.
    echo [STOPPED] Interrupted. Re-run run_2023.bat to resume from the checkpoint.
) else (
    echo.
    echo [FAILED] Exit code %RC%. See translation_2023.log for the traceback.
)

exit /b %RC%
