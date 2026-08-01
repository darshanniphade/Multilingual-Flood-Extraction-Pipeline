@echo off
REM ===========================================================================
REM  run.bat -- translate flood-event articles to English (offline, CUDA)
REM
REM  Usage:
REM     run.bat                     translate everything (resumes automatically)
REM     run.bat --limit 200         smoke test on 200 articles
REM     run.bat --dry-run           resolve ids + build index, no model load
REM     run.bat --no-resume         ignore checkpoint, start over
REM
REM  Safe to interrupt with Ctrl+C: progress is checkpointed and the next run
REM  resumes where this one stopped.
REM ===========================================================================

setlocal

set "PY=C:\darsh\AI_MODELS\translator_env\Scripts\python.exe"
set "HERE=%~dp0"

REM -- Force fully-offline operation: never contact the Hugging Face hub. ----
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"

REM -- Workers do their own batching; per-worker thread pools would oversubscribe.
set "TOKENIZERS_PARALLELISM=false"

REM -- Allocator tuning is set inside translator.py (it must be applied before
REM    CUDA initialises). Note expandable_segments is NOT used: it is silently
REM    unsupported on Windows. Do not set PYTORCH_CUDA_ALLOC_CONF here.

if not exist "%PY%" (
    echo [ERROR] Python not found at "%PY%".
    echo         Edit the PY variable at the top of run.bat.
    exit /b 1
)

echo ============================================================
echo  Flood article translator  ^|  NLLB-200-distilled-1.3B
echo  Python : %PY%
echo  Args   : %*
echo ============================================================

pushd "%HERE%"
"%PY%" "%HERE%translate_flood_articles.py" %*
set "RC=%ERRORLEVEL%"
popd

if "%RC%"=="0" (
    echo.
    echo [OK] Finished. See translation.log for details.
) else if "%RC%"=="130" (
    echo.
    echo [STOPPED] Interrupted. Re-run run.bat to resume from the checkpoint.
) else (
    echo.
    echo [FAILED] Exit code %RC%. See translation.log for the traceback.
)

exit /b %RC%
