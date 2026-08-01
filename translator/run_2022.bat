@echo off
REM ===========================================================================
REM  run_2022.bat -- translate the 2022 flood-event articles (offline, CUDA)
REM
REM  Same as run.bat but uses config_2022.json:
REM     events : C:\darsh\pipeline\data\flood_events\2022
REM     data   : C:\darsh\pipeline\translator\data2022_links  (junctions ->
REM              C:\darsh\pipeline\data\ssd\2022\MM, because article ids are
REM              "2022_MM/article_..." but the source tree uses bare "MM")
REM     output : C:\darsh\pipeline\data\translated_articles\2022
REM     state  : C:\darsh\pipeline\translator\state_2022
REM
REM  Usage:
REM     run_2022.bat                 translate everything (resumes automatically)
REM     run_2022.bat --limit 200     smoke test on 200 articles
REM     run_2022.bat --dry-run       resolve ids + build index, no model load
REM     run_2022.bat --no-resume     ignore checkpoint, start over
REM
REM  Safe to interrupt with Ctrl+C: progress is checkpointed and the next run
REM  resumes where this one stopped.
REM ===========================================================================

setlocal

set "PY=C:\darsh\AI_MODELS\translator_env\Scripts\python.exe"
set "HERE=%~dp0"
set "CFG=%HERE%config_2022.json"

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

if not exist "%HERE%data2022_links\2022_01\articles" (
    echo [ERROR] Month junctions missing or broken: "%HERE%data2022_links".
    echo         Recreate them with:
    echo             for /L %%%%m in ^(1,1,9^) do mklink /J "%HERE%data2022_links\2022_0%%%%m" "C:\darsh\pipeline\data\ssd\2022\0%%%%m"
    echo             for /L %%%%m in ^(10,1,12^) do mklink /J "%HERE%data2022_links\2022_%%%%m" "C:\darsh\pipeline\data\ssd\2022\%%%%m"
    exit /b 1
)

echo ============================================================
echo  Flood article translator  ^|  NLLB-200-distilled-1.3B  ^|  2022
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
    echo [OK] Finished. See translation_2022.log for details.
) else if "%RC%"=="130" (
    echo.
    echo [STOPPED] Interrupted. Re-run run_2022.bat to resume from the checkpoint.
) else (
    echo.
    echo [FAILED] Exit code %RC%. See translation_2022.log for the traceback.
)

exit /b %RC%
