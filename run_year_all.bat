@echo off
REM ===========================================================================
REM  run_year_all.bat -- run the whole pipeline for one year, in the correct
REM  order, from the raw crawl to extractions.csv.
REM
REM      stage 1  1_translate_titles  ssd\<year>                -> translated titles\<year>
REM      stage 2  2_filter_titles     translated titles\<year>  -> flood_events\<year>
REM      stage 3  translator       ssd + flood_events        -> translated_articles\<year>
REM      stage 4  5_english_only          translated_articles\<year>-> only eng\<year>
REM      stage 5  extract          only eng\<year>           -> extracted\<year>
REM      stage 6  to_csv           extracted\<year>          -> extractions.csv
REM
REM  Every stage resumes from its own output, so this is safe to Ctrl+C and
REM  re-run: finished stages skip, the interrupted one picks up where it left.
REM
REM  Usage:
REM      run_year_all.bat [year]        (default: 2025)
REM
REM  A new year only needs translator\config_<year>.json and
REM  extract\config_<year>.json to exist; the raw-crawl root is detected
REM  automatically (some years nest month folders under downloaded_articles).
REM
REM  Stage 5 needs a running Ollama with qwen3:14b and these server-side vars:
REM      OLLAMA_NUM_PARALLEL=12  OLLAMA_KV_CACHE_TYPE=q8_0  OLLAMA_FLASH_ATTENTION=1
REM ===========================================================================

setlocal enabledelayedexpansion

set "YEAR=%~1"
if "%YEAR%"=="" set "YEAR=2025"

REM -- Raw crawl root: prefer the nested downloaded_articles layout (2024),
REM    fall back to month folders directly under ssd\<year> (2025).
set "RAW=C:\darsh\pipeline\data\ssd\%YEAR%\downloaded_articles"
if not exist "%RAW%\%YEAR%_01\articles" set "RAW=C:\darsh\pipeline\data\ssd\%YEAR%"

set "PY=C:\darsh\AI_MODELS\translator_env\Scripts\python.exe"
set "ROOT=%~dp0"
set "DATA=C:\darsh\pipeline\data"

REM -- Fully offline: never contact the Hugging Face hub. --------------------
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "TOKENIZERS_PARALLELISM=false"

if not exist "%PY%" (
    echo [ERROR] Python not found at "%PY%".
    exit /b 1
)
if not exist "%RAW%\%YEAR%_01\articles" (
    echo [ERROR] Raw crawl tree missing: "%RAW%\%YEAR%_01\articles"
    exit /b 1
)
if not exist "%ROOT%3_translate_articles\config_%YEAR%.json" (
    echo [ERROR] Missing translator\config_%YEAR%.json
    exit /b 1
)
if not exist "%ROOT%6_extract_events\config_%YEAR%.json" (
    echo [ERROR] Missing extract\config_%YEAR%.json
    exit /b 1
)

echo ============================================================
echo  Flood pipeline ^| year %YEAR% ^| stages 1-6
echo  Raw    : %RAW%
echo  Python : %PY%
echo ============================================================

REM --------------------------------------------------------------------------
REM  Stage 0 -- quarantine artifacts from the out-of-order run.
REM
REM  flood_events\<year> must hold ONLY the stage-2 files <year>_MM.jsonl.
REM  Anything else there (month FOLDERS written by a mis-aimed 4_clean_dedup run, a
REM  stray translated_titles.jsonl) would be picked up by stage 3's event scan.
REM  These are RENAMED into data\_quarantine_<year>, never deleted.
REM --------------------------------------------------------------------------
set "QUAR=%DATA%\_quarantine_%YEAR%"
if exist "%DATA%\flood_events\%YEAR%\%YEAR%_01\" (
    echo [stage 0] Moving out-of-order artifacts to "%QUAR%" ...
    if not exist "%QUAR%" mkdir "%QUAR%"
    if not exist "%QUAR%\flood_events_%YEAR%" mkdir "%QUAR%\flood_events_%YEAR%"
    for /l %%M in (1,1,9) do (
        if exist "%DATA%\flood_events\%YEAR%\%YEAR%_0%%M\" move /y "%DATA%\flood_events\%YEAR%\%YEAR%_0%%M" "%QUAR%\flood_events_%YEAR%\" >nul
    )
    for /l %%M in (10,1,12) do (
        if exist "%DATA%\flood_events\%YEAR%\%YEAR%_%%M\" move /y "%DATA%\flood_events\%YEAR%\%YEAR%_%%M" "%QUAR%\flood_events_%YEAR%\" >nul
    )
)
if exist "%DATA%\flood_events\%YEAR%\translated_titles.jsonl" (
    if not exist "%QUAR%" mkdir "%QUAR%"
    move /y "%DATA%\flood_events\%YEAR%\translated_titles.jsonl" "%QUAR%\misplaced_translated_titles.jsonl" >nul
)
if exist "%DATA%\translated titles\%YEAR%\%YEAR%_01.jsonl" (
    if not exist "%QUAR%\translated_titles_%YEAR%" mkdir "%QUAR%\translated_titles_%YEAR%"
    move /y "%DATA%\translated titles\%YEAR%\%YEAR%_*.jsonl" "%QUAR%\translated_titles_%YEAR%\" >nul
)

if not exist "%DATA%\translated titles\%YEAR%" mkdir "%DATA%\translated titles\%YEAR%"
if not exist "%DATA%\flood_events\%YEAR%" mkdir "%DATA%\flood_events\%YEAR%"

REM --------------------------------------------------------------------------
echo.
echo [stage 1/6] Title translation  ^(NLLB-200^)
pushd "%ROOT%1_translate_titles"
"%PY%" translate_titles.py --input-dir "%RAW%" --output-dir "%DATA%\translated titles\%YEAR%" --output-file translated_titles.jsonl
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" goto :fail

REM --------------------------------------------------------------------------
echo.
echo [stage 2/6] Lexical flood filter  ^(no model^)
pushd "%ROOT%2_filter_titles"
"%PY%" filter_flood_titles.py --input "%DATA%\translated titles\%YEAR%\translated_titles.jsonl" --output-dir "%DATA%\flood_events\%YEAR%"
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" goto :fail

REM --------------------------------------------------------------------------
echo.
echo [stage 3/6] Article translation  ^(the long one^)
pushd "%ROOT%3_translate_articles"
"%PY%" translate_flood_articles.py -c "%ROOT%3_translate_articles\config_%YEAR%.json"
set "RC=%ERRORLEVEL%"
popd
if "%RC%"=="130" goto :interrupted
if not "%RC%"=="0" goto :fail

REM --------------------------------------------------------------------------
echo.
echo [stage 4/6] Project to id/title/text
pushd "%ROOT%5_english_only"
"%PY%" extract_english.py %YEAR%
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" goto :fail

REM --------------------------------------------------------------------------
echo.
echo [stage 5/6] LLM extraction  ^(needs Ollama + qwen3:14b^)
pushd "%ROOT%6_extract_events"
"%PY%" extract.py --config "%ROOT%6_extract_events\config_%YEAR%.json"
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" goto :fail

REM --------------------------------------------------------------------------
echo.
echo [stage 6/6] Flatten to CSV
pushd "%ROOT%6_extract_events"
"%PY%" to_csv.py --config "%ROOT%6_extract_events\config_%YEAR%.json"
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" goto :fail

echo.
echo ============================================================
echo  [OK] %YEAR% complete.
echo   titles     : %DATA%\translated titles\%YEAR%\translated_titles.jsonl
echo   flood ids  : %DATA%\flood_events\%YEAR%\%YEAR%_MM.jsonl
echo   articles   : %DATA%\translated_articles\%YEAR%
echo   only eng   : %DATA%\only eng\%YEAR%
echo   extracted  : %DATA%\extracted\%YEAR%
echo   csv        : %DATA%\extracted\%YEAR%\extractions.csv
echo ============================================================
exit /b 0

:interrupted
echo.
echo [STOPPED] Interrupted at stage 3. Re-run this script to resume.
exit /b 130

:fail
echo.
echo [FAILED] Exit code %RC%. Nothing after this stage ran; re-run to resume.
exit /b %RC%
