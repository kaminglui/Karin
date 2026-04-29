@echo off
:: start.bat -- Launch KarinTTS.
::
:: Double-click: starts hidden with tray icon (pythonw).
:: start.bat --visible: starts in a visible console (for debugging).
::
:: Stop: right-click tray icon > Quit, or run setup.bat uninstall.

setlocal

set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%..\.."
set REPO_ROOT=%CD%
popd

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if "%KARIN_VOICE_CPU_THREADS%"=="" set KARIN_VOICE_CPU_THREADS=4
if "%KARIN_STT_NUM_WORKERS%"=="" set KARIN_STT_NUM_WORKERS=1
if "%KARIN_TTS_MAX_CHARS%"=="" set KARIN_TTS_MAX_CHARS=1000
if "%KARIN_STT_MAX_SECONDS%"=="" set KARIN_STT_MAX_SECONDS=30

rem Resource caps for parallel libraries. OMP/MKL throttle BLAS thread
rem pools that ctranslate2 + torch use under the hood. LOKY/JOBLIB caps
rem are best-effort — empirical testing showed they don't fully prevent
rem the dual-pythonw situation (one venv parent + one global child via
rem sys._base_executable per popen_loky_win32.py), but they're harmless
rem and may help if a future dep changes its spawn pattern. The dual
rem process is benign: setup.bat's :stop_server matches by basename so
rem cleanup catches both.
set LOKY_MAX_CPU_COUNT=1
set JOBLIB_MULTIPROCESSING=0
set OMP_NUM_THREADS=%KARIN_VOICE_CPU_THREADS%
set MKL_NUM_THREADS=%KARIN_VOICE_CPU_THREADS%
set SERVER=%SCRIPT_DIR%tts_server.py
set PYTHON_EXE=%REPO_ROOT%\.venv\tts-server\Scripts\python.exe
set PYTHONW_EXE=%REPO_ROOT%\.venv\tts-server\Scripts\pythonw.exe

if not exist "%PYTHON_EXE%" (
    echo KarinTTS venv not found. Run setup.bat install first.
    if /i "%~1"=="--visible" pause
    exit /b 1
)

call :is_running
if %ERRORLEVEL% equ 0 (
    echo KarinTTS is already running.
    if /i "%~1"=="--visible" pause
    exit /b 0
)

if /i "%~1"=="--visible" (
    echo Starting KarinTTS in console mode...
    cd /d "%REPO_ROOT%\third_party\GPT-SoVITS"
    "%PYTHON_EXE%" "%SERVER%" --tray
    pause
    exit /b %ERRORLEVEL%
)

rem Hidden launch: must cd to the GPT-SoVITS dir so tts_server.py
rem resolves its config paths the same way it does in --visible mode.
rem Then explicit `exit /b 0`: `start "" /b` doesn't reliably update
rem ERRORLEVEL on a successful detached launch, so without this the
rem stale 1 from `call :is_running` bleeds through to the caller.
cd /d "%REPO_ROOT%\third_party\GPT-SoVITS"
start "" /b "%PYTHONW_EXE%" "%SERVER%"
exit /b 0

:is_running
:: Restrict the match to python(w).exe (the PowerShell running this
:: check has the script name in its own CommandLine — without the Name
:: filter it would match itself and report a false positive). Match by
:: basename "tts_server.py" not full SERVER path so we also catch
:: instances launched with a relative path.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$procs = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'python.exe','pythonw.exe' -and $_.CommandLine -and $_.CommandLine.Contains('tts_server.py') }); if ($procs.Count -gt 0) { exit 0 } else { exit 1 }" >nul 2>&1
exit /b %ERRORLEVEL%
