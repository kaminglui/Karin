@echo off
:: setup.bat -- Install or uninstall KarinTTS.
::
:: Usage:
::   setup.bat install     Set up venv, deps, Task Scheduler auto-start
::   setup.bat uninstall   Remove task, stop KarinTTS, clean logs

setlocal enabledelayedexpansion

set TASK_NAME=KarinTTS
set GPT_SOVITS_COMMIT=2d9193b0d3c0eae0c3a14d8c68a839f1bae157dc
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%..\.."
set REPO_ROOT=%CD%
popd

set SOVITS_ROOT=%REPO_ROOT%\third_party\GPT-SoVITS
set VENV_DIR=%REPO_ROOT%\.venv\tts-server
set PYTHON=%VENV_DIR%\Scripts\python.exe
set START_BAT=%SCRIPT_DIR%start.bat
set SERVER_PY=%SCRIPT_DIR%tts_server.py

if /i "%~1"=="uninstall" goto :uninstall
if /i "%~1"=="install" goto :install
if /i "%~1"=="clean" goto :clean

echo Usage: setup.bat [install ^| uninstall ^| clean]
echo.
echo   install     Set up venv, deps, Task Scheduler auto-start, then start
echo   uninstall   Remove task, stop sidecar, clean logs
echo   clean       Nuclear: kill ANY pythonw running tts_server.py, free port 9880
exit /b 1

:: =====================================================================
:install
:: =====================================================================
echo.
echo ============================================================
echo   KarinTTS -- Install
echo ============================================================
echo   Repo: %REPO_ROOT%
echo.

where git >nul 2>&1 || goto :fail_git
where python >nul 2>&1 || goto :fail_python

:: 1. GPT-SoVITS source
if exist "%SOVITS_ROOT%\GPT_SoVITS\TTS_infer_pack\TTS.py" (
    echo   [OK]   GPT-SoVITS source
    pushd "%SOVITS_ROOT%"
    for /f "tokens=*" %%h in ('git rev-parse HEAD 2^>nul') do set SOVITS_HEAD=%%h
    popd
    rem Nested ifs (not chained) — chained "if defined ... if /i not ..."
    rem with delayed expansion inside an "if exist (...)" block trips
    rem cmd's parser. Also: must be rem not :: inside a paren block, because
    rem cmd parses :: as a label and chokes on punctuation in the comment text.
    if defined SOVITS_HEAD (
        if /i not "!SOVITS_HEAD!"=="%GPT_SOVITS_COMMIT%" (
            echo   [WARN] GPT-SoVITS is at !SOVITS_HEAD!, expected %GPT_SOVITS_COMMIT%.
            echo          Keeping the existing checkout. Delete third_party\GPT-SoVITS to reclone.
        )
    )
) else (
    echo   [----] Cloning GPT-SoVITS...
    git clone https://github.com/RVC-Boss/GPT-SoVITS.git "%SOVITS_ROOT%" 2>&1
    if !ERRORLEVEL! neq 0 goto :fail_clone
    pushd "%SOVITS_ROOT%"
    git checkout %GPT_SOVITS_COMMIT% -q 2>nul
    if !ERRORLEVEL! neq 0 (
        popd
        goto :fail_checkout
    )
    popd
    echo   [OK]   GPT-SoVITS cloned
)

:: 2. Python venv + deps
if exist "%PYTHON%" (
    echo   [OK]   Python venv
) else (
    echo   [----] Creating venv...
    python -m venv "%VENV_DIR%" 2>&1
    if not exist "%PYTHON%" goto :fail_venv
)

"%PYTHON%" -m pip install --upgrade pip -q 2>&1
if !ERRORLEVEL! neq 0 goto :fail_pip

"%PYTHON%" -c "import torch, torchaudio" >nul 2>&1
if !ERRORLEVEL! neq 0 (
    rem Round parens inside an echo, inside an `if (...) else (...)` block,
    rem confuse cmd's parser even when balanced — escape with ^ to be safe.
    echo   Installing PyTorch ^(CUDA 12.1^)...
    "%PYTHON%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 -q 2>&1
    if !ERRORLEVEL! neq 0 goto :fail_torch
) else (
    echo   [OK]   PyTorch already installed
)

if exist "%SOVITS_ROOT%\requirements.txt" (
    echo   Installing GPT-SoVITS deps...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content '%SOVITS_ROOT%\requirements.txt' | Where-Object { $_ -notmatch '^(torch|torchaudio|torchvision)(\s*([=<>~]=?|$)|\s*$)' } | Set-Content -Encoding ASCII '%TEMP%\karin-gpt-sovits-reqs.txt'"
    if !ERRORLEVEL! neq 0 goto :fail_sovits_deps
    "%PYTHON%" -m pip install -r "%TEMP%\karin-gpt-sovits-reqs.txt" -q 2>&1
    if !ERRORLEVEL! neq 0 goto :fail_sovits_deps
)

echo   Installing sidecar deps...
"%PYTHON%" -m pip install fastapi "uvicorn[standard]" pydantic soundfile numpy httpx pystray pillow faster-whisper -q 2>&1
if !ERRORLEVEL! neq 0 goto :fail_sidecar_deps
echo   [OK]   Dependencies installed (TTS + STT)

:: 3. Tailscale
set TS_IP=
for /f "tokens=*" %%i in ('tailscale ip -4 2^>nul') do if not defined TS_IP set TS_IP=%%i
if "!TS_IP!"=="" (
    echo   [FAIL] Tailscale not running. Download: https://tailscale.com/download
    pause & exit /b 1
)
echo   [OK]   Tailscale IP: !TS_IP!

:: 4. Voice models
set FOUND=0
for /r "%REPO_ROOT%\characters" %%f in (ref.wav) do set FOUND=1
if "!FOUND!"=="0" (
    for %%f in ("%REPO_ROOT%\voice_training\*_ref.wav") do if exist "%%~f" set FOUND=1
)
if "!FOUND!"=="1" (
    echo   [OK]   Voice models found
) else (
    echo   [WARN] No voice models under characters\*\voices\ or legacy voice_training\
)

:: 5. Task Scheduler
echo   Registering auto-start task...
schtasks /Create /TN "%TASK_NAME%" /TR "\"%START_BAT%\"" /SC ONLOGON /RL LIMITED /F >nul 2>&1
if !ERRORLEVEL! neq 0 goto :fail_task
echo   [OK]   Task "%TASK_NAME%" registered

:: 6. Start now?
echo.
set /p GO="   Start now? [Y/n] "
if /i "!GO!"=="n" goto :done
call "%START_BAT%"
if !ERRORLEVEL! neq 0 goto :fail_start

rem Post-launch verification — `start "" /b` returns immediately, but
rem the actual server takes ~20-40s to bind 9880 (TTS pipeline load).
rem Poll inside PowerShell so we don't depend on cmd's `timeout` (which
rem fails with "Input redirection is not supported" when this script
rem is invoked with a piped stdin).
echo   Waiting for sidecar to bind port 9880 (up to 60s)...
powershell -NoProfile -Command "for ($i=0; $i -lt 30; $i++) { if (Get-NetTCPConnection -LocalPort 9880 -State Listen -ErrorAction SilentlyContinue) { exit 0 }; Start-Sleep -Seconds 2 }; exit 1"
if !ERRORLEVEL! equ 0 (
    echo   [OK]   Sidecar bound to port 9880
    echo   Test it from another shell:
    echo     curl -s http://!TS_IP!:9880/stt/status
) else (
    echo   [WARN] Sidecar didn't bind 9880 within 60s.
    echo          Check the log:
    echo            type "%SCRIPT_DIR%tts_server.log"
    echo          OR launch in foreground to see errors:
    echo            "%START_BAT%" --visible
)
goto :done

:: =====================================================================
:uninstall
:: =====================================================================
echo.
echo ============================================================
echo   KarinTTS -- Uninstall
echo ============================================================
echo.

schtasks /Query /TN "%TASK_NAME%" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    schtasks /Delete /TN "%TASK_NAME%" /F >nul 2>&1
    echo   [OK]   Task removed
) else (
    echo   [OK]   Task not found
)

call :stop_server

if exist "%SCRIPT_DIR%voice_server.log" del /f "%SCRIPT_DIR%voice_server.log" 2>nul
if exist "%SCRIPT_DIR%voice_server.2.log" del /f "%SCRIPT_DIR%voice_server.2.log" 2>nul
if exist "%SCRIPT_DIR%tts_server.log" del /f "%SCRIPT_DIR%tts_server.log" 2>nul
if exist "%SCRIPT_DIR%tts_server.2.log" del /f "%SCRIPT_DIR%tts_server.2.log" 2>nul
echo   [OK]   Logs cleaned

goto :done

:: =====================================================================
:clean
:: =====================================================================
echo.
echo ============================================================
echo   KarinTTS -- Force Clean
echo ============================================================
echo.
echo   Killing ANY pythonw running tts_server.py...
call :stop_server
echo.
echo   Checking port 9880...
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 9880 -State Listen -ErrorAction SilentlyContinue; if ($c) { Write-Output ('   [WARN] still bound by PID ' + $c.OwningProcess + ' — try elevating to Administrator and re-running clean'); exit 1 } else { Write-Output '   [OK]   port 9880 is free'; exit 0 }"
echo.
goto :done

:done
echo.
echo ============================================================
echo   Done.
echo ============================================================
pause
exit /b 0

:stop_server
:: Restrict the match to python(w).exe — the PowerShell process running
:: this very check has the script name in its own CommandLine and would
:: match itself, then Stop-Process would target the running PowerShell.
:: Match by basename "tts_server.py" rather than full SERVER_PY path so
:: we also catch instances launched with a relative path
:: ("python ../../deploy/pc-tts/tts_server.py") that the full-path
:: filter would miss. Same fix shape as start.bat::is_running.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$procs = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'python.exe','pythonw.exe' -and $_.CommandLine -and $_.CommandLine.Contains('tts_server.py') }); foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force }; if ($procs.Count -gt 0) { exit 0 } else { exit 2 }" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo   [OK]   Stopped KarinTTS process
) else (
    echo   [OK]   No KarinTTS process running
)
exit /b 0

:fail_git
set "FAIL_MSG=git not found. Install Git for Windows and retry."
goto :fail

:fail_python
set "FAIL_MSG=python not found. Install Python from https://www.python.org/downloads/ and retry."
goto :fail

:fail_clone
set "FAIL_MSG=GPT-SoVITS clone failed."
goto :fail

:fail_checkout
set "FAIL_MSG=GPT-SoVITS checkout failed."
goto :fail

:fail_venv
set "FAIL_MSG=venv creation failed."
goto :fail

:fail_pip
set "FAIL_MSG=pip upgrade failed."
goto :fail

:fail_torch
set "FAIL_MSG=PyTorch install failed."
goto :fail

:fail_sovits_deps
set "FAIL_MSG=GPT-SoVITS dependency install failed."
goto :fail

:fail_sidecar_deps
set "FAIL_MSG=Sidecar dependency install failed."
goto :fail

:fail_task
set "FAIL_MSG=Run as Administrator to register the task."
goto :fail

:fail_start
set "FAIL_MSG=start.bat failed."
goto :fail

:fail
echo   [FAIL] %FAIL_MSG%
pause
exit /b 1
