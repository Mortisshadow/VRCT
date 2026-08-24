@echo off
call "%~dp0build_whisper_worker.bat"
if errorlevel 1 exit /b %errorlevel%
call .venv_cuda/Scripts/activate
if "%VRCT_PYINSTALLER_CLEAN%"=="1" (
    pyinstaller spec/backend_cuda.spec --distpath src-tauri/bin --clean --noconfirm --log-level ERROR
) else (
    pyinstaller spec/backend_cuda.spec --distpath src-tauri/bin --noconfirm --log-level ERROR
)
