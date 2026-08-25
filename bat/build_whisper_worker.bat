@echo off
setlocal
set ROOT=%~dp0..\
set STAGE=%ROOT%native\whisper_cpp_worker\stage\whisper_cpp
cmake -S "%ROOT%native\whisper_cpp_worker" -B "%ROOT%native\whisper_cpp_worker\build" -A x64 -DGGML_VULKAN=ON
if errorlevel 1 exit /b %errorlevel%
cmake --build "%ROOT%native\whisper_cpp_worker\build" --config Release --target vrct-whisper-worker
if errorlevel 1 exit /b %errorlevel%
if exist "%STAGE%" rmdir /s /q "%STAGE%"
mkdir "%STAGE%"
for /r "%ROOT%native\whisper_cpp_worker\build" %%F in (vrct-whisper-worker.exe) do if exist "%%~fF" copy /y "%%~fF" "%STAGE%\" >nul
if not exist "%STAGE%\vrct-whisper-worker.exe" exit /b 1
echo Staged worker under %STAGE%
exit /b 0
