@echo off
setlocal
if exist .venv rmdir /s /q .venv
python -m venv .venv
if errorlevel 1 exit /b %errorlevel%
call .venv\Scripts\activate.bat
python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b %errorlevel%
pip install -r requirements.txt
exit /b %errorlevel%
