@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>&1
if not errorlevel 1 (
    py -3 app.py
    if errorlevel 1 goto failed
    exit /b 0
)
where python >nul 2>&1
if not errorlevel 1 (
    python app.py
    if errorlevel 1 goto failed
    exit /b 0
)
:failed
echo.
echo The application could not start. Read any error above.
echo Extract the entire ZIP and install Python 3.10 or newer with Tcl/Tk.
echo You can also open a terminal in this folder and run: python app.py
pause
exit /b 1
