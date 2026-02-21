@echo off
REM TemporalX Examples Launcher
REM Activates virtual environment and runs examples

echo Starting TemporalX Examples...
echo.

call .venv\Scripts\activate.bat
python examples.py

pause
