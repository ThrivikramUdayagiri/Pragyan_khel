@echo off
REM TemporalX CLI Launcher
REM Activates virtual environment and runs CLI with arguments

call .venv\Scripts\activate.bat
python cli.py %*
