@echo off
REM TemporalX Visualizer Launcher
REM Activates virtual environment and runs visualizer with arguments

call .venv\Scripts\activate.bat
python visualizer.py %*
