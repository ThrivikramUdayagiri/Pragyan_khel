@echo off
REM TemporalX Webcam Detector Launcher
REM Activates virtual environment and runs webcam detector with arguments

call .venv\Scripts\activate.bat
python webcam_detector.py %*
