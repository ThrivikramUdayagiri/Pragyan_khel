@echo off
REM TemporalX Web Application Launcher
REM Starts the Streamlit web interface

echo.
echo ====================================================================
echo                     TEMPORALX WEB APPLICATION
echo ====================================================================
echo.
echo Starting web server...
echo.
echo The application will open in your default web browser.
echo.
echo To stop the server, press Ctrl+C in this window.
echo.
echo ====================================================================
echo.

call .venv\Scripts\activate.bat
streamlit run web_app.py

pause
