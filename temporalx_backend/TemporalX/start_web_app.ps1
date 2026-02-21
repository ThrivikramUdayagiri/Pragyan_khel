# TemporalX Web Application Launcher (PowerShell)
# Starts the Streamlit web interface

Write-Host ""
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "                   TEMPORALX WEB APPLICATION" -ForegroundColor Green
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting web server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "The application will open in your default web browser." -ForegroundColor White
Write-Host ""
Write-Host "To stop the server, press Ctrl+C in this window." -ForegroundColor Gray
Write-Host ""
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

& "$PSScriptRoot\.venv\Scripts\streamlit.exe" run "$PSScriptRoot\web_app.py"
