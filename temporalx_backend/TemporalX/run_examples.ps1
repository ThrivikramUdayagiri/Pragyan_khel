# TemporalX Examples Launcher (PowerShell)
# Runs examples using virtual environment Python

Write-Host "Starting TemporalX Examples..." -ForegroundColor Green
Write-Host ""

& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\examples.py"

Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
