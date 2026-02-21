# TemporalX CLI Launcher (PowerShell)
# Runs CLI using virtual environment Python with all arguments

& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\cli.py" $args
