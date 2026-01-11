# Quick start script for RLM Dashboard
# Run: .\start_dashboard.ps1

Write-Host "================================" -ForegroundColor Cyan
Write-Host " Starting RLM Dashboard" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "[WARNING] Virtual environment not activated" -ForegroundColor Yellow
    Write-Host "Activating venv..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
}

# Check if Flask is installed
try {
    python -c "import flask" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[INSTALL] Installing dashboard dependencies..." -ForegroundColor Green
        pip install -r web_dashboard/requirements.txt
    }
} catch {
    Write-Host "[INSTALL] Installing dashboard dependencies..." -ForegroundColor Green
    pip install -r web_dashboard/requirements.txt
}

# Start the dashboard
Write-Host ""
Write-Host "[START] Launching dashboard..." -ForegroundColor Green
Write-Host ""
Write-Host "Dashboard will open at: http://localhost:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

Set-Location web_dashboard
python app.py

