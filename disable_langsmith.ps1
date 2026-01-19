# PowerShell script to permanently disable LangSmith tracing
# Run this script as Administrator to set system-wide environment variables

Write-Host "Disabling LangSmith tracing to prevent 403 errors..." -ForegroundColor Yellow

# Set environment variables for current session
$env:LANGCHAIN_TRACING_V2 = "false"
$env:LANGCHAIN_API_KEY = ""

# Set environment variables permanently for current user
[Environment]::SetEnvironmentVariable("LANGCHAIN_TRACING_V2", "false", "User")
[Environment]::SetEnvironmentVariable("LANGCHAIN_API_KEY", "", "User")

Write-Host "✅ LangSmith tracing has been disabled!" -ForegroundColor Green
Write-Host "Environment variables set:" -ForegroundColor Cyan
Write-Host "  LANGCHAIN_TRACING_V2 = false" -ForegroundColor White
Write-Host "  LANGCHAIN_API_KEY = (empty)" -ForegroundColor White
Write-Host ""
Write-Host "You may need to restart your terminal/IDE for changes to take effect." -ForegroundColor Yellow
