param([string]$EnvFile = ".env")
$ErrorActionPreference = "Stop"
$key = Read-Host "OpenRouter API key"
if ([string]::IsNullOrWhiteSpace($key)) { throw "OPENROUTER_API_KEY cannot be empty." }
@"
# Local JEV-Benchmark credentials -- DO NOT COMMIT
OPENROUTER_API_KEY=$key
J05_JEV_MODEL=jev-latest
"@ | Set-Content -Path $EnvFile -Encoding UTF8
Write-Host "Created $EnvFile (gitignored)."
