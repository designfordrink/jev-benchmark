param([string]$EnvFile = ".env")
$ErrorActionPreference = "Stop"
$secure = Read-Host "OpenRouter API key" -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try { $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) } finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
if ([string]::IsNullOrWhiteSpace($key)) { throw "OPENROUTER_API_KEY cannot be empty." }
@"
# Local JEV-Benchmark credentials -- DO NOT COMMIT
OPENROUTER_API_KEY=$key
J05_JEV_MODEL=jev-latest
"@ | Set-Content -Path $EnvFile -Encoding UTF8
Write-Host "Created $EnvFile (gitignored)."
