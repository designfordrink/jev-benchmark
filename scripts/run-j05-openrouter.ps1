param([string]$EnvFile = ".env",[string]$Model = "",[int[]]$Seeds = @(0,1,2,3,4),[int]$Episodes = 100)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $EnvFile)) { throw "Missing $EnvFile. Run .\scripts\setup-openrouter.ps1" }
Get-Content $EnvFile | ForEach-Object { $line=$_.Trim(); if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) { $name,$value=$line.Split("=",2); [Environment]::SetEnvironmentVariable($name.Trim(),$value.Trim(),"Process") } }
if ([string]::IsNullOrWhiteSpace($env:OPENROUTER_API_KEY)) { throw "OPENROUTER_API_KEY is missing." }
if ([string]::IsNullOrWhiteSpace($Model)) { $Model=$env:J05_JEV_MODEL }
if ([string]::IsNullOrWhiteSpace($Model)) { $Model="jev-latest" }
$seedArg=$Seeds -join ","
New-Item -ItemType Directory -Force -Path "results/j05" | Out-Null
Write-Host "Running J05 with OpenRouter model: $Model; seeds: $seedArg; episodes/seed: $Episodes"
jev-bench j05-protocol --provider jev --model $Model --cache .cache/j05-jev.json --seeds $seedArg --episodes $Episodes --labels-output results/j05/labels-v1.json | Tee-Object -FilePath results/j05/protocol-v1.json
