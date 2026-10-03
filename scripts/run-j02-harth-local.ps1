param(
  [string]$DatasetRoot = "D:\Downloads\harth\harth",
  [string]$OutputDir = "results\j02-harth"
)

$ErrorActionPreference = "Stop"

$Subjects = @("S006","S008","S009","S010","S012","S013","S014","S015","S016","S017","S018","S019","S020","S021","S022","S023","S024","S025","S026","S027","S028","S029")

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "=== J02 HARTH validation ==="
jev-bench harth-validate --dataset-root $DatasetRoot --output "$OutputDir\validation.json"

Write-Host "=== J02 HARTH window inspection: S006 ==="
jev-bench harth-inspect --dataset-root $DatasetRoot --subject S006 --window-size 128 --stride 128 --max-windows 1000 | Tee-Object "$OutputDir\inspect-S006.json"

Write-Host "=== J02 HARTH smoke LOSO: all 22 subjects ==="
jev-bench harth-loso --dataset-root $DatasetRoot --model tiny_mlp --hidden-units 8 --subjects ($Subjects -join ",") --window-size 128 --stride 128 --max-train-windows-per-subject 500 --max-test-windows 2000 --epochs 10 --lr 0.01 --batch-size 128 --seed 0 --abstain-threshold 0 --output "$OutputDir\loso-tiny-mlp-seed0.json"

Write-Host "=== J02 HARTH smoke complete ==="
Write-Host "Results: $OutputDir"
