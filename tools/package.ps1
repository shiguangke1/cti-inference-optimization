# Package a submission from online_best/ with the required files at ZIP root.
# Usage: .\tools\package.ps1 online_best

param([string]$Label = "online_best")

$RepoRoot = Split-Path -Parent $PSScriptRoot
$SourceDir = Join-Path $RepoRoot "online_best"
$OutputDir = Join-Path $RepoRoot "dist"
$Date = Get-Date -Format "yyyyMMdd"
$OutputPath = Join-Path $OutputDir "submit_${Label}_${Date}.zip"

$RequiredFiles = @("infer.py", "build_env.sh", "requirements.txt")
foreach ($File in $RequiredFiles) {
    $Candidate = Join-Path $SourceDir $File
    if (-not (Test-Path -LiteralPath $Candidate)) {
        throw "Missing required submission file: $Candidate"
    }
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

Push-Location $SourceDir
try {
    Compress-Archive -LiteralPath $RequiredFiles -DestinationPath $OutputPath -Force
} finally {
    Pop-Location
}

$SizeKB = (Get-Item -LiteralPath $OutputPath).Length / 1KB
Write-Host "[package] $OutputPath ($([math]::Round($SizeKB, 1)) KB)"
