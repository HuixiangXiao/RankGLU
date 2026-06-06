param(
    [string]$Python = "python",
    [string]$Gpu = "0",
    [string]$Universe = "csi300",
    [string]$Sections = "main,ablation,diagnostic"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutDir = Join-Path $Root "results\$($Universe)_multiseed_$Timestamp"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "Repository root: $Root"
Write-Host "Output dir: $OutDir"
Write-Host "GPU: $Gpu"
Write-Host "Universe: $Universe"
Write-Host "Sections: $Sections"

Push-Location $Root
try {
    & $Python "scripts\run_multiseed_protocol.py" `
        --universe $Universe `
        --prefix "opensource" `
        --gpu $Gpu `
        --sections $Sections `
        --out-dir $OutDir 2>&1 | Tee-Object -FilePath (Join-Path $OutDir "console_output.txt")

    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        Write-Host "Experiment runner failed with exit code $ExitCode"
        exit $ExitCode
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "All requested runs finished."
Write-Host "Please send back this folder:"
Write-Host $OutDir
Write-Host ""
Write-Host "Most important summary file:"
Write-Host (Join-Path $OutDir "final_results.txt")
