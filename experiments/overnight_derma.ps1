# DermaMNIST overnight runner. Twelve configs total: 3 case studies x 4 settings.
# Order: smoke -> classical -> hybrid (slowest last). Logs to results/derma_log.txt.
#
# Estimated total wall-clock on CPU: 25-30 hours, dominated by C3 hybrid runs.
#
# Suggested usage:
#   .\experiments\overnight_derma.ps1
# Or test individual config first:
#   python experiments\run_centralized.py --config experiments\configs\baseline_C2_derma.yaml --device cpu

$ErrorActionPreference = "Continue"
$LogFile = "results\derma_log.txt"
New-Item -ItemType Directory -Force -Path "results" | Out-Null
"=== DermaMNIST run started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $LogFile

function Run-Centralized($ConfigPath, $Label) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "[$Label] $ConfigPath (centralized)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    "[$Label] $(Get-Date -Format 'HH:mm:ss') START $ConfigPath" | Out-File $LogFile -Append
    $start = Get-Date
    python experiments\run_centralized.py --config $ConfigPath
    $elapsed = ((Get-Date) - $start).TotalMinutes
    "[$Label] $(Get-Date -Format 'HH:mm:ss') END   $ConfigPath ({0:N1} min)" -f $elapsed | Out-File $LogFile -Append
    Write-Host ("[$Label] done in {0:N1} min" -f $elapsed) -ForegroundColor Green
}

function Run-Federated($ConfigPath, $Label) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "[$Label] $ConfigPath (federated)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    "[$Label] $(Get-Date -Format 'HH:mm:ss') START $ConfigPath" | Out-File $LogFile -Append
    $start = Get-Date
    python experiments\run_federated.py --config $ConfigPath
    $elapsed = ((Get-Date) - $start).TotalMinutes
    "[$Label] $(Get-Date -Format 'HH:mm:ss') END   $ConfigPath ({0:N1} min)" -f $elapsed | Out-File $LogFile -Append
    Write-Host ("[$Label] done in {0:N1} min" -f $elapsed) -ForegroundColor Green
}

# ---- Phase A: classical centralized baselines (fast) ----
Run-Centralized "experiments\configs\baseline_C1_derma.yaml" "A1/2"
Run-Centralized "experiments\configs\baseline_C2_derma.yaml" "A2/2"

# ---- Phase B: classical FL (fast) ----
Run-Federated "experiments\configs\fl_C1_derma.yaml"             "B1/6"
Run-Federated "experiments\configs\fl_C2_derma.yaml"             "B2/6"
Run-Federated "experiments\configs\fl_C1_derma_dirichlet05.yaml" "B3/6"
Run-Federated "experiments\configs\fl_C2_derma_dirichlet05.yaml" "B4/6"
Run-Federated "experiments\configs\fl_C1_derma_dirichlet01.yaml" "B5/6"
Run-Federated "experiments\configs\fl_C2_derma_dirichlet01.yaml" "B6/6"

# ---- Phase C: hybrid centralized (slow) ----
Run-Centralized "experiments\configs\baseline_C3_derma.yaml" "C1/1"

# ---- Phase D: hybrid FL (slowest) ----
Run-Federated "experiments\configs\fl_C3_derma.yaml"             "D1/3"
Run-Federated "experiments\configs\fl_C3_derma_dirichlet05.yaml" "D2/3"
Run-Federated "experiments\configs\fl_C3_derma_dirichlet01.yaml" "D3/3"

"=== DermaMNIST run finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $LogFile -Append
Write-Host "`nAll DermaMNIST configs finished. Run analysis:" -ForegroundColor Green
Write-Host "    python notebooks\03_susqa_analysis.py" -ForegroundColor Yellow
Write-Host "    python notebooks\06_effect_sizes.py" -ForegroundColor Yellow
