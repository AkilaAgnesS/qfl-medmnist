# Overnight runner — executes all remaining experiments in fastest-first order.
# Each run logs to results/<experiment_id>/ as usual, plus a combined log file
# at results/overnight_log.txt for post-mortem if anything errors.
#
# Suggested usage:
#   .\experiments\overnight.ps1
#
# Total estimated wall-clock on CPU: 30-40 hours (dominated by C3 hybrid runs).
# Configs are ordered fastest -> slowest so you get early signal on bugs and
# can monitor progress; if you ctrl-C halfway through, classical results are
# already saved and the hybrid runs can resume by re-launching the script
# (skipping any configs whose results already exist is left as a manual step).

$ErrorActionPreference = "Continue"   # don't abort the whole script if one config errors
$LogFile = "results\overnight_log.txt"
New-Item -ItemType Directory -Force -Path "results" | Out-Null
"=== Overnight run started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $LogFile

function Run-Config($ConfigPath, $Label) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "[$Label] $ConfigPath" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    "[$Label] $(Get-Date -Format 'HH:mm:ss') START $ConfigPath" | Out-File $LogFile -Append
    $start = Get-Date
    python experiments\run_federated.py --config $ConfigPath
    $elapsed = ((Get-Date) - $start).TotalMinutes
    "[$Label] $(Get-Date -Format 'HH:mm:ss') END   $ConfigPath ({0:N1} min)" -f $elapsed | Out-File $LogFile -Append
    Write-Host ("[$Label] done in {0:N1} min" -f $elapsed) -ForegroundColor Green
}

# ---- Phase A: BreastMNIST 5-seed bumps at alpha=0.5 (fast classical first) ----
Run-Config "experiments\configs\fl_C1_breast_dirichlet05.yaml" "A1/3"
Run-Config "experiments\configs\fl_C2_breast_dirichlet05.yaml" "A2/3"

# ---- Phase B: Pneumonia classical FL (fast) ----
Run-Config "experiments\configs\fl_C1_pneumonia.yaml"             "B1/6"
Run-Config "experiments\configs\fl_C2_pneumonia.yaml"             "B2/6"
Run-Config "experiments\configs\fl_C1_pneumonia_dirichlet05.yaml" "B3/6"
Run-Config "experiments\configs\fl_C2_pneumonia_dirichlet05.yaml" "B4/6"
Run-Config "experiments\configs\fl_C1_pneumonia_dirichlet01.yaml" "B5/6"
Run-Config "experiments\configs\fl_C2_pneumonia_dirichlet01.yaml" "B6/6"

# ---- Phase C: Pneumonia centralized baselines (medium speed) ----
Run-Config "experiments\configs\baseline_C1_pneumonia.yaml" "C1/3"
Run-Config "experiments\configs\baseline_C2_pneumonia.yaml" "C2/3"

# ---- Phase D: BreastMNIST hybrid 5-seed bump (slow) ----
Run-Config "experiments\configs\fl_C3_breast_dirichlet05.yaml" "A3/3"

# ---- Phase E: Pneumonia hybrid (slowest — these are the long ones) ----
Run-Config "experiments\configs\baseline_C3_pneumonia.yaml"      "C3/3"
Run-Config "experiments\configs\fl_C3_pneumonia.yaml"            "E1/3"
Run-Config "experiments\configs\fl_C3_pneumonia_dirichlet05.yaml" "E2/3"
Run-Config "experiments\configs\fl_C3_pneumonia_dirichlet01.yaml" "E3/3"

"=== Overnight run finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $LogFile -Append
Write-Host "`nAll configs finished. Run analysis:" -ForegroundColor Green
Write-Host "    python notebooks\03_susqa_analysis.py" -ForegroundColor Yellow
