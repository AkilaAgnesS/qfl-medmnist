# Revision runner (SUSCOM major revision, due 2026-08-10).
# Covers the reviewer-requested experiments in fastest-first order:
#   R1: 10 seeds, qubit ablation, backend-calibrated noise, CodeCarbon energy
#   R2: finite-shot evaluation (part of eval_backend_noise.py)
#
# Prerequisites (once):
#   pip install codecarbon pennylane-qiskit qiskit-aer qiskit-ibm-runtime
#
# Usage:
#   .\experiments\revision_overnight.ps1
#
# Notes:
# - Seed extension REUSES existing seed 0-2 results (they are not rerun),
#   except hybrid centralized baselines, which are rerun on all 10 seeds
#   because the old runs saved no checkpoint.pt (needed for Phase E).
#   Runs are deterministic per seed, so reruns reproduce the paper numbers.
# - Every run uses --codecarbon: emissions.csv lands in each results dir.
# - Estimated wall-clock on CPU: dominated by hybrid runs; run C3 phases
#   overnight and monitor the fast classical phases first.

$ErrorActionPreference = "Continue"
$LogFile = "results\revision_log.txt"
New-Item -ItemType Directory -Force -Path "results" | Out-Null
"=== Revision run started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $LogFile

function Run-Step($Script, $ConfigPath, $Seeds, $Label) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "[$Label] $ConfigPath  (seeds: $Seeds)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    "[$Label] $(Get-Date -Format 'HH:mm:ss') START $ConfigPath seeds=$Seeds" | Out-File $LogFile -Append
    $start = Get-Date
    if ($Seeds) {
        python $Script --config $ConfigPath --codecarbon --seeds $Seeds.Split(" ")
    } else {
        python $Script --config $ConfigPath --codecarbon
    }
    $elapsed = ((Get-Date) - $start).TotalMinutes
    "[$Label] $(Get-Date -Format 'HH:mm:ss') END   $ConfigPath ({0:N1} min)" -f $elapsed | Out-File $LogFile -Append
    Write-Host ("[$Label] done in {0:N1} min" -f $elapsed) -ForegroundColor Green
}

$FL  = "experiments\run_federated.py"
$CEN = "experiments\run_centralized.py"

# Seeds to ADD to existing 0-2 results (breast dirichlet05 already has 0-4).
$Ext   = "3 4 5 6 7 8 9"
$Ext05 = "5 6 7 8 9"

# ---- Phase A: classical FL 10-seed extension (fast) -----------------------
foreach ($cs in @("C1", "C2")) {
    foreach ($ds in @("breast", "pneumonia", "derma")) {
        Run-Step $FL "experiments\configs\fl_${cs}_${ds}.yaml"             $Ext "A-$cs-$ds-iid"
        Run-Step $FL "experiments\configs\fl_${cs}_${ds}_dirichlet01.yaml" $Ext "A-$cs-$ds-d01"
        $seeds = if ($ds -eq "breast") { $Ext05 } else { $Ext }
        Run-Step $FL "experiments\configs\fl_${cs}_${ds}_dirichlet05.yaml" $seeds "A-$cs-$ds-d05"
    }
}

# ---- Phase B: classical centralized 10-seed extension (fast) --------------
foreach ($cs in @("C1", "C2")) {
    foreach ($ds in @("breast", "pneumonia", "derma")) {
        Run-Step $CEN "experiments\configs\baseline_${cs}_${ds}.yaml" $Ext "B-$cs-$ds"
    }
}

# ---- Phase C: hybrid centralized - FULL 10-seed rerun (checkpoints!) ------
# Old seed 0-2 runs saved no checkpoint.pt; rerun all seeds so Phase E works.
foreach ($ds in @("breast", "pneumonia", "derma")) {
    Run-Step $CEN "experiments\configs\baseline_C3_${ds}.yaml" "0 1 2 3 4 5 6 7 8 9" "C-C3-$ds"
}

# ---- Phase D: qubit/bottleneck ablation (R1 comment 3) --------------------
foreach ($q in @("q4", "q6", "q12")) {
    Run-Step $CEN "experiments\configs\revision\ablation_C3_breast_$q.yaml" "" "D-breast-$q"
    Run-Step $CEN "experiments\configs\revision\ablation_C3_derma_$q.yaml"  "" "D-derma-$q"
}

# ---- Phase E: backend-calibrated noise evaluation (R1 comment 2) ----------
# Uses checkpoints from Phase C. Fake 16-qubit calibration snapshot by default;
# pass --backend ibm_<name> manually if an IBM Quantum token is configured.
foreach ($ds in @("breast", "pneumonia", "derma")) {
    Write-Host "`n[E-$ds] backend-noise eval" -ForegroundColor Cyan
    "[E-$ds] $(Get-Date -Format 'HH:mm:ss') START backend eval" | Out-File $LogFile -Append
    python experiments\eval_backend_noise.py --config "experiments\configs\baseline_C3_${ds}.yaml" --seeds 0 1 2 3 4 5 6 7 8 9
    "[E-$ds] $(Get-Date -Format 'HH:mm:ss') END backend eval" | Out-File $LogFile -Append
}

# ---- Phase F: hybrid FL 10-seed extension (slowest - can run separately) --
foreach ($ds in @("breast", "pneumonia", "derma")) {
    Run-Step $FL "experiments\configs\fl_C3_${ds}.yaml"             $Ext "F-C3-$ds-iid"
    Run-Step $FL "experiments\configs\fl_C3_${ds}_dirichlet01.yaml" $Ext "F-C3-$ds-d01"
    $seeds = if ($ds -eq "breast") { $Ext05 } else { $Ext }
    Run-Step $FL "experiments\configs\fl_C3_${ds}_dirichlet05.yaml" $seeds "F-C3-$ds-d05"
}

"=== Revision run finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $LogFile -Append
Write-Host "`nAll revision experiments finished. Next:" -ForegroundColor Green
Write-Host "    python notebooks\03_susqa_analysis.py" -ForegroundColor Yellow
