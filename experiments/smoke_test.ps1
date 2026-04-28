# PowerShell driver for the end-to-end smoke test.
# Runs all three case studies on BreastMNIST with tiny configs, then
# aggregates the SUSQA reports into a single CSV for inspection.
#
# Run from repo root with venv activated:
#   .\experiments\smoke_test.ps1

$ErrorActionPreference = "Stop"

Write-Host "== C1 classical CNN ==" -ForegroundColor Cyan
python experiments/run_centralized.py --config experiments/configs/smoke_test_C1.yaml

Write-Host "`n== C2 classical compressed ==" -ForegroundColor Cyan
python experiments/run_centralized.py --config experiments/configs/smoke_test_C2.yaml

Write-Host "`n== C3 hybrid quantum (slow) ==" -ForegroundColor Cyan
python experiments/run_centralized.py --config experiments/configs/smoke_test_C3.yaml

Write-Host "`n== Aggregating SUSQA reports ==" -ForegroundColor Cyan
python -c "from susqa import aggregate_reports; import json; rows = aggregate_reports('results'); print(json.dumps(rows, indent=2))"

Write-Host "`nSmoke test complete. JSON reports under results\<experiment_id>\susqa_report.json" -ForegroundColor Green
