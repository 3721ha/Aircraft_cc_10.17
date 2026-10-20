param([switch]$PreflightOnly)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$pythonPath = Join-Path $projectRoot 'Aircraft_mumt\.venv\Scripts\python.exe'
$sourceRun = 'runs/model_optimization_v1/window_diverse_event_learning_k4_w8a10_v1'
$extensionRun = 'runs/model_optimization_v1/window_diverse_event_extension_k4_w8a10_v1'
$trainingRun = 'runs/model_optimization_v1/window_diverse_event_learning_k4_w8a10_v2'
$extensionArgs = @(
    '_extend_arbitration_event_bank.py',
    '--source-run', $sourceRun,
    '--output', $extensionRun,
    '--heldout-seeds', ((185016..185047) -join ','),
    '--collection-seed', '20261023'
)
$auditArgs = @(
    '_audit_arbitration_learning.py',
    '--checkpoint', 'runs/model_optimization_v1/residual_pg_k4_w8a10_v1/curr_k4_best.pt',
    '--expected-sha256', '38815f375df492ee320d9a0d4a9105efaf3944a6694165b295f646c9917aea18',
    '--event-bank', "$extensionRun/event_bank.json",
    '--output', $trainingRun,
    '--train-seeds', ((184000..184063) -join ','),
    '--heldout-seeds', ((185000..185047) -join ','),
    '--max-events-per-split', '256',
    '--max-events-per-episode', '4',
    '--min-event-gap', '24',
    '--episode-balanced',
    '--min-train-event-episodes', '16',
    '--min-heldout-event-episodes', '8',
    '--cf-bid-horizon', '8',
    '--conflict-exposure', '1',
    '--updates', '48',
    '--events-per-update', '32',
    '--epochs', '2',
    '--learning-rate', '0.0001',
    '--utility-scale', '0.1',
    '--mc-samples', '4096',
    '--training-seed', '20261021',
    '--repair-seed', '20261021'
)
if ($PreflightOnly) {
    $extensionArgs += '--preflight-only'
    $auditArgs += '--preflight-only'
}
Push-Location $projectRoot
try {
    & $pythonPath @extensionArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Held-out collection failed with code $LASTEXITCODE. Original data has not been overwritten."
    }
    & $pythonPath @auditArgs
    $auditCode = $LASTEXITCODE
    if ($auditCode -eq 2) {
        $reportPath = Join-Path $projectRoot "$trainingRun/learning_audit.json"
        $coverageReport = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        if ($coverageReport.status -ne 'insufficient_independent_event_coverage') {
            throw "Unexpected audit exit code 2; inspect $reportPath"
        }
        Write-Warning ("Stopped before training: informative episodes train={0}/16, heldout={1}/8. All collected data is saved. Report: {2}" -f `
            $coverageReport.coverage.train.informative_episodes, `
            $coverageReport.coverage.heldout.informative_episodes, $reportPath)
        return
    }
    if ($auditCode -ne 0) {
        throw "Audit failed with code $auditCode. Inspect $trainingRun."
    }
}
finally {
    Pop-Location
}
