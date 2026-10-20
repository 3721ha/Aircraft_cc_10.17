param([switch]$PreflightOnly)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$pythonPath = Join-Path $projectRoot 'Aircraft_mumt\.venv\Scripts\python.exe'
$trainingSeeds = (184000..184063) -join ','
$heldoutSeeds = (185000..185015) -join ','
$auditArgs = @(
    '_audit_arbitration_learning.py',
    '--checkpoint', 'runs/model_optimization_v1/residual_pg_k4_w8a10_v1/curr_k4_best.pt',
    '--expected-sha256', '38815f375df492ee320d9a0d4a9105efaf3944a6694165b295f646c9917aea18',
    '--output', 'runs/model_optimization_v1/window_diverse_event_learning_k4_w8a10_v1',
    '--train-seeds', $trainingSeeds,
    '--heldout-seeds', $heldoutSeeds,
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
if ($PreflightOnly) { $auditArgs += '--preflight-only' }
Push-Location $projectRoot
try {
    & $pythonPath @auditArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Audit exited with code $LASTEXITCODE. Inspect learning_audit.json; insufficient coverage stops before training."
    }
}
finally {
    Pop-Location
}
