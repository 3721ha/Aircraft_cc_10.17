param([switch]$PreflightOnly)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$projectPython = Join-Path $repoRoot 'Aircraft_mumt/.venv/Scripts/python.exe'
$launchArguments = @(
    '_run_best_refine.py',
    '--k', '4',
    '--source-run', 'runs/model_optimization_v1/curriculum_k4_w8a10',
    '--output', 'runs/model_optimization_v1/residual_pg_k4_w8a10_v1',
    '--mode', 'conflict-residual',
    '--iterations', '30',
    '--episodes-per-iter', '8',
    '--learning-rate', '0.00001',
    '--conflict-exposure', '0',
    '--penalty-scale', '1',
    '--policy-anchor-coef', '1',
    '--patience', '0',
    '--validation-every', '2',
    '--training-seed-start', '160000',
    '--validation-seeds', '102000,102001,102002,102003,102004,102005,102006,102007,102008,102009',
    '--repair-dead-candidate-head',
    '--candidate-repair-seed', '20261021'
)
if ($PreflightOnly) { $launchArguments += '--preflight-only' }
Push-Location -LiteralPath $repoRoot
try {
    & $projectPython @launchArguments
    $launchExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($launchExitCode -ne 0) {
    throw "Residual PPO launcher failed with exit code $launchExitCode"
}
