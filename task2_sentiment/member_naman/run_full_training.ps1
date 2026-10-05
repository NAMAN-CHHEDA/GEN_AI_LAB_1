# Trains the three models on the full gpu config, one after another, and stops at the first failure.
# Safe to re-run: finished models print "already finished" and are skipped; an interrupted one resumes from last.pt.
#   Start:    powershell -ExecutionPolicy Bypass -File .\run_full_training.ps1
#   Check:    add -DryRun to only print which Python and which commands would be used.
param([switch]$DryRun)

$trainScript = Join-Path $PSScriptRoot "src\train.py"
$models = @("ngram_bag", "transformer", "bigru")

# True if this Python can import the packages training needs (the Microsoft Store "python" stub cannot).
function Test-Python($exe, $exeArgs) {
    try {
        & $exe @exeArgs -c "import torch, yaml" 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

# Prefer Python 3.12 (py -3.12), then the python on PATH, then any py -3.
function Find-Python {
    foreach ($candidate in @(@("py", "-3.12"), @("python"), @("py", "-3"))) {
        $exe = $candidate[0]
        $exeArgs = @($candidate | Select-Object -Skip 1)
        if ((Get-Command $exe -ErrorAction SilentlyContinue) -and (Test-Python $exe $exeArgs)) {
            return , $candidate
        }
    }
    return $null
}

$python = Find-Python
if ($null -eq $python) {
    Write-Host "ERROR: no working Python found (tried 'py -3.12', 'python', 'py -3'). Install the packages from requirements.txt first."
    exit 1
}
$pythonExe = $python[0]
$pythonArgs = @($python | Select-Object -Skip 1)
Write-Host "Using Python: $($python -join ' ')"

foreach ($model in $models) {
    if ($DryRun) {
        Write-Host "[dry run] $($python -join ' ') $trainScript --config gpu --model $model --resume"
        continue
    }
    Write-Host ("[{0}] START {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $model)
    & $pythonExe @pythonArgs $trainScript --config gpu --model $model --resume
    $exitCode = $LASTEXITCODE
    Write-Host ("[{0}] END   {1} (exit code {2})" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $model, $exitCode)
    if ($exitCode -ne 0) {
        Write-Host "ERROR: training of '$model' failed with exit code $exitCode. Stopping here; the models after it were not started."
        exit 1
    }
}

if (-not $DryRun) { Write-Host "All three models are finished." }
