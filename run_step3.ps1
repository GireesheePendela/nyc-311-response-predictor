$ErrorActionPreference = 'Stop'
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $venvPython)) {
    Write-Error "Missing .venv interpreter at $venvPython"
}

& $venvPython (Join-Path $PSScriptRoot 'step3_clean.py')
