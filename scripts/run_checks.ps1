$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$python = Join-Path $repo '.venv/Scripts/python.exe'
if (-not (Test-Path $python)) { throw 'Run scripts/setup.ps1 with an explicit CPython 3.11 x64 executable first.' }
& $python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,11) else 1)"
if ($LASTEXITCODE -ne 0) { throw '.venv is not Python 3.11' }
$priorPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repo 'src'
    & $python -B -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Synthetic tests failed' }
    & $python -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependencies failed validation' }
    & $python -B scripts/check_staged.py
    if ($LASTEXITCODE -ne 0) { throw 'Unsafe staged files' }
} finally {
    $env:PYTHONPATH = $priorPythonPath
}
