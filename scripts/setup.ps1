param([Parameter(Mandatory=$true)][string]$Python311)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
& $Python311 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,11) else 1)"
if ($LASTEXITCODE -ne 0) { throw 'CPython 3.11 x64 is required. This script will not use a different Python.' }
& $Python311 -c "import struct,sys; sys.exit(0 if struct.calcsize('P') == 8 else 1)"
if ($LASTEXITCODE -ne 0) { throw 'A 64-bit interpreter is required.' }
if (Test-Path .venv) { throw '.venv already exists; inspect its interpreter before changing it.' }
& $Python311 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
$venvPython = Join-Path $repo '.venv/Scripts/python.exe'
& $venvPython -m pip install -r requirements.lock
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $venvPython -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Editable package installation failed' }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
New-Item -ItemType Directory -Force artifacts/environment | Out-Null
& $venvPython -m pip freeze | Set-Content -Encoding utf8 artifacts/environment/requirements-resolved.txt
& $venvPython -c "import sys,numpy,pandas; print(sys.version); print(numpy.__version__,pandas.__version__)" | Set-Content -Encoding utf8 artifacts/environment/versions.txt
Write-Output 'Environment ready from requirements.lock. Run scripts/check_environment.py and scripts/run_checks.ps1 next.'
