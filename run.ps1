# run.ps1 -- Windows PowerShell equivalent of Makefile targets
# Usage: .\run.ps1 <target>
# Example: .\run.ps1 test

param(
    [Parameter(Position=0)]
    [string]$Target = "help"
)

function Install {
    pip install -r requirements.txt
    pip install -e . --no-deps
}

function InstallDev {
    pip install -r requirements-dev.txt
    pip install -e . --no-deps
}

function RunTests {
    pytest tests/ -v --tb=short
}

function RunTestsCov {
    pytest tests/ -v --tb=short --cov=src/nn_core --cov-report=term-missing
}

function RunLint {
    ruff check src/nn_core/
}

function RunAudit {
    pip-audit -r requirements.txt --skip-editable
}

function RunEx09 {
    python exercises/ex09_full_backprop.py
}

function RunEx10 {
    python exercises/ex10_bias_early_stop.py
}

function RunDebugger {
    python tools/backprop_debugger.py
}

function Clean {
    Get-ChildItem -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force
    Get-ChildItem -Recurse -Filter "*.pyc"   | Remove-Item -Force
    Get-ChildItem -Recurse -Filter "*.pyo"   | Remove-Item -Force
    @(".pytest_cache","htmlcov","dist","build") | ForEach-Object {
        if (Test-Path $_) { Remove-Item $_ -Recurse -Force }
    }
    Write-Host "Clean complete."
}

function ShowHelp {
    Write-Host ""
    Write-Host "mlp-study-kit PowerShell runner"
    Write-Host "Usage: .\run.ps1 <target>"
    Write-Host ""
    Write-Host "Targets:"
    Write-Host "  install      -- install runtime deps + editable package"
    Write-Host "  install-dev  -- install dev deps (pytest, ruff, pip-audit) + package"
    Write-Host "  test         -- run pytest suite"
    Write-Host "  test-cov     -- run pytest with coverage report"
    Write-Host "  lint         -- run ruff on src/nn_core/"
    Write-Host "  audit        -- run pip-audit security scan"
    Write-Host "  run-ex09     -- run ex09_full_backprop.py"
    Write-Host "  run-ex10     -- run ex10_bias_early_stop.py"
    Write-Host "  run-debugger -- run tools/backprop_debugger.py"
    Write-Host "  clean        -- remove __pycache__, .pyc, dist, build"
    Write-Host ""
}

switch ($Target) {
    "install"      { Install }
    "install-dev"  { InstallDev }
    "test"         { RunTests }
    "test-cov"     { RunTestsCov }
    "lint"         { RunLint }
    "audit"        { RunAudit }
    "run-ex09"     { RunEx09 }
    "run-ex10"     { RunEx10 }
    "run-debugger" { RunDebugger }
    "clean"        { Clean }
    default        { ShowHelp }
}
