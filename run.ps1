# run.ps1 -- Windows PowerShell equivalent of Makefile targets
# Usage: .\run.ps1 <target>
# Example: .\run.ps1 test

param(
    [Parameter(Position=0)]
    [string]$Target = "help"
)

function ShowHelp {
    Write-Host ""
    Write-Host "mlp-study-kit PowerShell runner"
    Write-Host "Usage: .\run.ps1 <target>"
    Write-Host ""
    Write-Host "Setup:"
    Write-Host "  install          core deps (requirements.txt) + editable package"
    Write-Host "  install-dev      dev tools (requirements_dev.txt) + editable package"
    Write-Host "  install-gpu      GPU deps for Windows (requirements_gpu_windows.txt)"
    Write-Host "  install-win      Windows runtime with TF (requirements_windows.txt)"
    Write-Host ""
    Write-Host "Quality:"
    Write-Host "  test             run pytest suite"
    Write-Host "  test-cov         run pytest with coverage report"
    Write-Host "  lint             ruff on src/nn_core/ (strict) + exercises/ (info)"
    Write-Host "  typecheck        mypy on src/nn_core/"
    Write-Host "  bandit           bandit static security scan"
    Write-Host "  audit            pip-audit CVE scan"
    Write-Host "  security         bandit + audit combined"
    Write-Host ""
    Write-Host "Run:"
    Write-Host "  run-ex09         exercises/ex09_full_backprop.py"
    Write-Host "  run-ex10         exercises/ex10_bias_early_stop.py"
    Write-Host "  run-debugger     tools/backprop_debugger.py"
    Write-Host "  notebook         open Jupyter in notebooks/"
    Write-Host ""
    Write-Host "Clean:"
    Write-Host "  clean            remove __pycache__, build, dist, coverage"
    Write-Host ""
}

function Install {
    pip install -e . --no-deps
    pip install -r requirements.txt
}

function InstallDev {
    pip install -e . --no-deps
    pip install -r requirements_dev.txt
}

function InstallGPU {
    pip install -r requirements_gpu_windows.txt
}

function InstallWin {
    pip install -r requirements_windows.txt
}

function RunTests {
    pytest tests/ -v --tb=short
}

function RunTestsCov {
    pytest tests/ -v --tb=short --cov=src/nn_core --cov-report=term-missing --cov-report=html:htmlcov
}

function RunLint {
    ruff check src/nn_core/
    Write-Host "--- exercises/ tools/ (informational) ---"
    ruff check exercises/ tools/
}

function RunTypecheck {
    mypy src/nn_core/ --ignore-missing-imports
}

function RunBandit {
    bandit -r src/ -c pyproject.toml
}

function RunAudit {
    pip-audit -r requirements.txt --skip-editable
}

function RunSecurity {
    RunBandit
    RunAudit
}

function RunPreCommit {
    pre-commit run --all-files
}

function RunEx09 { python exercises/ex09_full_backprop.py }
function RunEx10 { python exercises/ex10_bias_early_stop.py }
function RunDebugger { python tools/backprop_debugger.py }
function RunNotebook { jupyter notebook notebooks/ }

function Clean {
    Get-ChildItem -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force
    Get-ChildItem -Recurse -Filter "*.pyc"   | Remove-Item -Force
    Get-ChildItem -Recurse -Filter "*.pyo"   | Remove-Item -Force
    @(".pytest_cache", "htmlcov", "dist", "build", ".coverage") | ForEach-Object {
        if (Test-Path $_) { Remove-Item $_ -Recurse -Force }
    }
    Get-ChildItem -Recurse -Filter "*.egg-info" -Directory | Remove-Item -Recurse -Force
    Write-Host "Clean done."
}

switch ($Target) {
    "install"      { Install }
    "install-dev"  { InstallDev }
    "install-gpu"  { InstallGPU }
    "install-win"  { InstallWin }
    "test"         { RunTests }
    "test-cov"     { RunTestsCov }
    "lint"         { RunLint }
    "typecheck"    { RunTypecheck }
    "bandit"       { RunBandit }
    "audit"        { RunAudit }
    "security"     { RunSecurity }
    "pre-commit"   { RunPreCommit }
    "run-ex09"     { RunEx09 }
    "run-ex10"     { RunEx10 }
    "run-debugger" { RunDebugger }
    "notebook"     { RunNotebook }
    "clean"        { Clean }
    default        { ShowHelp }
}
