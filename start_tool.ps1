$ErrorActionPreference = "Stop"
$ProjectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectDirectory

$VirtualEnvironment = Join-Path $ProjectDirectory ".venv"
$VirtualPython = Join-Path $VirtualEnvironment "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VirtualPython)) {
    Write-Host "Setting up Python ..."
    $PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($PythonLauncher) {
        & $PythonLauncher.Source -3 -m venv $VirtualEnvironment
    }
    else {
        $PythonCommand = Get-Command python -ErrorAction Stop
        & $PythonCommand.Source -m venv $VirtualEnvironment
    }
}

Write-Host "Installing packages ..."
& $VirtualPython -m pip install --disable-pip-version-check --quiet -r (Join-Path $ProjectDirectory "requirements.txt")

Write-Host "Starting MMFM Paper Lab ..."
& $VirtualPython -m streamlit run (Join-Path $ProjectDirectory "app.py")
