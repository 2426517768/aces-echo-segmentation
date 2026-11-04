param([ValidateSet("main","suite","smoke","doctor","prepare","evaluate","tests")][string]$Mode = "main")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Check-Exit([string]$What) {
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit $LASTEXITCODE). Read the error above." }
}

function Invoke-CheckedNative([string]$Executable, [string[]]$Arguments) {
    $SavedPreference = $ErrorActionPreference
    try {
        # Python/pip warnings on stderr are not PowerShell setup failures.
        $ErrorActionPreference = "Continue"
        & $Executable @Arguments
        $NativeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $SavedPreference
    }
    if ($NativeExitCode -ne 0) { throw "Native command failed (exit $NativeExitCode): $Executable" }
}

try {
    $VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $VenvPython)) {
        $Python = $null
        foreach ($Version in @("3.12", "3.11", "3.10")) {
            if (Get-Command py -ErrorAction SilentlyContinue) {
                try {
                    $Found = & py "-$Version" -c "import sys; print(sys.executable)" 2>$null
                    if ($LASTEXITCODE -eq 0 -and $Found) { $Python = [string]($Found | Select-Object -Last 1); break }
                } catch { }
            }
        }
        if (-not $Python -and (Get-Command python -ErrorAction SilentlyContinue)) {
            try {
                $Found = & python -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,12); print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $Found) { $Python = [string]($Found | Select-Object -Last 1) }
            } catch { }
        }
        if (-not $Python) {
            $PrivatePython = Join-Path $PSScriptRoot "runtime\python\python.exe"
            if (-not (Test-Path $PrivatePython)) {
                Write-Host "No supported Python found. Installing Python 3.12.10 in this package's runtime directory; administrator access and PATH changes are not used."
                New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "runtime") | Out-Null
                $Installer = Join-Path $PSScriptRoot "runtime\python-installer.exe"
                Invoke-WebRequest -UseBasicParsing -Uri "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe" -OutFile $Installer
                $Signature = Get-AuthenticodeSignature $Installer
                if ($Signature.Status -ne "Valid" -or $Signature.SignerCertificate.Subject -notmatch "Python Software Foundation") {
                    throw "Python installer signature validation failed. Install supported Python from python.org manually."
                }
                $Target = Join-Path $PSScriptRoot "runtime\python"
                $InstallArgs = @("/quiet", "InstallAllUsers=0", "TargetDir=`"$Target`"", "Include_launcher=0", "PrependPath=0", "AssociateFiles=0", "Shortcuts=0", "Include_doc=0", "Include_test=0")
                $Process = Start-Process -FilePath $Installer -ArgumentList $InstallArgs -Wait -PassThru
                if ($Process.ExitCode -notin @(0,3010)) { throw "Python installer failed: $($Process.ExitCode)" }
            }
            $Python = $PrivatePython
        }
        Invoke-CheckedNative -Executable $Python -Arguments @("-m", "venv", ".venv")
    }
    $Stamp = Join-Path $PSScriptRoot ".venv\aces_dependencies.txt"
    $Wanted = "torch2.7.1-cu128-" + (Get-FileHash "requirements.txt" -Algorithm SHA256).Hash
    $Ready = (Test-Path $Stamp) -and ((Get-Content $Stamp -Raw).Trim() -eq $Wanted)
    if ($Ready) {
        & $VenvPython -c "import torch, numpy, scipy, PIL, SimpleITK, transformers; assert torch.__version__.startswith('2.7.1'); assert torch.version.cuda == '12.8'" 2>$null
        $Ready = $LASTEXITCODE -eq 0
    }
    if (-not $Ready) {
        Write-Host "First setup downloads several GB. Keep this window open."
        Invoke-CheckedNative -Executable $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
        Invoke-CheckedNative -Executable $VenvPython -Arguments @("-m", "pip", "install", "torch==2.7.1+cu128", "--index-url", "https://download.pytorch.org/whl/cu128")
        Invoke-CheckedNative -Executable $VenvPython -Arguments @("-m", "pip", "install", "-r", "requirements.txt")
        Invoke-CheckedNative -Executable $VenvPython -Arguments @("-m", "pip", "check")
        Invoke-CheckedNative -Executable $VenvPython -Arguments @("-c", "import torch; assert torch.__version__.startswith('2.7.1'); assert torch.version.cuda == '12.8'")
        Set-Content -Path $Stamp -Value $Wanted -Encoding ASCII
    }
    Invoke-CheckedNative -Executable $VenvPython -Arguments @("-u", "launcher.py", "--mode", $Mode)
    exit 0
}
catch {
    Write-Host $_ -ForegroundColor Red
    Write-Host "Setup or execution did not finish. Keep the error text; no clinical result was created."
    exit 1
}
