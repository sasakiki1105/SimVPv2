param(
    [int]$WaitPid = 39632,
    [int]$FreeMemoryThresholdMiB = 1500
)

$ErrorActionPreference = "Stop"
$pilotRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = "C:\Users\astro\anaconda3\envs\OpenSTL\python.exe"
$script = Join-Path $pilotRoot "evaluate_radaz_e20_native_fine_only_pilot.py"

function Write-QueueLog([string]$Message) {
    Write-Output "[$(Get-Date -Format s)] $Message"
}

Write-QueueLog "queue started; protecting existing GPU PID=$WaitPid"
$lastNotice = [DateTime]::MinValue
while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
    if (((Get-Date) - $lastNotice).TotalMinutes -ge 5) {
        Write-QueueLog "waiting for existing GPU PID=$WaitPid"
        $lastNotice = Get-Date
    }
    Start-Sleep -Seconds 15
}

Write-QueueLog "PID=$WaitPid finished; waiting for GPU memory to become available"
$lastNotice = [DateTime]::MinValue
while ($true) {
    $usedText = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi failed with exit code $LASTEXITCODE"
    }
    $usedMiB = [int](($usedText | Select-Object -First 1).Trim())
    if ($usedMiB -le $FreeMemoryThresholdMiB) {
        break
    }
    if (((Get-Date) - $lastNotice).TotalMinutes -ge 5) {
        Write-QueueLog "GPU still occupied: ${usedMiB} MiB used"
        $lastNotice = Get-Date
    }
    Start-Sleep -Seconds 15
}

Write-QueueLog "GPU available (${usedMiB} MiB used); starting A-B-C pilot"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Push-Location $pilotRoot
try {
    & $python $script --phase all --device cuda
    $pilotExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
Write-QueueLog "pilot finished with exit code $pilotExitCode"
exit $pilotExitCode
