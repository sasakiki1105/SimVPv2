param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingPid,
    [Parameter(Mandatory = $true)]
    [int]$QueuePid,
    [Parameter(Mandatory = $true)]
    [int]$TargetEpoch,
    [Parameter(Mandatory = $true)]
    [string]$TrainingLog,
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [Parameter(Mandatory = $true)]
    [string]$StatusDirectory
)

$ErrorActionPreference = 'Stop'
$statusDir = [System.IO.Path]::GetFullPath($StatusDirectory)
$trainingLogPath = [System.IO.Path]::GetFullPath($TrainingLog)
$checkpointPath = [System.IO.Path]::GetFullPath($Checkpoint)
$watchLog = Join-Path $statusDir 'pause_after_epoch.log'
$markerPath = Join-Path $statusDir 'paused_state.json'

function Write-WatchLog([string]$Message) {
    $line = '[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Add-Content -LiteralPath $watchLog -Value $line -Encoding UTF8
}

New-Item -ItemType Directory -Path $statusDir -Force | Out-Null
$baselineCheckpointTime = if (Test-Path -LiteralPath $checkpointPath) {
    (Get-Item -LiteralPath $checkpointPath).LastWriteTimeUtc
} else {
    [datetime]::MinValue
}

Write-WatchLog "armed target_epoch=$TargetEpoch training_pid=$TrainingPid queue_pid=$QueuePid"
Write-WatchLog "baseline_checkpoint_utc=$($baselineCheckpointTime.ToString('o'))"

$queue = Get-Process -Id $QueuePid -ErrorAction SilentlyContinue
if ($null -ne $queue) {
    Stop-Process -Id $QueuePid -Force
    $queue.WaitForExit(10000) | Out-Null
    Write-WatchLog 'queue parent stopped; active training is allowed to finish the target epoch'
} else {
    Write-WatchLog 'queue parent was already stopped'
}

while ($true) {
    $training = Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue
    if ($null -eq $training) {
        Write-WatchLog 'training process exited before the target checkpoint was observed'
        exit 2
    }

    $epochLogged = $false
    if (Test-Path -LiteralPath $trainingLogPath) {
        $epochLogged = $null -ne (Select-String -LiteralPath $trainingLogPath -Pattern "Epoch $TargetEpoch`:" -Quiet)
    }

    if ($epochLogged -and (Test-Path -LiteralPath $checkpointPath)) {
        $checkpointInfo = Get-Item -LiteralPath $checkpointPath
        $checkpointUpdated = $checkpointInfo.LastWriteTimeUtc -gt $baselineCheckpointTime
        $checkpointLargeEnough = $checkpointInfo.Length -gt 100MB
        if ($checkpointUpdated -and $checkpointLargeEnough) {
            try {
                $stream = [System.IO.File]::Open(
                    $checkpointPath,
                    [System.IO.FileMode]::Open,
                    [System.IO.FileAccess]::Read,
                    [System.IO.FileShare]::None
                )
                $stream.Close()
                Start-Sleep -Seconds 3
                Stop-Process -Id $TrainingPid -Force
                $training.WaitForExit(10000) | Out-Null

                $payload = [ordered]@{
                    paused_at = (Get-Date).ToString('s')
                    completed_epoch = $TargetEpoch
                    resume_epoch = $TargetEpoch + 1
                    checkpoint = $checkpointPath
                    checkpoint_bytes = $checkpointInfo.Length
                    checkpoint_updated_utc = $checkpointInfo.LastWriteTimeUtc.ToString('o')
                    training_pid = $TrainingPid
                    queue_pid = $QueuePid
                }
                $payload | ConvertTo-Json | Set-Content -LiteralPath $markerPath -Encoding UTF8
                Write-WatchLog "training stopped after epoch=$TargetEpoch; resume_epoch=$($TargetEpoch + 1)"
                exit 0
            } catch [System.IO.IOException] {
                # The checkpoint writer still owns the file. Check again shortly.
            }
        }
    }

    Start-Sleep -Seconds 10
}
