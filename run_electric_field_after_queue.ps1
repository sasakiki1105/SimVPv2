param(
    [Parameter(Mandatory = $true)]
    [int]$WaitPid
)

$ErrorActionPreference = "Stop"
Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue
Set-Location "C:\Users\astro\research\SimVPv2"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
& "C:\Users\astro\anaconda3\envs\OpenSTL\python.exe" "plot_forecast_horizon_electric_field_error.py" *> "C:\Users\astro\research\SimVPv2\workdirs\forecast_horizon_queue_logs\electric_field_after_queue.log"
