$ErrorActionPreference = "Stop"
Set-Location "C:\Users\astro\research\SimVPv2"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
& "C:\Users\astro\anaconda3\envs\OpenSTL\python.exe" "monitor_forecast_horizon_queue.py" "--interval" "60"
