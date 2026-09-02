@echo off
cd /d C:\Users\astro\research\SimVPv2
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONDONTWRITEBYTECODE=1
C:\Users\astro\anaconda3\envs\OpenSTL\python.exe run_radaz_spectral_full_100ep_queue.py 1>>workdirs\radaz_spectral_full_100ep_queue_logs\runner_stdout.log 2>>workdirs\radaz_spectral_full_100ep_queue_logs\runner_stderr.log
