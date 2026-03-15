@echo off
cd /d C:\thesis\relaynet2
echo START %date% %time% > norm_sic_output.log
"C:\Program Files\Python311\python.exe" -u scripts\run_normalized_sic_only.py >> norm_sic_output.log 2>&1
echo END %date% %time% EXITCODE=%ERRORLEVEL% >> norm_sic_output.log
