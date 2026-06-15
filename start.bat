@echo off
title HAFA Node & Miner
echo ===================================================
echo   🚀 Starting HAFA Genesis Node...
echo ===================================================
start "HAFA Node" cmd /k "cargo run --release"

echo Waiting 5 seconds for the node to initialize...
timeout /t 5 /nobreak >nul

echo ===================================================
echo   ⛏️  Starting HAFA Miner...
echo ===================================================
start "HAFA Miner" cmd /k "cargo run --bin hafa-miner --release"

echo.
echo ✅ Both Node and Miner are running in separate windows!
echo Press any key to close this launcher...
pause >nul