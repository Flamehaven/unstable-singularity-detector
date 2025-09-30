@echo off
REM Run script for Unstable Singularity Detector Docker container (Windows)
REM Usage: run.bat [command] [args...]

set IMAGE=unstable-singularity-detector:latest

echo [*] Running Unstable Singularity Detector
echo.

REM Check if image exists
docker image inspect %IMAGE% >nul 2>&1
if errorlevel 1 (
    echo [!] Image not found. Building first...
    call build.bat
)

REM Default: interactive shell
if "%~1"=="" (
    echo [*] Starting interactive shell
    docker run -it --rm ^
        --gpus all ^
        -v "%cd%\outputs:/app/outputs" ^
        -v "%cd%\logs:/app/logs" ^
        -v "%cd%\checkpoints:/app/checkpoints" ^
        %IMAGE% ^
        /bin/bash
) else (
    REM Run with provided command
    echo [*] Running: %*
    docker run -it --rm ^
        --gpus all ^
        -v "%cd%\outputs:/app/outputs" ^
        -v "%cd%\logs:/app/logs" ^
        -v "%cd%\checkpoints:/app/checkpoints" ^
        %IMAGE% ^
        %*
)