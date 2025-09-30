@echo off
REM Build script for Unstable Singularity Detector Docker image (Windows)
REM Usage: build.bat

echo [*] Building Unstable Singularity Detector Docker Image
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo [-] Docker is not installed. Please install Docker Desktop first.
    exit /b 1
)

REM Build image
echo [*] Building Docker image...
docker build ^
    -t unstable-singularity-detector:1.0.0 ^
    -t unstable-singularity-detector:latest ^
    .

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [+] Build successful!
    echo.
    echo Image tags:
    echo   - unstable-singularity-detector:1.0.0
    echo   - unstable-singularity-detector:latest
    echo.
    echo Run with:
    echo   docker run -it --rm unstable-singularity-detector:latest
    echo.
    echo Or use docker-compose:
    echo   docker-compose up detector
) else (
    echo [-] Build failed!
    exit /b 1
)