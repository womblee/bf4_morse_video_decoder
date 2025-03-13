@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Morse Code Decoder Setup
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed on this system or not in PATH.
    echo.
    echo Please download and install Python from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    echo After installing Python, run this script again.
    echo.
    pause
    exit /b 1
)

:: Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo Found Python version: %python_version%
echo.

:: Check if requirements.txt exists
if not exist requirements.txt (
    echo Error: requirements.txt file not found.
    echo Please make sure the file is in the same directory as this script.
    echo.
    pause
    exit /b 1
)

echo Installing required packages...
echo This may take a few minutes depending on your internet connection.
echo.

:: Install packages from requirements.txt
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Error upgrading pip. Continuing with installation...
)

python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo Error: Failed to install required packages.
    echo Please check your internet connection and try again.
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo You can now run the Morse Code Decoder by executing:
echo python morse_decoder_gui.py
echo.
echo Make sure the morse_decoder_gui.py file is in the current directory.
echo.
pause
exit /b 0