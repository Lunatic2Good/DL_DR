@echo off
echo ========================================
echo Deep Learning Model Comparison System
echo Installation Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo Python found!
python --version
echo.

echo Installing dependencies...
echo.

REM Try pip first
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo pip command failed, trying python -m pip...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Installation failed. Please check the error messages above.
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo You can now run the application with:
echo   python app.py
echo.
pause

