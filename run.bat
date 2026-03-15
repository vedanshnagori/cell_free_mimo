@echo off
REM ================================================================
REM Windows Run Script for QNN Wireless Optimization
REM Non-Centralized QNN for Cell-Free MIMO Systems
REM ================================================================

title QNN Wireless Optimization Simulator

echo ========================================
echo  QNN Wireless Optimization Simulator
echo  Cell-Free MIMO System (Algorithm 1)
echo ========================================
echo.

REM ── Python Version Check ────────────────────────────────────────
REM Qiskit requires Python 3.8+
echo Checking Python version...
python --version > temp_ver.txt 2>&1
findstr /r "3\.[89]\|3\.1[0-9]" temp_ver.txt >nul
if errorlevel 1 (
    echo ERROR: Python 3.8 or higher is required for Qiskit.
    echo Please install Python from https://www.python.org/downloads/
    del temp_ver.txt
    pause
    exit /b 1
)
del temp_ver.txt
echo Python version OK.
echo.

REM ── Virtual Environment ─────────────────────────────────────────
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
    echo.
)

echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)
echo.

REM ── Dependencies ────────────────────────────────────────────────
REM Check all required Qiskit packages
echo Checking dependencies...
set MISSING=0
pip show qiskit            >nul 2>&1 || set MISSING=1
pip show qiskit-aer        >nul 2>&1 || set MISSING=1
pip show qiskit-algorithms >nul 2>&1 || set MISSING=1
pip show numpy             >nul 2>&1 || set MISSING=1
pip show matplotlib        >nul 2>&1 || set MISSING=1

if "%MISSING%"=="1" (
    echo Installing missing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies.
        pause
        exit /b 1
    )
    echo Dependencies installed.
) else (
    echo All dependencies found.
)
echo.

REM ── Results Directory ───────────────────────────────────────────
if not exist "results\" (
    echo Creating results directory...
    mkdir results
)

REM ── Main Menu ───────────────────────────────────────────────────
:menu
echo.
echo ========================================
echo  MAIN MENU
echo ========================================
echo  1. Run test suite
echo  2. Run main simulation (Algorithm 1)
echo  3. Run custom scenarios
echo  4. Exit
echo ========================================
echo.
set /p choice="Enter your choice (1-4): "

REM ── Option 1: Test Suite ────────────────────────────────────────
if "%choice%"=="1" (
    echo.
    echo Running test suite...
    echo.
    python test_suite.py
    if errorlevel 1 (
        echo.
        echo WARNING: Some tests failed. Check output above.
    ) else (
        echo.
        echo All tests passed!
    )
    goto menu
)

REM ── Option 2: Main Simulation ───────────────────────────────────
if "%choice%"=="2" (
    echo.
    echo Running main simulation...
    echo This implements Algorithm 1 from the paper.
    echo Results will be saved to results\
    echo.
    python main.py
    if errorlevel 1 (
        echo.
        echo ERROR: Simulation failed. Check output above.
    ) else (
        echo.
        echo Simulation completed! Check results\ for output files.
    )
    goto menu
)

REM ── Option 3: Custom Scenarios ──────────────────────────────────
if "%choice%"=="3" (
    echo.
    echo ========================================
    echo  CUSTOM SCENARIOS
    echo ========================================
    echo  a. Small Network  (NAP=6,  Nuser=3)
    echo  b. Dense Network  (NAP=12, Nuser=6)
    echo  c. More Antennas  (NAP=8,  Nuser=4, NTx=8)
    echo  d. High Interference (Fig. 4, mu_d=0.6)
    echo  e. Compare all scenarios
    echo ========================================
    echo.
    set /p example="Enter your choice (a-e): "

    if "%example%"=="a" (
        echo Running Small Network scenario...
        python examples.py --scenario small
        if errorlevel 1 echo ERROR: Scenario failed.
    )
    if "%example%"=="b" (
        echo Running Dense Network scenario...
        python examples.py --scenario dense
        if errorlevel 1 echo ERROR: Scenario failed.
    )
    if "%example%"=="c" (
        echo Running More Antennas scenario...
        python examples.py --scenario mimo
        if errorlevel 1 echo ERROR: Scenario failed.
    )
    if "%example%"=="d" (
        echo Running High Interference scenario (Fig. 4)...
        python examples.py --scenario interference
        if errorlevel 1 echo ERROR: Scenario failed.
    )
    if "%example%"=="e" (
        echo Comparing all scenarios...
        echo This may take several minutes.
        python examples.py --scenario compare
        if errorlevel 1 echo ERROR: Comparison failed.
    )
    goto menu
)

REM ── Option 4: Exit ──────────────────────────────────────────────
if "%choice%"=="4" (
    echo.
    echo Deactivating virtual environment...
    call venv\Scripts\deactivate.bat
    echo Goodbye!
    exit /b 0
)

echo.
echo Invalid choice. Please enter 1, 2, 3, or 4.
goto menu
