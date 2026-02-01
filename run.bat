@echo off
REM Windows Run Script for QNN Wireless Optimization

echo ========================================
echo QNN Wireless Optimization Simulator
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Check if requirements are installed
echo Checking dependencies...
pip show qiskit >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo.
)

REM Menu
:menu
echo.
echo What would you like to do?
echo 1. Run test suite
echo 2. Run main simulation
echo 3. Run custom examples
echo 4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Running test suite...
    python test_suite.py
    goto menu
)

if "%choice%"=="2" (
    echo.
    echo Running main simulation...
    python main.py
    goto menu
)

if "%choice%"=="3" (
    echo.
    echo Custom Examples:
    echo a. Small network
    echo b. Dense network
    echo c. Massive MIMO
    echo d. Compare all scenarios
    set /p example="Enter your choice (a-d): "
    
    if "%example%"=="a" python examples.py --scenario small
    if "%example%"=="b" python examples.py --scenario dense
    if "%example%"=="c" python examples.py --scenario mimo
    if "%example%"=="d" python examples.py --scenario compare
    
    goto menu
)

if "%choice%"=="4" (
    echo.
    echo Goodbye!
    exit /b
)

echo Invalid choice. Please try again.
goto menu
