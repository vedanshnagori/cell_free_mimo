#!/bin/bash
# ================================================================
# macOS/Linux Run Script for QNN Wireless Optimization
# Non-Centralized QNN for Cell-Free MIMO Systems
# ================================================================

echo "========================================"
echo " QNN Wireless Optimization Simulator"
echo " Cell-Free MIMO System (Algorithm 1)"
echo "========================================"
echo ""

# ── Python Version Check ─────────────────────────────────────────
# Qiskit requires Python 3.8+
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '3\.[89]|3\.[1-9][0-9]')
if [ -z "$PYTHON_VERSION" ]; then
    echo "ERROR: Python 3.8 or higher is required for Qiskit."
    echo "Please install Python from https://www.python.org/downloads/"
    exit 1
fi
echo "Python version OK."
echo ""

# ── Virtual Environment ──────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment."
        exit 1
    fi
    echo "Virtual environment created."
    echo ""
fi

echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment."
    exit 1
fi
echo ""

# ── Dependencies ─────────────────────────────────────────────────
# Check all required Qiskit packages
echo "Checking dependencies..."
MISSING=0
pip show qiskit            > /dev/null 2>&1 || MISSING=1
pip show qiskit-aer        > /dev/null 2>&1 || MISSING=1
pip show qiskit-algorithms > /dev/null 2>&1 || MISSING=1
pip show numpy             > /dev/null 2>&1 || MISSING=1
pip show matplotlib        > /dev/null 2>&1 || MISSING=1

if [ "$MISSING" -eq 1 ]; then
    echo "Installing missing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies."
        exit 1
    fi
    echo "Dependencies installed."
else
    echo "All dependencies found."
fi
echo ""

# ── Results Directory ─────────────────────────────────────────────
if [ ! -d "results" ]; then
    echo "Creating results directory..."
    mkdir -p results
fi

# ── Menu Function ─────────────────────────────────────────────────
show_menu() {
    echo ""
    echo "========================================"
    echo " MAIN MENU"
    echo "========================================"
    echo " 1. Run test suite"
    echo " 2. Run main simulation (Algorithm 1)"
    echo " 3. Run custom scenarios"
    echo " 4. Exit"
    echo "========================================"
    echo ""
    read -p "Enter your choice (1-4): " choice

    case $choice in

        # ── Option 1: Test Suite ──────────────────────────────────
        1)
            echo ""
            echo "Running test suite..."
            echo ""
            python test_suite.py
            if [ $? -ne 0 ]; then
                echo ""
                echo "WARNING: Some tests failed. Check output above."
            else
                echo ""
                echo "All tests passed!"
            fi
            show_menu
            ;;

        # ── Option 2: Main Simulation ─────────────────────────────
        2)
            echo ""
            echo "Running main simulation..."
            echo "This implements Algorithm 1 from the paper."
            echo "Results will be saved to results/"
            echo ""
            python main.py
            if [ $? -ne 0 ]; then
                echo ""
                echo "ERROR: Simulation failed. Check output above."
            else
                echo ""
                echo "Simulation completed! Check results/ for output files."
            fi
            show_menu
            ;;

        # ── Option 3: Custom Scenarios ────────────────────────────
        3)
            echo ""
            echo "========================================"
            echo " CUSTOM SCENARIOS"
            echo "========================================"
            echo " a. Small Network  (NAP=6,  Nuser=3)"
            echo " b. Dense Network  (NAP=12, Nuser=6)"
            echo " c. More Antennas  (NAP=8,  Nuser=4, NTx=8)"
            echo " d. High Interference (Fig. 4, mu_d=0.6)"
            echo " e. Compare all scenarios"
            echo "========================================"
            echo ""
            read -p "Enter your choice (a-e): " example

            case $example in
                a)
                    echo "Running Small Network scenario..."
                    python examples.py --scenario small
                    if [ $? -ne 0 ]; then echo "ERROR: Scenario failed."; fi
                    ;;
                b)
                    echo "Running Dense Network scenario..."
                    python examples.py --scenario dense
                    if [ $? -ne 0 ]; then echo "ERROR: Scenario failed."; fi
                    ;;
                c)
                    echo "Running More Antennas scenario..."
                    python examples.py --scenario mimo
                    if [ $? -ne 0 ]; then echo "ERROR: Scenario failed."; fi
                    ;;
                d)
                    echo "Running High Interference scenario (Fig. 4)..."
                    python examples.py --scenario interference
                    if [ $? -ne 0 ]; then echo "ERROR: Scenario failed."; fi
                    ;;
                e)
                    echo "Comparing all scenarios..."
                    echo "This may take several minutes."
                    python examples.py --scenario compare
                    if [ $? -ne 0 ]; then echo "ERROR: Comparison failed."; fi
                    ;;
                *)
                    echo "Invalid choice."
                    ;;
            esac
            show_menu
            ;;

        # ── Option 4: Exit ────────────────────────────────────────
        4)
            echo ""
            echo "Deactivating virtual environment..."
            deactivate
            echo "Goodbye!"
            exit 0
            ;;

        *)
            echo "Invalid choice. Please enter 1, 2, 3, or 4."
            show_menu
            ;;
    esac
}

# ── Start ─────────────────────────────────────────────────────────
show_menu
