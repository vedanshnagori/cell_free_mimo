#!/bin/bash
# macOS/Linux Run Script for QNN Wireless Optimization

echo "========================================"
echo "QNN Wireless Optimization Simulator"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo ""

# Check if requirements are installed
echo "Checking dependencies..."
if ! pip show qiskit > /dev/null 2>&1; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

# Menu function
show_menu() {
    echo ""
    echo "What would you like to do?"
    echo "1. Run test suite"
    echo "2. Run main simulation"
    echo "3. Run custom examples"
    echo "4. Exit"
    echo ""
    read -p "Enter your choice (1-4): " choice
    
    case $choice in
        1)
            echo ""
            echo "Running test suite..."
            python test_suite.py
            show_menu
            ;;
        2)
            echo ""
            echo "Running main simulation..."
            python main.py
            show_menu
            ;;
        3)
            echo ""
            echo "Custom Examples:"
            echo "a. Small network"
            echo "b. Dense network"
            echo "c. Massive MIMO"
            echo "d. Compare all scenarios"
            read -p "Enter your choice (a-d): " example
            
            case $example in
                a) python examples.py --scenario small ;;
                b) python examples.py --scenario dense ;;
                c) python examples.py --scenario mimo ;;
                d) python examples.py --scenario compare ;;
                *) echo "Invalid choice" ;;
            esac
            show_menu
            ;;
        4)
            echo ""
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice. Please try again."
            show_menu
            ;;
    esac
}

# Start menu
show_menu
