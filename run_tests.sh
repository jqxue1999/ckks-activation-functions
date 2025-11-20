#!/bin/bash
# Test script for OpenFHE CKKS Softmax implementation

set -e  # Exit on error

echo "=============================================================================="
echo "  OpenFHE CKKS Softmax - Test Runner"
echo "=============================================================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check if required packages are installed
echo ""
echo "Checking dependencies..."

python3 -c "import numpy" 2>/dev/null && echo "✅ numpy installed" || {
    echo "❌ numpy not found. Install with: pip install numpy"
    exit 1
}

python3 -c "import openfhe" 2>/dev/null && echo "✅ openfhe installed" || {
    echo "❌ openfhe not found. Install with: pip install openfhe"
    exit 1
}

python3 -c "import openfhe_numpy" 2>/dev/null && echo "✅ openfhe_numpy installed" || {
    echo "❌ openfhe_numpy not found. Install with: pip install openfhe_numpy"
    exit 1
}

# Run tests
echo ""
echo "=============================================================================="
echo "  Running Tests"
echo "=============================================================================="
echo ""

python3 test_softmax.py

# Capture exit code
TEST_EXIT_CODE=$?

echo ""
echo "=============================================================================="

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "  ✅ All tests passed successfully!"
    echo "=============================================================================="
    exit 0
else
    echo "  ❌ Tests failed with exit code $TEST_EXIT_CODE"
    echo "=============================================================================="
    exit $TEST_EXIT_CODE
fi
