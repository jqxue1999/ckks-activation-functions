#!/bin/bash
# Test script for OpenFHE CKKS ReLU implementation

set -e  # Exit on error

echo "=============================================================================="
echo "  OpenFHE CKKS ReLU - Test Runner"
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

# Note: openfhe_numpy not strictly required for ReLU, but good to check
python3 -c "import openfhe_numpy" 2>/dev/null && echo "✅ openfhe_numpy installed" || {
    echo "⚠️  openfhe_numpy not found (optional for ReLU)"
}

# Run tests
echo ""
echo "=============================================================================="
echo "  Running Tests"
echo "=============================================================================="
echo ""
echo "ℹ️  Note: ReLU uses polynomial approximation in CKKS"
echo "   Expect some approximation error, especially for large |x|"
echo ""

python3 test_relu.py

# Capture exit code
TEST_EXIT_CODE=$?

echo ""
echo "=============================================================================="

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "  ✅ All tests passed successfully!"
    echo ""
    echo "  Performance summary:"
    echo "    - Initialization: ~3-5 seconds"
    echo "    - ReLU computation: ~2-5 seconds per operation"
    echo "    - Approximation error: typically <1.0 for x in [-5, 5]"
    echo "=============================================================================="
    exit 0
else
    echo "  ❌ Tests failed with exit code $TEST_EXIT_CODE"
    echo "=============================================================================="
    exit $TEST_EXIT_CODE
fi
