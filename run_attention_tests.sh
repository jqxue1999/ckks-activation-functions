#!/bin/bash

# Test runner for OpenFHE CKKS Attention Block
# This script runs the comprehensive test suite for attention_openfhe.py

echo "================================================================================"
echo "  OpenFHE CKKS Attention Block - Test Runner"
echo "================================================================================"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: python3 is not installed"
    echo "Please install Python 3.10+ to run these tests"
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ ERROR: numpy is not installed"
    echo "Install with: pip install numpy"
    exit 1
fi

python3 -c "import openfhe" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ ERROR: openfhe is not installed"
    echo "Install with: pip install openfhe openfhe_numpy"
    exit 1
fi

python3 -c "import openfhe_numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ ERROR: openfhe_numpy is not installed"
    echo "Install with: pip install openfhe_numpy"
    exit 1
fi

echo "✅ All dependencies installed"
echo ""

# Check if required files exist
if [ ! -f "attention_openfhe.py" ]; then
    echo "❌ ERROR: attention_openfhe.py not found"
    exit 1
fi

if [ ! -f "softmax_openfhe.py" ]; then
    echo "❌ ERROR: softmax_openfhe.py not found (required dependency)"
    exit 1
fi

if [ ! -f "test_attention.py" ]; then
    echo "❌ ERROR: test_attention.py not found"
    exit 1
fi

echo "✅ All required files found"
echo ""

# Run the tests
echo "================================================================================"
echo "  Running Attention Block Tests"
echo "================================================================================"
echo ""
echo "Note: This may take several minutes due to:"
echo "  - CKKS encryption/decryption operations"
echo "  - Matrix multiplications on encrypted data"
echo "  - Row-wise softmax computation"
echo ""

# Run tests with timeout (10 minutes)
timeout 600 python3 test_attention.py

# Check exit code
exit_code=$?

echo ""
echo "================================================================================"

if [ $exit_code -eq 0 ]; then
    echo "  ✅ All tests completed successfully!"
    echo "================================================================================"
    echo ""
    echo "The attention block implementation is working correctly."
    echo ""
    echo "Usage example:"
    echo "  from attention_openfhe import AttentionBlockOpenFHE"
    echo "  attention = AttentionBlockOpenFHE(seq_len=8, d_k=8, d_v=8)"
    echo "  output, weights = attention.attention_encrypted(Q, K, V)"
    exit 0
elif [ $exit_code -eq 124 ]; then
    echo "  ⏱️  Tests timed out (>10 minutes)"
    echo "================================================================================"
    echo ""
    echo "The tests took too long. This might be normal for large configurations."
    exit 1
else
    echo "  ❌ Tests failed (exit code: $exit_code)"
    echo "================================================================================"
    echo ""
    echo "Please check the error messages above for details."
    exit 1
fi
