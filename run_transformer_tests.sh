#!/bin/bash

# Test runner for OpenFHE CKKS Transformer Block
# This script runs the comprehensive test suite for transformer_openfhe.py

echo "================================================================================"
echo "  OpenFHE CKKS Transformer Block - Test Runner"
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
required_files=(
    "transformer_openfhe.py"
    "attention_openfhe.py"
    "relu_openfhe.py"
    "matmul_openfhe.py"
    "softmax_openfhe.py"
    "test_transformer.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ ERROR: $file not found"
        exit 1
    fi
done

echo "✅ All required files found"
echo ""

# Run the tests
echo "================================================================================"
echo "  Running Transformer Block Tests"
echo "================================================================================"
echo ""
echo "⚠️  WARNING: This will take 15-30 minutes due to:"
echo "  - Self-attention computation (~4 min)"
echo "  - Feed-forward network with ReLU (~5 min per layer)"
echo "  - Layer normalization"
echo "  - Multiple test cases"
echo ""
echo "Components tested:"
echo "  1. Layer Normalization"
echo "  2. Feed-Forward Network"
echo "  3. Basic Transformer Block"
echo "  4. Transformer Properties"
echo ""

# Run tests with timeout (30 minutes)
timeout 1800 python3 test_transformer.py

# Check exit code
exit_code=$?

echo ""
echo "================================================================================"

if [ $exit_code -eq 0 ]; then
    echo "  ✅ All tests completed successfully!"
    echo "================================================================================"
    echo ""
    echo "The transformer block implementation is working correctly."
    echo ""
    echo "Usage example:"
    echo "  from transformer_openfhe import TransformerBlockOpenFHE"
    echo "  transformer = TransformerBlockOpenFHE(d_model=8, d_ff=32)"
    echo "  output, weights = transformer.forward(x)"
    echo ""
    echo "Architecture:"
    echo "  Input → Self-Attention → (+) → LayerNorm → FFN → (+) → LayerNorm → Output"
    echo "           ↓_______________|                  ↓_______|"
    echo "         (residual)                      (residual)"
    exit 0
elif [ $exit_code -eq 124 ]; then
    echo "  ⏱️  Tests timed out (>30 minutes)"
    echo "================================================================================"
    echo ""
    echo "The tests took too long. This might be normal for larger configurations."
    echo "Try reducing parameters (smaller d_model, d_ff, or softmax_K)."
    exit 1
else
    echo "  ❌ Tests failed (exit code: $exit_code)"
    echo "================================================================================"
    echo ""
    echo "Please check the error messages above for details."
    echo ""
    echo "Common issues:"
    echo "  - Approximation errors: Expected due to ReLU/Softmax approximations"
    echo "  - Memory issues: Try smaller dimensions"
    echo "  - Timeout: Reduce softmax_K or sequence length"
    exit 1
fi
