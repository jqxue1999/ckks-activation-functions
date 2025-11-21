#!/bin/bash

# Test runner for Fully Encrypted Transformer
# This script runs the comprehensive test suite for fully encrypted transformer components

echo "================================================================================"
echo "  Fully Encrypted Transformer - Test Runner"
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
    "attention_fully_encrypted.py"
    "relu_openfhe.py"
    "matmul_openfhe.py"
    "softmax_openfhe.py"
    "test_full_transformer_encrypted.py"
    "test_softmax_fully_encrypted.py"
    "test_layernorm_encrypted.py"
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
echo "  Running Fully Encrypted Transformer Tests"
echo "================================================================================"
echo ""
echo "⚠️  WARNING: This will take 10-12 minutes due to:"
echo "  - Complete transformer pipeline test (~6 min)"
echo "  - Softmax fully encrypted test (~3 min)"
echo "  - LayerNorm encrypted test (~3 min)"
echo ""
echo "Components tested:"
echo "  1. Softmax (fully encrypted - ciphertext input)"
echo "  2. Attention (fully encrypted - no decryption)"
echo "  3. LayerNorm (Goldschmidt algorithm)"
echo "  4. Complete Pipeline (Input → Attention → LayerNorm → Output)"
echo ""

# Initialize counters
total_tests=0
passed_tests=0

# Test 1: Complete Pipeline
echo "================================================================================"
echo "Test 1/3: Complete Transformer Pipeline"
echo "================================================================================"
timeout 600 python3 test_full_transformer_encrypted.py
exit_code=$?
total_tests=$((total_tests + 1))
if [ $exit_code -eq 0 ]; then
    passed_tests=$((passed_tests + 1))
    echo "✅ Test 1 passed"
else
    echo "❌ Test 1 failed (exit code: $exit_code)"
fi
echo ""

# Test 2: Softmax Fully Encrypted
echo "================================================================================"
echo "Test 2/3: Softmax Fully Encrypted"
echo "================================================================================"
timeout 300 python3 test_softmax_fully_encrypted.py
exit_code=$?
total_tests=$((total_tests + 1))
if [ $exit_code -eq 0 ]; then
    passed_tests=$((passed_tests + 1))
    echo "✅ Test 2 passed"
else
    echo "❌ Test 2 failed (exit code: $exit_code)"
fi
echo ""

# Test 3: LayerNorm Encrypted
echo "================================================================================"
echo "Test 3/3: LayerNorm Encrypted"
echo "================================================================================"
timeout 300 python3 test_layernorm_encrypted.py
exit_code=$?
total_tests=$((total_tests + 1))
if [ $exit_code -eq 0 ]; then
    passed_tests=$((passed_tests + 1))
    echo "✅ Test 3 passed"
else
    echo "❌ Test 3 failed (exit code: $exit_code)"
fi
echo ""

# Print summary
echo "================================================================================"
echo "  TEST SUMMARY"
echo "================================================================================"
echo ""
echo "Tests passed: $passed_tests/$total_tests"
echo ""

if [ $passed_tests -eq $total_tests ]; then
    echo "✅ ALL TESTS PASSED!"
    echo ""
    echo "The fully encrypted transformer implementation is working correctly."
    echo ""
    echo "Key Features:"
    echo "  ✓ Softmax accepts ciphertext input directly (0 decryptions)"
    echo "  ✓ Attention fully encrypted (0 decryptions)"
    echo "  ✓ LayerNorm with Goldschmidt (0 decryptions)"
    echo "  ✓ Complete pipeline: Only 1 decryption (final output)"
    echo ""
    echo "Usage example:"
    echo "  from attention_fully_encrypted import AttentionFullyEncrypted"
    echo "  from transformer_openfhe import LayerNormOpenFHE"
    echo ""
    echo "  # Complete pipeline"
    echo "  attention = AttentionFullyEncrypted(d_k=8, mult_depth=60)"
    echo "  output = attention.attention_single_encrypted(q, k, v, return_ciphertext=True)"
    echo ""
    echo "Architecture:"
    echo "  Input → [Encrypt] → Attention → Residual → LayerNorm → [Decrypt] → Output"
    echo "          └─────────────── ALL ON CIPHERTEXT ──────────────┘"
    echo ""
    exit 0
else
    echo "❌ SOME TESTS FAILED"
    echo ""
    echo "Please check the error messages above for details."
    echo ""
    echo "Common issues:"
    echo "  - Approximation errors: Check tolerance settings"
    echo "  - Memory issues: Try smaller dimensions (d_model=4)"
    echo "  - Timeout: Normal for first run (key generation)"
    echo "  - Depth errors: Increase mult_depth parameter"
    echo ""
    exit 1
fi
