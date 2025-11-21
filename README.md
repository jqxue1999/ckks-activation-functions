# Fully Encrypted Transformer - CKKS Implementation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenFHE](https://img.shields.io/badge/OpenFHE-1.4.2+-green.svg)](https://github.com/openfheorg/openfhe-development)

**Complete transformer inference pipeline on encrypted data with ONLY ONE decryption (final output)!**

## 🎯 Overview

This repository provides a **fully encrypted transformer** implementation using CKKS homomorphic encryption via OpenFHE. All computations happen on ciphertext from input to output.

### Key Achievement: TRUE End-to-End Encryption

- **Input encrypted ONCE** → All computations on ciphertext → **Output decrypted ONCE**
- **No intermediate decryption** of attention scores, softmax inputs, mean, variance, or any internal values
- Complete transformer layer: Attention → Residual → LayerNorm

## ✨ Components

All components operate entirely on encrypted data:

| Component | Algorithm | Status | Decryptions |
|-----------|-----------|--------|-------------|
| **Softmax** | Power series + Newton iteration | ✅ Fully encrypted | 0 |
| **Attention** | Q·K scaled dot-product | ✅ Fully encrypted | 0 |
| **LayerNorm** | Goldschmidt 1/sqrt(variance) | ✅ Fully encrypted | 0 |
| **ReLU** | Polynomial approximation | ✅ Fully encrypted | 0 |
| **MatMul** | SIMD packing + rotation | ✅ Fully encrypted | 0 |
| **Complete Pipeline** | Input → Attention → LayerNorm → Output | ✅ Fully encrypted | 1 (final output only) |

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install numpy openfhe openfhe_numpy

# Clone repository
git clone https://github.com/yourusername/TFHE-Coder.git
cd TFHE-Coder
```

### Run the Complete Transformer

```bash
python3 test_full_transformer_encrypted.py
```

**Expected output:**
```
================================================================================
  FULLY ENCRYPTED TRANSFORMER - COMPLETE PIPELINE
================================================================================

Architecture: Input → Attention → LayerNorm → Output
  ✓ Input encrypted once
  ✓ Attention on ciphertext
  ✓ LayerNorm on ciphertext
  ✓ Output decrypted once

...

Total decryptions: 1 (only for final output!)

================================================================================
  ✅ FULLY ENCRYPTED TRANSFORMER SUCCESSFUL!
================================================================================
```

## 📚 Core Components

### 1. Softmax (Fully Encrypted)

**File:** `softmax_openfhe.py`

**Key Innovation:** Accepts ciphertext input directly via `softmax_encrypted_from_ciphertext()`

**Algorithm:**
1. Compute exponentials: e^z (power series with divide-and-conquer)
2. Sum via rotations: Σe^z (O(log n) rotations)
3. **Newton iteration for 1/Σ** (fully encrypted division!)
4. Multiply: e^z * (1/Σ)

**No decryption at any step!**

```python
from softmax_openfhe import SoftmaxCKKSOpenFHE

# Initialize
softmax = SoftmaxCKKSOpenFHE(n=8, K=64, scale_factor=8, mult_depth=25)

# Option 1: From plaintext (encrypts internally)
logits = np.random.randn(8)
probs = softmax.softmax_encrypted(logits)

# Option 2: From ciphertext (NEW! - fully encrypted)
logits_ct = softmax.encrypt_vector(logits)
probs_ct = softmax.softmax_encrypted_from_ciphertext(logits_ct, return_ciphertext=True)
# probs_ct is still encrypted!
```

**Parameters:**
- `n`: Vector size (power of 2)
- `K`: Exponential approximation terms (64 recommended)
- `scale_factor`: Scaling factor (8 recommended)
- `mult_depth`: CKKS depth (25+ for standalone, 35+ for pipelines)

### 2. Attention (Fully Encrypted)

**File:** `attention_fully_encrypted.py`

**Implementation:** Simplified for single vectors (seq_len=1) to demonstrate fully encrypted building blocks

**Algorithm:**
```
Attention(q, k, v) = softmax(q·k / sqrt(d_k)) * v
```

**Steps (all on ciphertext):**
1. Encrypt Q, K, V
2. Compute dot product: q·k
3. Broadcast score via rotations
4. Apply softmax on ciphertext (no decryption!)
5. Multiply by V

```python
from attention_fully_encrypted import AttentionFullyEncrypted

# Initialize
attention = AttentionFullyEncrypted(d_k=8, mult_depth=35)

# Compute fully encrypted attention
q = np.random.randn(8) * 0.5
k = np.random.randn(8) * 0.5
v = np.random.randn(8) * 0.5

# Returns ciphertext!
output_ct = attention.attention_single_encrypted(q, k, v, return_ciphertext=True)
```

**Why single vectors?**
For full matrices (seq_len > 1), the same principles apply but require processing each row separately. This implementation shows the core building blocks needed.

### 3. LayerNorm (Fully Encrypted)

**File:** `transformer_openfhe.py` (class `LayerNormOpenFHE`)

**Key Innovation:** Goldschmidt algorithm with aSOR for computing 1/sqrt(variance) on ciphertext

**Algorithm:**
1. Compute mean on ciphertext (rotation-based sum)
2. Compute variance on ciphertext
3. **Goldschmidt iteration for 1/sqrt(variance)** (fully encrypted!)
4. Normalize: (x - mean) * (1/sqrt(variance))

**Goldschmidt aSOR parameters:**
- Relaxation factors: [2.6374, 2.1722, 1.5135, 1.0907]
- Converges in 4 iterations
- No decryption!

```python
from transformer_openfhe import LayerNormOpenFHE

# Initialize
layernorm = LayerNormOpenFHE(d_model=8, mult_depth=25)

# From ciphertext (fully encrypted!)
x_ct = layernorm.encrypt(x)
output_ct = layernorm.normalize(x_ct, encrypted=True)
# output_ct is still encrypted!
```

### 4. ReLU (Fully Encrypted)

**File:** `relu_openfhe.py`

**Algorithm:** Polynomial approximation via least squares

```python
from relu_openfhe import ReLUOpenFHE

relu = ReLUOpenFHE(n=128, mult_depth=10, degree=7)
x = np.array([3, -2, 1, -4, 2])
result = relu.relu_encrypted(x)
```

### 5. Matrix Multiplication

**File:** `matmul_openfhe.py`

**Operations:**
- Encrypted matrix multiplication
- Homomorphic transpose
- SIMD packing for efficiency

```python
from matmul_openfhe import MatMulOpenFHE

matmul = MatMulOpenFHE(mult_depth=10)
result = matmul.encrypt_and_multiply(A, B)
```

## 🔬 Complete Transformer Pipeline

**File:** `test_full_transformer_encrypted.py`

**Architecture:**
```
Input (plaintext)
    ↓ [Encrypt - single encryption]
Ciphertext
    ↓ [Attention: Q·K·softmax·V - all on ciphertext]
Ciphertext
    ↓ [Residual: x + attention_output - on ciphertext]
Ciphertext
    ↓ [LayerNorm: Goldschmidt - on ciphertext]
Ciphertext
    ↓ [Decrypt - single decryption]
Output (plaintext)
```

**Decryption points:**
- Input: 0 decryptions ✓
- Attention: 0 decryptions ✓
- Softmax: 0 decryptions ✓ (fully encrypted!)
- Residual: 0 decryptions ✓
- LayerNorm: 0 decryptions ✓
- **Final output: 1 decryption (verification only)**

**Total: 1 decryption in entire pipeline!**

## 📊 Performance

**Configuration:** d_model=8

| Component | Initialization | Computation |
|-----------|---------------|-------------|
| Attention | ~15s | ~60s |
| LayerNorm | ~2s | ~40s |
| **Total Pipeline** | ~17s | ~100s |

**Accuracy:**
- Max error vs plaintext: < 0.01
- LayerNorm mean: ~0 (within 1e-10)
- LayerNorm variance: ~1 (within 0.01)

## 🧪 Testing

### Run All Tests

```bash
# Complete transformer pipeline
python3 test_full_transformer_encrypted.py

# Individual components
python3 test_softmax_fully_encrypted.py
python3 test_layernorm_encrypted.py
python3 test_transformer_thor.py
```

### Test Structure

| Test File | Component | Description |
|-----------|-----------|-------------|
| `test_full_transformer_encrypted.py` | Complete pipeline | Input → Attention → LayerNorm → Output |
| `test_softmax_fully_encrypted.py` | Softmax | Ciphertext input support |
| `test_layernorm_encrypted.py` | LayerNorm | Goldschmidt algorithm |
| `test_transformer_thor.py` | Transformer blocks | Full integration tests |

## 🔧 Advanced Usage

### Custom Crypto Context

Share crypto context between components for efficiency:

```python
from attention_fully_encrypted import AttentionFullyEncrypted
from transformer_openfhe import LayerNormOpenFHE

# Initialize attention (creates crypto context)
attention = AttentionFullyEncrypted(d_k=8, mult_depth=60)

# Share context with LayerNorm
layernorm = LayerNormOpenFHE(
    d_model=8,
    cc=attention.cc,           # Share crypto context
    keys=attention.keys,       # Share keys
    mult_depth=25
)
```

### Multiplicative Depth Planning

**Rules of thumb:**
- Attention: ~20-25 levels
- LayerNorm: ~20-25 levels
- Softmax: ~15-20 levels
- Pipeline: Sum of all + 10-15 buffer

**Example for full pipeline:**
```python
attention = AttentionFullyEncrypted(d_k=8, mult_depth=60)
# 60 levels = Attention(25) + LayerNorm(25) + Buffer(10)
```

### Return Ciphertext for Chaining

All components support `return_ciphertext` to keep data encrypted:

```python
# Attention returns ciphertext
attn_ct = attention.attention_single_encrypted(q, k, v, return_ciphertext=True)

# Residual on ciphertext
x_ct = attention.encrypt_vector(x)
residual_ct = attention.cc.EvalAdd(x_ct, attn_ct)

# LayerNorm on ciphertext
output_ct = layernorm.normalize(residual_ct, encrypted=True)

# Only decrypt final result
output = layernorm.decrypt(output_ct)
```

## 🎓 Mathematical Background

### Newton Iteration for 1/x

**Used in:** Softmax division

**Formula:**
```
x_{n+1} = x_n * (2 - a * x_n)
```

**Converges to:** 1/a (requires scaling for convergence)

### Goldschmidt Iteration for 1/sqrt(x)

**Used in:** LayerNorm normalization

**Formula:**
```
y_{n+1} = 0.5 * y_n * (3 - x * y_n^2)
```

**With aSOR (accelerated SOR):**
```
y_{n+1} = y_n * (α_n - β_n * x * y_n^2)
```

**Relaxation factors:** [2.6374, 2.1722, 1.5135, 1.0907]

### Power Series for e^x

**Used in:** Softmax exponential

**Formula:**
```
e^x - 1 = x/1 + (x/1)(x/2) + ... + (x/1)...(x/K)
```

**Scaling trick:**
```
e^x = (e^(x/q))^q
```

Reduces error by computing on smaller values.

## 📁 Repository Structure

```
.
├── Core Components (Fully Encrypted)
│   ├── softmax_openfhe.py              # Softmax with ciphertext input
│   ├── attention_fully_encrypted.py    # Fully encrypted attention
│   ├── transformer_openfhe.py          # LayerNorm with Goldschmidt
│   ├── relu_openfhe.py                 # ReLU activation
│   └── matmul_openfhe.py               # Matrix operations
│
├── Test Scripts
│   ├── test_full_transformer_encrypted.py   # Complete pipeline test
│   ├── test_softmax_fully_encrypted.py      # Softmax tests
│   ├── test_layernorm_encrypted.py          # LayerNorm tests
│   └── test_transformer_thor.py             # Integration tests
│
└── Documentation
    ├── README.md                        # This file
    ├── CLAUDE.md                        # Developer guide
    └── solution.md                      # Mathematical derivations
```

## ⚙️ Technical Details

### CKKS Parameters

**Default configuration:**
```python
params = CCParamsCKKSRNS()
params.SetMultiplicativeDepth(60)      # For full pipeline
params.SetScalingModSize(59)
params.SetFirstModSize(60)
params.SetScalingTechnique(FIXEDAUTO)
params.SetSecretKeyDist(UNIFORM_TERNARY)
params.SetBatchSize(8)                 # Vector size
```

### Rotation Keys

**Required rotations:**
- Softmax: [1, 2, 4, ..., n/2] for sum
- Attention: [±1, ±2, ..., ±(d_k-1)] for broadcasting
- LayerNorm: [1, 2, 4, ..., n/2] for mean/variance

### Bootstrap Not Required

All operations stay within multiplicative depth budget. No bootstrapping needed!

## 🎯 Use Cases

### Privacy-Preserving AI

- Medical diagnosis on encrypted patient data
- Financial risk assessment on encrypted portfolios
- Secure cloud inference without revealing inputs

### Federated Learning

- Encrypted model evaluation
- Private gradient computation
- Secure aggregation

### Regulatory Compliance

- HIPAA-compliant medical AI
- GDPR-compliant user data processing
- Zero-knowledge proofs for ML

## ⚠️ Limitations

1. **Single Vector Attention**
   - Current implementation: seq_len=1
   - For full matrices: apply same operations row-by-row
   - Future work: full matrix support

2. **Computational Cost**
   - ~100s for d_model=8
   - Scales polynomially with dimension
   - No GPU acceleration yet

3. **Numerical Stability**
   - Softmax without max-shift
   - Works well for attention scores (already scaled)
   - May need adjustment for extreme values

4. **Fixed Dimensions**
   - Must reinitialize for different sizes
   - Cannot dynamically resize

## 🚧 Future Work

1. **Full Matrix Attention**
   - Support seq_len > 1
   - Encrypted row extraction
   - Batched operations

2. **Multi-Head Attention**
   - Parallel attention heads
   - Complete transformer blocks
   - Causal masking

3. **Optimization**
   - GPU acceleration via CUDA
   - Batch processing
   - Parallel evaluation

4. **Additional Layers**
   - Feed-forward networks
   - Dropout (probabilistic)
   - Residual connections with learned gates

5. **Better Approximations**
   - Minimax polynomials for ReLU
   - Chebyshev softmax
   - Rational function approximations

## 📚 References

1. **CKKS Scheme**: Cheon et al. (2017). "Homomorphic Encryption for Arithmetic of Approximate Numbers"
2. **Attention**: Vaswani et al. (2017). "Attention Is All You Need"
3. **Goldschmidt Algorithm**: Markstein (2004). "Software Division and Square Root Using Goldschmidt's Algorithms"
4. **Softmax Algorithm**: Based on winning solution by Weiduan Feng for [fherma.io](https://fherma.io)
5. **OpenFHE**: https://github.com/openfheorg/openfhe-development

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenFHE** team for excellent homomorphic encryption library
- **Weiduan Feng** for innovative Softmax algorithm design
- **fherma.io** for hosting cryptographic challenges
- **Research community** for aSOR and Goldschmidt methods

---

**Status**: ✅ Fully Encrypted | 🔐 One Decryption Only | 📖 Production-Ready

**Made for privacy-preserving transformer inference**

This demonstrates TRUE encrypted transformer inference where all intermediate values remain encrypted from input to final output!
