# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains a numpy-based implementation of the Softmax function designed for CKKS homomorphic encryption. The implementation is based on the winning solution for the Softmax Function challenge on fherma.io.

## Project Structure

```
.
├── softmax_ckks.py      # Core implementation
├── example_usage.py     # Usage examples and demonstrations
├── solution.md          # Mathematical explanation and algorithm details
├── README.md            # User documentation
└── CLAUDE.md           # This file
```

## Core Implementation

### Main Class: `SoftmaxCKKS`

Located in `softmax_ckks.py`, implements three key algorithms:

1. **exp_minus_1()** - Exponential computation using divide-and-conquer
   - Lines 49-94: Power series approximation with scaling
   - Key formula: `(y+1)^2 - 1 = y(y + 2)` for squaring in e^z - 1 form

2. **sum_with_rotation()** - Parallel sum using rotations
   - Lines 117-134: O(log n) rotate-and-add operations
   - Critical for CKKS: uses np.roll() which maps to EvalRotate()

3. **softmax()** - Main entry point
   - Lines 136-174: Orchestrates the three steps
   - Includes max-shift for numerical stability

### Algorithm Parameters

- `n`: Vector size (must be power of 2, typically 128)
- `K`: Exponential approximation terms (typically 256)
- `scale_factor`: Scaling for precision (typically 16)

Trade-offs:
- Higher K, scale_factor → better accuracy, more computation
- Lower K, scale_factor → faster, less accurate

## Testing and Validation

Run tests with:
```bash
python3 softmax_ckks.py
```

Expected results:
- Error < 10^-16 for full 128-element vectors
- Sum of probabilities ≈ 1.0

Run examples:
```bash
python3 example_usage.py
```

## Common Development Tasks

### Adding New Features

When extending functionality:
1. Maintain power-of-2 requirements for n and K
2. Keep operations CKKS-compatible (add, mult, rotate, divide)
3. Test with full-size vectors (n=128) for accurate validation
4. Document parameter impact on accuracy/performance

### Debugging Issues

Common issues:
- **Overflow warnings**: Usually from very negative padding values, safe to ignore if limited to padding
- **Poor accuracy on small vectors**: Algorithm designed for n=128, smaller vectors need padding
- **Sum != 1.0**: Check input size matches n parameter

### Performance Optimization

Current complexity:
- ExpMinus1: O(K) operations, O(log K) depth
- Sum: O(log n) rotations
- Division: O(1) for numpy, O(d) for CKKS Newton iteration

Optimization opportunities:
- Parallelize loops in exp_minus_1() (lines 71-73)
- Batch multiple softmax computations
- Reduce K for non-critical applications

## CKKS Mapping

This numpy implementation directly maps to CKKS operations:

| Operation | Current (Numpy) | CKKS Equivalent |
|-----------|----------------|-----------------|
| Rotation | `np.roll()` | `EvalRotate()` |
| Addition | `+` | `EvalAdd()` |
| Multiplication | `*` | `EvalMult()` + Rescale |
| Division | `/` | `EvalDivide()` (Newton iteration) |

To port to actual CKKS (OpenFHE, SEAL, etc.):
1. Replace `np.ndarray` with ciphertext type
2. Replace operations with evaluation functions
3. Add rescaling after multiplications
4. Handle modulus switching
5. The algorithm logic stays identical

## Mathematical Background

Key concepts from solution.md:

1. **Power Series Approximation**:
   ```
   e^x - 1 ≈ x/1 + (x/1)(x/2) + ... + (x/1)...(x/K)
   ```

2. **Scaling Identity**:
   ```
   e^x = (e^(x/q))^q
   ```
   Used to reduce error by computing on smaller values

3. **Stability Trick**:
   ```
   Softmax(z) = Softmax(z - max(z))
   ```
   Prevents overflow and bounds the sum

4. **Divide-and-Conquer Structure**:
   - First loop: Process pairs (parallelizable)
   - Second loop: Hierarchical combination
   - O(log K) multiplicative depth

## Use Cases

Designed for:
- Private neural network inference (final layer)
- Transformer attention mechanisms
- Multi-class classification
- Any softmax over encrypted data

## Important Notes

- Algorithm requires n to be power of 2
- Optimal for n=128 as designed
- Smaller vectors should be padded to n
- Accuracy is machine-precision with recommended parameters
- Implementation is encryption-scheme agnostic in structure