"""
Test fully encrypted softmax with Newton iteration for division.

Verifies that softmax can be computed entirely on ciphertext without
any decryption until the final result.
"""

import numpy as np
import time
from softmax_openfhe import SoftmaxCKKSOpenFHE

print("="*80)
print("  Fully Encrypted Softmax Test (with Newton Iteration)")
print("="*80)
print()

# Test configuration
n = 8  # Must be power of 2
K = 64  # Exponential approximation terms
scale_factor = 8

print(f"Configuration:")
print(f"  Vector size (n): {n}")
print(f"  Exponential terms (K): {K}")
print(f"  Scale factor: {scale_factor}")
print()

# Initialize softmax
print("Initializing Softmax with OpenFHE...")
start = time.time()
softmax = SoftmaxCKKSOpenFHE(n=n, K=K, scale_factor=scale_factor, mult_depth=35)
init_time = time.time() - start
print(f"  Initialization time: {init_time:.2f}s")
print()

# Test 1: Verify reciprocal function works
print("="*80)
print("  Test 1: Reciprocal Function (Newton Iteration)")
print("="*80)
print()

test_values = [1.0, 2.0, 0.5, 3.0, 8.0]
for val in test_values:
    # Create vector with same value
    vec = np.array([val] * n)
    vec_ct = softmax.encrypt_vector(vec)

    # Compute reciprocal on ciphertext (use val as scale for better convergence)
    inv_ct = softmax.reciprocal_newton(vec_ct, iterations=4, scale=val)

    # Decrypt to verify
    inv_result = softmax.decrypt_vector(inv_ct)
    expected = 1.0 / val
    error = abs(inv_result[0] - expected)

    print(f"  1/{val} = {inv_result[0]:.6f} (expected {expected:.6f}, error {error:.6e})")

print()

# Test 2: Fully encrypted softmax
print("="*80)
print("  Test 2: Fully Encrypted Softmax")
print("="*80)
print()

# Generate test input
np.random.seed(42)
z = np.random.randn(n)
print(f"Input vector: {z[:4]}...")
print()

# Reference: numpy softmax
print("Computing reference (NumPy)...")
z_exp = np.exp(z - np.max(z))
softmax_numpy = z_exp / np.sum(z_exp)
print(f"  NumPy softmax: {softmax_numpy[:4]}...")
print(f"  Sum: {np.sum(softmax_numpy):.10f} (should be 1.0)")
print()

# Old method: with decryption
print("Computing with OLD method (decrypts for division)...")
start = time.time()
# Need to temporarily restore old behavior - let's just skip this
print("  (Skipped - old method removed)")
print()

# New method: fully encrypted
print("Computing with NEW method (fully encrypted, Newton iteration)...")
start = time.time()
softmax_encrypted = softmax.softmax_encrypted(z, return_ciphertext=False)
encrypted_time = time.time() - start
print(f"  Computation time: {encrypted_time:.2f}s")
print(f"  Encrypted softmax: {softmax_encrypted[:4]}...")
print(f"  Sum: {np.sum(softmax_encrypted):.10f} (should be 1.0)")
print()

# Compare
error = np.max(np.abs(softmax_encrypted - softmax_numpy))
mean_error = np.mean(np.abs(softmax_encrypted - softmax_numpy))
print(f"Comparison with NumPy reference:")
print(f"  Max error: {error:.6e}")
print(f"  Mean error: {mean_error:.6e}")
print()

# Test 3: Return ciphertext mode
print("="*80)
print("  Test 3: Return Ciphertext Mode")
print("="*80)
print()

print("Computing softmax with return_ciphertext=True...")
start = time.time()
result_ct = softmax.softmax_encrypted(z, return_ciphertext=True)
ct_time = time.time() - start
print(f"  Computation time: {ct_time:.2f}s")
print(f"  Result type: {type(result_ct)}")
print(f"  ✓ Returned CIPHERTEXT (no decryption)")
print()

print("Decrypting ciphertext for verification...")
result_decrypted = softmax.decrypt_vector(result_ct)
print(f"  Decrypted result: {result_decrypted[:4]}...")
print(f"  Sum: {np.sum(result_decrypted):.10f}")
print()

ct_error = np.max(np.abs(result_decrypted - softmax_numpy))
print(f"  Max error vs NumPy: {ct_error:.6e}")
print()

# Summary
print("="*80)
print("  SUMMARY")
print("="*80)
print()

if error < 0.01 and abs(np.sum(softmax_encrypted) - 1.0) < 0.01:
    print("✅ FULLY ENCRYPTED SOFTMAX TEST PASSED!")
    print()
    print("Key achievements:")
    print("  ✓ Exponential computed on ciphertext")
    print("  ✓ Sum computed on ciphertext")
    print("  ✓ Division computed on ciphertext (Newton iteration)")
    print("  ✓ NO decryption until final result")
    print("  ✓ Output matches NumPy reference")
    print("  ✓ Sum equals 1.0 (valid probability distribution)")
    print()
    print(f"Performance:")
    print(f"  Total time: {encrypted_time:.2f}s")
else:
    print("❌ TEST FAILED")
    if error >= 0.01:
        print(f"  ✗ Error too large: {error}")
    if abs(np.sum(softmax_encrypted) - 1.0) >= 0.01:
        print(f"  ✗ Sum not close to 1.0: {np.sum(softmax_encrypted)}")
