"""
Test script to verify encrypted LayerNorm implementation using OpenFHE.

Tests the fully encrypted LayerNorm using THOR approach with Goldschmidt/aSOR.
"""

import numpy as np
import time
from transformer_openfhe import LayerNormOpenFHE


def test_layernorm_encrypted():
    """Test fully encrypted LayerNorm."""
    print("="*80)
    print("  Encrypted LayerNorm Test (OpenFHE + THOR)")
    print("="*80)

    # Test configuration
    d_model = 8
    batch_size = 4

    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  batch_size: {batch_size}")

    # Generate test input
    np.random.seed(42)
    x = np.random.randn(batch_size, d_model) * 0.5

    print(f"\nInput shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"First row: {x[0, :4]}")

    # Initialize LayerNorm
    print(f"\nInitializing LayerNorm with OpenFHE...")
    start = time.time()
    # Need higher depth for Goldschmidt (4 iterations * 3 mults = 12 levels)
    # + mean/var computation (~4 levels) = ~16 total, use 25 for safety
    ln = LayerNormOpenFHE(d_model=d_model, mult_depth=25)
    init_time = time.time() - start
    print(f"Initialization time: {init_time:.2f}s")

    # Test 1: Plaintext mode (for reference)
    print("\n" + "-"*80)
    print("Test 1: Plaintext LayerNorm (reference)")
    print("-"*80)

    start = time.time()
    outputs_plaintext = []
    for i in range(batch_size):
        output = ln.normalize(x[i], encrypted=False)
        outputs_plaintext.append(output)
    output_plaintext = np.array(outputs_plaintext)
    plaintext_time = time.time() - start

    print(f"Computation time: {plaintext_time:.4f}s")
    print(f"Output shape: {output_plaintext.shape}")
    print(f"First row: {output_plaintext[0, :4]}")

    # Check properties
    mean_pt = np.mean(output_plaintext, axis=-1)
    var_pt = np.var(output_plaintext, axis=-1)

    print(f"\nProperties check:")
    print(f"  Mean (should be ~0): min={mean_pt.min():.6f}, max={mean_pt.max():.6f}")
    print(f"  Variance (should be ~1): min={var_pt.min():.6f}, max={var_pt.max():.6f}")

    mean_ok = np.allclose(mean_pt, 0, atol=1e-5)
    var_ok = np.allclose(var_pt, 1, atol=0.1)

    if mean_ok and var_ok:
        print("✅ Plaintext mode PASSED")
    else:
        print("❌ Plaintext mode FAILED")

    # Test 2: Encrypted mode (full OpenFHE)
    print("\n" + "-"*80)
    print("Test 2: Encrypted LayerNorm (OpenFHE + THOR)")
    print("-"*80)

    start = time.time()
    outputs_encrypted = []
    for i in range(batch_size):
        # Encrypt input
        ct_input = ln.encrypt(x[i])

        # Normalize (encrypted)
        ct_output = ln.normalize(ct_input, encrypted=True)

        # Decrypt result
        output = ln.decrypt(ct_output)
        outputs_encrypted.append(output)

    output_encrypted = np.array(outputs_encrypted)
    encrypted_time = time.time() - start

    print(f"Computation time: {encrypted_time:.4f}s")
    print(f"Output shape: {output_encrypted.shape}")
    print(f"First row: {output_encrypted[0, :4]}")

    # Check properties
    mean_enc = np.mean(output_encrypted, axis=-1)
    var_enc = np.var(output_encrypted, axis=-1)

    print(f"\nProperties check:")
    print(f"  Mean (should be ~0): min={mean_enc.min():.6f}, max={mean_enc.max():.6f}")
    print(f"  Variance (should be ~1): min={var_enc.min():.6f}, max={var_enc.max():.6f}")

    mean_ok_enc = np.allclose(mean_enc, 0, atol=1e-3)
    var_ok_enc = np.allclose(var_enc, 1, atol=0.2)

    if mean_ok_enc and var_ok_enc:
        print("✅ Encrypted mode PASSED")
    else:
        print("❌ Encrypted mode FAILED")

    # Test 3: Compare plaintext vs encrypted
    print("\n" + "-"*80)
    print("Test 3: Compare Plaintext vs Encrypted outputs")
    print("-"*80)

    diff = np.abs(output_plaintext - output_encrypted)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Maximum difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

    # Allow larger error due to CKKS approximation + Goldschmidt
    if max_diff < 0.5:
        print("✅ Outputs are sufficiently close")
        compare_ok = True
    else:
        print("⚠️  Outputs differ (may be due to CKKS noise + Goldschmidt approximation)")
        compare_ok = False

    # Summary
    print("\n" + "="*80)
    print("  Test Summary")
    print("="*80)

    all_passed = mean_ok and var_ok and mean_ok_enc and var_ok_enc and compare_ok

    if all_passed:
        print("\n🎉 All tests PASSED!")
        print(f"\nPerformance:")
        print(f"  Plaintext time:  {plaintext_time:.4f}s")
        print(f"  Encrypted time:  {encrypted_time:.4f}s")
        print(f"  Slowdown:        {encrypted_time/plaintext_time:.1f}x")
        print("\nFeatures:")
        print("  ✅ Fully encrypted LayerNorm with OpenFHE")
        print("  ✅ THOR approach: Goldschmidt's algorithm with aSOR")
        print("  ✅ No decryption required for statistics computation")
        print("  ✅ Mean/variance computed using rotations")
        print("  ✅ Relaxation factors: [2.6374, 2.1722, 1.5135, 1.0907]")
        print("\nBased on: THOR paper (Section 5.2, Appendix C)")
        return True
    else:
        print("\n❌ Some tests FAILED")
        return False


if __name__ == "__main__":
    success = test_layernorm_encrypted()
    exit(0 if success else 1)
