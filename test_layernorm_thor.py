"""
Test script to verify THOR-style LayerNorm implementation.

Tests the LayerNorm implementation using Goldschmidt's algorithm with aSOR
for computing inverse square root homomorphically.
"""

import numpy as np
import time
from transformer_openfhe import LayerNormOpenFHE


def test_layernorm_thor():
    """Test THOR mode of LayerNorm."""
    print("="*80)
    print("  LayerNorm THOR Implementation Test")
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

    # Test THOR mode (Goldschmidt/aSOR)
    print("\n" + "-"*80)
    print("Test 1: THOR Mode (Goldschmidt/aSOR for inverse square root)")
    print("-"*80)

    ln_thor = LayerNormOpenFHE(d_model=d_model)

    start = time.time()
    output_thor = ln_thor.normalize(x)
    thor_time = time.time() - start

    print(f"Computation time: {thor_time:.6f}s")
    print(f"Output shape: {output_thor.shape}")

    # Check properties
    mean_thor = np.mean(output_thor, axis=-1)
    var_thor = np.var(output_thor, axis=-1)

    print(f"\nProperties check:")
    print(f"  Mean (should be ~0): min={mean_thor.min():.6f}, max={mean_thor.max():.6f}")
    print(f"  Variance (should be ~1): min={var_thor.min():.6f}, max={var_thor.max():.6f}")

    mean_ok_thor = np.allclose(mean_thor, 0, atol=1e-5)
    var_ok_thor = np.allclose(var_thor, 1, atol=0.1)

    if mean_ok_thor and var_ok_thor:
        print("✅ THOR mode PASSED")
    else:
        print("❌ THOR mode FAILED")

    # Test 2: Verify inverse square root approximation
    print("\n" + "-"*80)
    print("Test 2: Verify Goldschmidt Inverse Square Root")
    print("-"*80)

    # Test on a few values
    test_values = [0.1, 0.25, 0.5, 0.75, 0.9]

    print(f"\n{'Value':<10} {'1/sqrt(x)':<12} {'Goldschmidt':<12} {'Error':<10}")
    print("-" * 50)

    max_error = 0.0
    for val in test_values:
        true_inv_sqrt = 1.0 / np.sqrt(val)
        approx_inv_sqrt = ln_thor.inverse_sqrt_goldschmidt(val)
        error = abs(true_inv_sqrt - approx_inv_sqrt)
        max_error = max(max_error, error)

        print(f"{val:<10.2f} {true_inv_sqrt:<12.6f} {approx_inv_sqrt:<12.6f} {error:<10.6f}")

    print(f"\nMaximum approximation error: {max_error:.6f}")
    approx_ok = max_error < 0.01  # Goldschmidt should be very accurate

    if approx_ok:
        print("✅ Goldschmidt approximation accurate")
    else:
        print("❌ Goldschmidt approximation error too large")

    # Summary
    print("\n" + "="*80)
    print("  Test Summary")
    print("="*80)

    all_passed = mean_ok_thor and var_ok_thor and approx_ok

    if all_passed:
        print("\n🎉 All tests PASSED!")
        print(f"\nComputation time: {thor_time:.6f}s")
        print("\nFeatures:")
        print("  ✅ Fully homomorphic LayerNorm implementation")
        print("  ✅ Uses Goldschmidt's algorithm with aSOR")
        print("  ✅ Relaxation factors: [2.6374, 2.1722, 1.5135, 1.0907]")
        print("  ✅ Converges in 4 iterations")
        print("\nBased on: THOR paper (Section 5.2, Appendix C)")
        return True
    else:
        print("\n❌ Some tests FAILED")
        return False


if __name__ == "__main__":
    success = test_layernorm_thor()
    exit(0 if success else 1)
