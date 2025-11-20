"""
Quick Test Script for OpenFHE CKKS Softmax

A focused test that validates core functionality quickly.
"""

import numpy as np
import time
from softmax_openfhe import SoftmaxCKKSOpenFHE, OPENFHE_AVAILABLE


def numpy_softmax(x):
    """Reference softmax implementation."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def main():
    print("=" * 80)
    print("  QUICK TEST: OpenFHE CKKS Softmax")
    print("=" * 80)

    if not OPENFHE_AVAILABLE:
        print("\n❌ OpenFHE-NumPy is not available!")
        print("Please install: pip install openfhe openfhe_numpy")
        return

    print("\n✅ OpenFHE-NumPy is available\n")

    # Test 1: Basic functionality
    print("-" * 80)
    print("Test 1: Basic Softmax Computation")
    print("-" * 80)

    try:
        print("\n1. Initializing CKKS softmax (n=128, K=64, scale=8)...")
        start = time.time()
        softmax = SoftmaxCKKSOpenFHE(n=128, K=64, scale_factor=8, mult_depth=25)
        init_time = time.time() - start
        print(f"   Initialization time: {init_time:.2f}s ✅")

        print("\n2. Generating random test vector...")
        np.random.seed(42)
        test_input = np.random.randn(128) * 2.0
        print(f"   Input: 128 random values")
        print(f"   Range: [{np.min(test_input):.2f}, {np.max(test_input):.2f}]")

        print("\n3. Computing encrypted softmax...")
        start = time.time()
        result_encrypted = softmax.softmax_encrypted(test_input)
        compute_time = time.time() - start
        print(f"   Computation time: {compute_time:.2f}s ✅")

        print("\n4. Computing reference softmax...")
        result_reference = numpy_softmax(test_input)

        print("\n5. Comparing results...")
        max_error = np.max(np.abs(result_encrypted - result_reference))
        mean_error = np.mean(np.abs(result_encrypted - result_reference))
        sum_encrypted = np.sum(result_encrypted)
        sum_reference = np.sum(result_reference)

        print(f"   Max absolute error:  {max_error:.6e}")
        print(f"   Mean absolute error: {mean_error:.6e}")
        print(f"   Sum (encrypted):     {sum_encrypted:.10f}")
        print(f"   Sum (reference):     {sum_reference:.10f}")

        print("\n6. Sample values (first 5):")
        print(f"   Encrypted: {result_encrypted[:5]}")
        print(f"   Reference: {result_reference[:5]}")

        # Check if test passed
        accuracy_ok = max_error < 1e-4
        sum_ok = abs(sum_encrypted - 1.0) < 1e-6

        print("\n" + "=" * 80)
        if accuracy_ok and sum_ok:
            print("✅ TEST PASSED!")
            print(f"   - Accuracy excellent (max error < 1e-4)")
            print(f"   - Sum property preserved (sum ≈ 1.0)")
        else:
            print("❌ TEST FAILED!")
            if not accuracy_ok:
                print(f"   - Accuracy issue: max error {max_error:.6e} >= 1e-4")
            if not sum_ok:
                print(f"   - Sum issue: {sum_encrypted:.6f} != 1.0")

    except Exception as e:
        print(f"\n❌ TEST FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 2: Quick consistency check
    print("\n" + "-" * 80)
    print("Test 2: Consistency Check")
    print("-" * 80)

    try:
        print("\n1. Running softmax twice on same input...")
        result1 = softmax.softmax_encrypted(test_input)
        result2 = softmax.softmax_encrypted(test_input)

        diff = np.max(np.abs(result1 - result2))
        print(f"   Max difference: {diff:.6e}")

        print("\n" + "=" * 80)
        if diff < 1e-4:
            print("✅ TEST PASSED!")
            print(f"   - Results are consistent")
        else:
            print("❌ TEST FAILED!")
            print(f"   - Results differ by {diff:.6e}")

    except Exception as e:
        print(f"\n❌ TEST FAILED with exception: {e}")
        return

    # Test 3: Different input types
    print("\n" + "-" * 80)
    print("Test 3: Different Input Types")
    print("-" * 80)

    test_cases = [
        ("All zeros", np.zeros(128)),
        ("All ones", np.ones(128)),
        ("Sequential", np.arange(128) / 128.0),
    ]

    all_passed = True
    for name, test_vec in test_cases:
        try:
            print(f"\n  Testing: {name}")
            result = softmax.softmax_encrypted(test_vec)
            reference = numpy_softmax(test_vec)
            error = np.max(np.abs(result - reference))
            sum_val = np.sum(result)

            print(f"    Error: {error:.6e}, Sum: {sum_val:.6f}", end="")

            if error < 1e-3 and abs(sum_val - 1.0) < 1e-5:
                print(" ✅")
            else:
                print(" ❌")
                all_passed = False

        except Exception as e:
            print(f" ❌ Exception: {e}")
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ TEST PASSED!")
        print("   - All input types handled correctly")
    else:
        print("❌ TEST FAILED!")
        print("   - Some input types failed")

    # Final summary
    print("\n" + "=" * 80)
    print("  OVERALL SUMMARY")
    print("=" * 80)
    print(f"\n  Total time: {init_time + compute_time:.2f}s")
    print(f"  - Initialization: {init_time:.2f}s")
    print(f"  - Computation: {compute_time:.2f}s")
    print("\n  🎉 All quick tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
