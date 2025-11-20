"""
Test Suite for OpenFHE CKKS Softmax Implementation

This test file validates the softmax_openfhe.py implementation with:
- Basic functionality tests
- Accuracy verification
- Different input types
- Consistency checks
"""

import numpy as np
import time
import sys


def numpy_softmax(x):
    """Reference softmax implementation using NumPy."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_section(title):
    """Print a section header."""
    print(f"\n{title}")
    print_separator("-")


def test_basic_functionality():
    """Test 1: Basic softmax computation."""
    print_section("Test 1: Basic Functionality")

    from softmax_openfhe import SoftmaxCKKSOpenFHE, OPENFHE_AVAILABLE

    if not OPENFHE_AVAILABLE:
        print("❌ FAILED: OpenFHE not available")
        print("Install with: pip install openfhe openfhe_numpy")
        return False

    try:
        # Initialize
        print("Initializing CKKS softmax (n=128, K=64, scale_factor=8)...")
        start = time.time()
        softmax = SoftmaxCKKSOpenFHE(n=128, K=64, scale_factor=8, mult_depth=25)
        init_time = time.time() - start
        print(f"  Initialization: {init_time:.2f}s ✅")

        # Test input
        np.random.seed(42)
        test_input = np.random.randn(128) * 2.0
        print(f"  Input: 128 random values, range [{np.min(test_input):.2f}, {np.max(test_input):.2f}]")

        # Compute encrypted softmax
        print("Computing encrypted softmax...")
        start = time.time()
        result_encrypted = softmax.softmax_encrypted(test_input)
        compute_time = time.time() - start
        print(f"  Computation: {compute_time:.2f}s ✅")

        # Reference result
        result_reference = numpy_softmax(test_input)

        # Compute errors
        max_error = np.max(np.abs(result_encrypted - result_reference))
        mean_error = np.mean(np.abs(result_encrypted - result_reference))
        sum_encrypted = np.sum(result_encrypted)

        # Print results
        print("\nAccuracy:")
        print(f"  Max error:  {max_error:.6e}")
        print(f"  Mean error: {mean_error:.6e}")
        print(f"  Sum:        {sum_encrypted:.10f}")

        print("\nSample values (first 5):")
        print(f"  Encrypted: {result_encrypted[:5]}")
        print(f"  Reference: {result_reference[:5]}")

        # Check pass criteria
        passed = (max_error < 1e-4 and abs(sum_encrypted - 1.0) < 1e-6)

        if passed:
            print("\n✅ TEST 1 PASSED")
        else:
            print("\n❌ TEST 1 FAILED")

        return passed, softmax

    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_different_inputs(softmax):
    """Test 2: Different input distributions."""
    print_section("Test 2: Different Input Types")

    test_cases = [
        ("Random normal", np.random.randn(128)),
        ("All zeros", np.zeros(128)),
        ("All ones", np.ones(128)),
        ("Sequential", np.arange(128) / 128.0),
        ("One hot", np.array([10.0] + [0.0] * 127)),
        ("Extreme values", np.array([100.0, -100.0] * 64)),
    ]

    all_passed = True

    for name, test_input in test_cases:
        try:
            result = softmax.softmax_encrypted(test_input)
            reference = numpy_softmax(test_input)

            error = np.max(np.abs(result - reference))
            sum_val = np.sum(result)

            # More lenient threshold for extreme cases
            threshold = 1e-2 if "Extreme" in name else 1e-3
            passed = (error < threshold and abs(sum_val - 1.0) < 1e-4)

            status = "✅" if passed else "❌"
            print(f"  {name:<20} Error: {error:.2e}, Sum: {sum_val:.6f} {status}")

            if not passed:
                all_passed = False

        except Exception as e:
            print(f"  {name:<20} ❌ Exception: {e}")
            all_passed = False

    if all_passed:
        print("\n✅ TEST 2 PASSED")
    else:
        print("\n❌ TEST 2 FAILED")

    return all_passed


def test_consistency(softmax):
    """Test 3: Consistency check."""
    print_section("Test 3: Consistency")

    try:
        test_input = np.random.randn(128)

        print("Running softmax 3 times on same input...")
        results = []
        for i in range(3):
            result = softmax.softmax_encrypted(test_input)
            results.append(result)
            print(f"  Run {i+1}: sum = {np.sum(result):.10f}")

        # Check consistency
        diff_1_2 = np.max(np.abs(results[0] - results[1]))
        diff_2_3 = np.max(np.abs(results[1] - results[2]))

        print(f"\nDifferences:")
        print(f"  Run 1 vs 2: {diff_1_2:.2e}")
        print(f"  Run 2 vs 3: {diff_2_3:.2e}")

        passed = (diff_1_2 < 1e-4 and diff_2_3 < 1e-4)

        if passed:
            print("\n✅ TEST 3 PASSED")
        else:
            print("\n❌ TEST 3 FAILED")

        return passed

    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        return False


def test_correctness_verification():
    """Test 4: Detailed correctness verification."""
    print_section("Test 4: Correctness Verification")

    from softmax_openfhe import SoftmaxCKKSOpenFHE

    try:
        print("Initializing...")
        softmax = SoftmaxCKKSOpenFHE(n=128, K=64, scale_factor=8, mult_depth=25)

        # Simple example for manual verification
        simple_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0] + [0.0] * 123)

        print("\nInput (first 5): [1.0, 2.0, 3.0, 4.0, 5.0]")
        print("Computing...")

        result_openfhe = softmax.softmax_encrypted(simple_input)
        result_numpy = numpy_softmax(simple_input)

        print("\nDetailed comparison (first 5 values):")
        print(f"{'Index':<8} {'OpenFHE':<18} {'NumPy':<18} {'Diff':<12} {'Status'}")
        print_separator("-")

        max_error = 0
        for i in range(5):
            diff = abs(result_openfhe[i] - result_numpy[i])
            max_error = max(max_error, diff)
            status = "✅" if diff < 1e-6 else "❌"
            print(f"{i:<8} {result_openfhe[i]:<18.12f} {result_numpy[i]:<18.12f} {diff:<12.2e} {status}")

        print_separator("-")
        print(f"Sum (OpenFHE): {np.sum(result_openfhe):.12f}")
        print(f"Sum (NumPy):   {np.sum(result_numpy):.12f}")
        print(f"Max error:     {max_error:.2e}")

        passed = (max_error < 1e-6 and abs(np.sum(result_openfhe) - 1.0) < 1e-6)

        if passed:
            print("\n✅ TEST 4 PASSED - Results are correct!")
        else:
            print("\n❌ TEST 4 FAILED")

        return passed

    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        return False


def main():
    """Run all tests."""
    print_separator()
    print("  OPENFHE CKKS SOFTMAX - TEST SUITE")
    print_separator()

    results = []

    # Test 1: Basic functionality
    result1, softmax = test_basic_functionality()
    results.append(("Basic Functionality", result1))

    if not result1:
        print("\n⚠️  Stopping tests - basic functionality failed")
        sys.exit(1)

    # Test 2: Different inputs
    result2 = test_different_inputs(softmax)
    results.append(("Different Inputs", result2))

    # Test 3: Consistency
    result3 = test_consistency(softmax)
    results.append(("Consistency", result3))

    # Test 4: Correctness verification
    result4 = test_correctness_verification()
    results.append(("Correctness Verification", result4))

    # Summary
    print_separator()
    print("  TEST SUMMARY")
    print_separator()

    passed = 0
    total = len(results)

    for test_name, test_result in results:
        if isinstance(test_result, tuple):
            test_result = test_result[0]
        status = "✅ PASSED" if test_result else "❌ FAILED"
        print(f"  {test_name:<30} {status}")
        if test_result:
            passed += 1

    print_separator()
    print(f"\n  Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 ALL TESTS PASSED! 🎉")
        print_separator()
        sys.exit(0)
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed")
        print_separator()
        sys.exit(1)


if __name__ == "__main__":
    main()
