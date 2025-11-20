"""
Test Suite for OpenFHE CKKS ReLU Implementation

This test file validates the relu_openfhe.py implementation with:
- Basic functionality tests
- Approximation accuracy verification
- Different input ranges
- Performance benchmarking
"""

import numpy as np
import time
import sys


def numpy_relu(x):
    """Reference ReLU implementation."""
    return np.maximum(0, x)


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_section(title):
    """Print a section header."""
    print(f"\n{title}")
    print_separator("-")


def compute_relu_metrics(predicted, reference):
    """Compute ReLU-specific metrics."""
    abs_error = np.abs(predicted - reference)

    # Separate errors for positive and negative regions
    positive_mask = reference > 0
    negative_mask = reference == 0

    metrics = {
        "max_error": np.max(abs_error),
        "mean_error": np.mean(abs_error),
        "max_error_positive": np.max(abs_error[positive_mask]) if np.any(positive_mask) else 0,
        "max_error_negative": np.max(abs_error[negative_mask]) if np.any(negative_mask) else 0,
    }
    return metrics


def test_basic_functionality():
    """Test 1: Basic ReLU computation."""
    print_section("Test 1: Basic Functionality")

    from relu_openfhe import ReLUOpenFHE, OPENFHE_AVAILABLE

    if not OPENFHE_AVAILABLE:
        print("❌ FAILED: OpenFHE not available")
        print("Install with: pip install openfhe openfhe_numpy")
        return False

    try:
        # Initialize
        print("Initializing CKKS ReLU (n=128, degree=7)...")
        start = time.time()
        relu = ReLUOpenFHE(n=128, mult_depth=10, degree=7)
        init_time = time.time() - start
        print(f"  Initialization: {init_time:.2f}s ✅")

        # Test input with clear positive and negative values
        test_input = np.array([3.0, -2.0, 1.0, -4.0, 2.0, -1.0, 4.0, -3.0] + [0.0] * 120)
        print(f"\n  Input (first 8): {test_input[:8]}")
        print(f"  Expected output: {numpy_relu(test_input[:8])}")

        # Compute ReLU
        print("\nComputing encrypted ReLU...")
        start = time.time()
        result = relu.relu_encrypted(test_input)
        compute_time = time.time() - start
        print(f"  Computation: {compute_time:.2f}s ✅")

        # Reference
        reference = numpy_relu(test_input)

        # Metrics
        metrics = compute_relu_metrics(result, reference)

        print("\nResults (first 8 values):")
        print(f"  OpenFHE:   {result[:8]}")
        print(f"  Reference: {reference[:8]}")

        print("\nAccuracy metrics:")
        print(f"  Max error (overall):  {metrics['max_error']:.4f}")
        print(f"  Mean error:           {metrics['mean_error']:.4f}")
        print(f"  Max error (positive): {metrics['max_error_positive']:.4f}")
        print(f"  Max error (negative): {metrics['max_error_negative']:.4f}")

        # Check pass criteria
        # ReLU polynomial approximation, so we're more lenient
        passed = (metrics['max_error'] < 1.0 and metrics['mean_error'] < 0.3)

        if passed:
            print("\n✅ TEST 1 PASSED - Approximation acceptable")
        else:
            print("\n❌ TEST 1 FAILED - Approximation error too large")

        return passed, relu

    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_different_ranges(relu):
    """Test 2: ReLU on different input ranges."""
    print_section("Test 2: Different Input Ranges")

    print("Note: Polynomial fitted for x ∈ [-5, 5]. Outside this range, expect larger errors.\n")

    test_cases = [
        ("Small positive", np.array([0.1, 0.2, 0.3, 0.4, 0.5] + [0.0] * 123), True),
        ("Medium positive", np.array([1.0, 2.0, 3.0, 4.0, 5.0] + [0.0] * 123), True),
        ("Small negative", np.array([-0.1, -0.2, -0.3, -0.4, -0.5] + [0.0] * 123), True),
        ("Medium negative", np.array([-1.0, -2.0, -3.0, -4.0, -5.0] + [0.0] * 123), True),
        ("Around zero", np.array([-0.5, -0.1, 0.0, 0.1, 0.5] + [0.0] * 123), True),
        ("Full range [-5,5]", np.linspace(-5, 5, 128), True),
        ("Out of range (large)", np.array([10.0, 15.0, 20.0] + [0.0] * 125), False),
    ]

    all_passed = True

    for name, test_input, in_range in test_cases:
        try:
            result = relu.relu_encrypted(test_input)
            reference = numpy_relu(test_input)

            metrics = compute_relu_metrics(result, reference)

            # Different thresholds for in-range vs out-of-range
            if in_range:
                # Strict criteria for in-range values
                passed = metrics['max_error'] < 0.5 and metrics['mean_error'] < 0.25
                status = "✅" if passed else "❌"
            else:
                # Lenient for out-of-range (just show it breaks down)
                passed = True  # Don't fail test for out-of-range
                status = "⚠️  (out of range)"

            print(f"  {name:<25} Max: {metrics['max_error']:.3f}, "
                  f"Mean: {metrics['mean_error']:.3f} {status}")

            if not passed:
                all_passed = False

        except Exception as e:
            print(f"  {name:<25} ❌ Exception: {e}")
            all_passed = False

    if all_passed:
        print("\n✅ TEST 2 PASSED - All ranges handled")
    else:
        print("\n⚠️  TEST 2 WARNING - Some ranges have large errors (expected for polynomial approx)")

    return all_passed


def test_approximation_quality():
    """Test 3: Detailed approximation quality analysis."""
    print_section("Test 3: Approximation Quality Analysis")

    from relu_openfhe import ReLUOpenFHE

    try:
        print("Testing different polynomial degrees...\n")

        degrees = [3, 5, 7, 9]
        test_input = np.linspace(-5, 5, 128)
        reference = numpy_relu(test_input)

        results = []
        for degree in degrees:
            print(f"  Degree {degree}:")
            relu = ReLUOpenFHE(n=128, mult_depth=15, degree=degree)

            start = time.time()
            result = relu.relu_encrypted(test_input)
            elapsed = time.time() - start

            metrics = compute_relu_metrics(result, reference)

            print(f"    Time: {elapsed:.2f}s")
            print(f"    Max error: {metrics['max_error']:.4f}")
            print(f"    Mean error: {metrics['mean_error']:.4f}")

            results.append((degree, metrics, elapsed))

        # Summary
        print("\n  Summary:")
        print(f"  {'Degree':<10} {'Max Error':<15} {'Mean Error':<15} {'Time (s)'}")
        print_separator("-")
        for degree, metrics, elapsed in results:
            print(f"  {degree:<10} {metrics['max_error']:<15.4f} "
                  f"{metrics['mean_error']:<15.4f} {elapsed:.2f}")

        # Best degree is 7 or 9
        best_error = min(r[1]['max_error'] for r in results)
        passed = best_error < 1.5

        if passed:
            print("\n✅ TEST 3 PASSED - Polynomial approximation quality acceptable")
        else:
            print("\n❌ TEST 3 FAILED - Approximation quality insufficient")

        return passed

    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """Test 4: Visualize approximation (text-based)."""
    print_section("Test 4: Approximation Visualization")

    from relu_openfhe import ReLUOpenFHE

    try:
        print("Computing ReLU approximation for x in [-5, 5]...\n")

        relu = ReLUOpenFHE(n=128, mult_depth=10, degree=7)

        # Test range
        x_vals = np.linspace(-5, 5, 128)
        result = relu.relu_encrypted(x_vals)
        reference = numpy_relu(x_vals)

        # Show sample points
        print("  Sample points:")
        print(f"  {'x':<10} {'Approx':<15} {'True':<15} {'Error':<10}")
        print_separator("-")

        sample_indices = [0, 25, 50, 63, 75, 100, 127]  # Include x=0
        for i in sample_indices:
            x = x_vals[i]
            approx = result[i]
            true_val = reference[i]
            error = abs(approx - true_val)
            print(f"  {x:<10.2f} {approx:<15.4f} {true_val:<15.4f} {error:<10.4f}")

        # Check critical point (around x=0)
        zero_idx = 63  # Middle point (should be close to 0)
        error_at_zero = abs(result[zero_idx] - reference[zero_idx])

        print(f"\n  Error at x≈0: {error_at_zero:.4f}")

        # Overall statistics
        max_error = np.max(np.abs(result - reference))
        mean_error = np.mean(np.abs(result - reference))

        print(f"  Overall max error: {max_error:.4f}")
        print(f"  Overall mean error: {mean_error:.4f}")

        passed = max_error < 1.5

        if passed:
            print("\n✅ TEST 4 PASSED - Approximation visualized successfully")
        else:
            print("\n❌ TEST 4 FAILED")

        return passed

    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        return False


def main():
    """Run all tests."""
    print_separator()
    print("  OPENFHE CKKS RELU - TEST SUITE")
    print_separator()
    print("\nNote: ReLU uses polynomial approximation in CKKS.")
    print("Expect some approximation error, especially for large |x|.")

    results = []

    # Test 1: Basic functionality
    result1, relu = test_basic_functionality()
    results.append(("Basic Functionality", result1))

    if not result1:
        print("\n⚠️  Stopping tests - basic functionality failed")
        sys.exit(1)

    # Test 2: Different ranges
    result2 = test_different_ranges(relu)
    results.append(("Different Ranges", result2))

    # Test 3: Approximation quality
    result3 = test_approximation_quality()
    results.append(("Approximation Quality", result3))

    # Test 4: Visualization
    result4 = test_visualization()
    results.append(("Visualization", result4))

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
        print("\n  Note: ReLU approximation has inherent error due to polynomial fitting.")
        print("  Typical error: <1.0 for x in [-5, 5]")
        print_separator()
        sys.exit(0)
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed")
        print_separator()
        sys.exit(1)


if __name__ == "__main__":
    main()
