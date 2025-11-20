"""
Comprehensive Test Suite for OpenFHE CKKS Softmax Implementation

This script thoroughly tests the softmax_openfhe.py implementation including:
- Accuracy validation
- Different vector configurations
- Parameter sensitivity
- Edge cases
- Performance benchmarking
"""

import numpy as np
import time
from typing import Tuple, List

try:
    from softmax_openfhe import SoftmaxCKKSOpenFHE, OPENFHE_AVAILABLE
except ImportError:
    print("Error: Could not import softmax_openfhe.py")
    exit(1)


def numpy_softmax(x: np.ndarray) -> np.ndarray:
    """Reference softmax implementation using numpy."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def compute_metrics(predicted: np.ndarray, reference: np.ndarray) -> dict:
    """Compute various error metrics."""
    abs_error = np.abs(predicted - reference)
    return {
        "max_error": np.max(abs_error),
        "mean_error": np.mean(abs_error),
        "median_error": np.median(abs_error),
        "std_error": np.std(abs_error),
        "sum_predicted": np.sum(predicted),
        "sum_reference": np.sum(reference),
    }


class TestSoftmaxOpenFHE:
    """Test suite for OpenFHE CKKS Softmax."""

    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def log_result(self, test_name: str, passed: bool, metrics: dict = None, message: str = ""):
        """Log test result."""
        status = "✅ PASSED" if passed else "❌ FAILED"
        self.results.append({
            "test": test_name,
            "status": status,
            "passed": passed,
            "metrics": metrics,
            "message": message
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def print_header(self, title: str):
        """Print test section header."""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)

    def print_metrics(self, metrics: dict):
        """Print error metrics."""
        print(f"  Max error:      {metrics['max_error']:.6e}")
        print(f"  Mean error:     {metrics['mean_error']:.6e}")
        print(f"  Median error:   {metrics['median_error']:.6e}")
        print(f"  Std error:      {metrics['std_error']:.6e}")
        print(f"  Sum (predicted): {metrics['sum_predicted']:.10f}")
        print(f"  Sum (reference): {metrics['sum_reference']:.10f}")

    def test_1_basic_functionality(self):
        """Test 1: Basic softmax computation on standard input."""
        self.print_header("Test 1: Basic Functionality")

        try:
            # Create softmax instance
            print("Initializing CKKS softmax...")
            softmax = SoftmaxCKKSOpenFHE(n=128, K=64, scale_factor=8, mult_depth=25)

            # Test vector
            np.random.seed(42)
            test_input = np.random.randn(128) * 2.0

            # Compute
            print("Computing encrypted softmax...")
            start_time = time.time()
            result = softmax.softmax_encrypted(test_input)
            elapsed = time.time() - start_time

            # Reference
            reference = numpy_softmax(test_input)

            # Metrics
            metrics = compute_metrics(result, reference)

            print(f"\nExecution time: {elapsed:.3f} seconds")
            print("\nAccuracy metrics:")
            self.print_metrics(metrics)

            # Check accuracy
            passed = (metrics['max_error'] < 1e-4 and
                     abs(metrics['sum_predicted'] - 1.0) < 1e-6)

            self.log_result("Basic Functionality", passed, metrics,
                          f"Time: {elapsed:.3f}s")

            if passed:
                print("\n✅ Test 1 PASSED")
            else:
                print("\n❌ Test 1 FAILED - Accuracy below threshold")

        except Exception as e:
            print(f"\n❌ Test 1 FAILED - Exception: {e}")
            self.log_result("Basic Functionality", False, message=str(e))

    def test_2_different_inputs(self):
        """Test 2: Various input distributions."""
        self.print_header("Test 2: Different Input Distributions")

        test_cases = [
            ("Uniform [0,1]", np.random.uniform(0, 1, 128)),
            ("Uniform [-5,5]", np.random.uniform(-5, 5, 128)),
            ("Normal(0,1)", np.random.randn(128)),
            ("Normal(0,3)", np.random.randn(128) * 3),
            ("All zeros", np.zeros(128)),
            ("All ones", np.ones(128)),
            ("One hot", np.array([1.0] + [0.0] * 127)),
            ("Sequential", np.arange(128) / 128.0),
        ]

        try:
            softmax = SoftmaxCKKSOpenFHE(n=128, K=64, scale_factor=8, mult_depth=25)

            all_passed = True
            for name, test_input in test_cases:
                print(f"\n  Testing: {name}")
                result = softmax.softmax_encrypted(test_input)
                reference = numpy_softmax(test_input)
                metrics = compute_metrics(result, reference)

                print(f"    Max error: {metrics['max_error']:.6e}, Sum: {metrics['sum_predicted']:.6f}")

                passed = (metrics['max_error'] < 1e-3 and
                         abs(metrics['sum_predicted'] - 1.0) < 1e-5)

                if not passed:
                    all_passed = False
                    print(f"    ❌ {name} failed")
                else:
                    print(f"    ✅ {name} passed")

            self.log_result("Different Input Distributions", all_passed)

            if all_passed:
                print("\n✅ Test 2 PASSED - All input distributions handled correctly")
            else:
                print("\n❌ Test 2 FAILED - Some distributions failed")

        except Exception as e:
            print(f"\n❌ Test 2 FAILED - Exception: {e}")
            self.log_result("Different Input Distributions", False, message=str(e))

    def test_3_parameter_sensitivity(self):
        """Test 3: Different parameter configurations."""
        self.print_header("Test 3: Parameter Sensitivity")

        configs = [
            (128, 32, 4, 20, "Fast (K=32, q=4)"),
            (128, 64, 8, 25, "Balanced (K=64, q=8)"),
            (128, 128, 16, 30, "Accurate (K=128, q=16)"),
        ]

        try:
            test_input = np.random.randn(128) * 2.0
            reference = numpy_softmax(test_input)

            all_passed = True
            for n, K, scale_factor, mult_depth, name in configs:
                print(f"\n  Testing: {name}")
                print(f"    Parameters: n={n}, K={K}, scale_factor={scale_factor}, depth={mult_depth}")

                start_time = time.time()
                softmax = SoftmaxCKKSOpenFHE(n=n, K=K, scale_factor=scale_factor,
                                            mult_depth=mult_depth)
                result = softmax.softmax_encrypted(test_input)
                elapsed = time.time() - start_time

                metrics = compute_metrics(result, reference)
                print(f"    Time: {elapsed:.3f}s, Max error: {metrics['max_error']:.6e}")

                passed = (metrics['max_error'] < 1e-2 and
                         abs(metrics['sum_predicted'] - 1.0) < 1e-4)

                if not passed:
                    all_passed = False
                    print(f"    ❌ {name} failed")
                else:
                    print(f"    ✅ {name} passed")

            self.log_result("Parameter Sensitivity", all_passed)

            if all_passed:
                print("\n✅ Test 3 PASSED - All parameter configurations work")
            else:
                print("\n❌ Test 3 FAILED - Some configurations failed")

        except Exception as e:
            print(f"\n❌ Test 3 FAILED - Exception: {e}")
            self.log_result("Parameter Sensitivity", False, message=str(e))

    def test_4_edge_cases(self):
        """Test 4: Edge cases and boundary conditions."""
        self.print_header("Test 4: Edge Cases")

        edge_cases = [
            ("Very large values", np.array([100.0] * 64 + [0.0] * 64)),
            ("Very small values", np.array([1e-6] * 128)),
            ("Mixed large/small", np.array([100.0, 1e-6] * 64)),
            ("Negative values", np.array([-10.0, -5.0, -1.0, 0.0] * 32)),
        ]

        try:
            softmax = SoftmaxCKKSOpenFHE(n=128, K=64, scale_factor=8, mult_depth=25)

            all_passed = True
            for name, test_input in edge_cases:
                print(f"\n  Testing: {name}")
                result = softmax.softmax_encrypted(test_input)
                reference = numpy_softmax(test_input)
                metrics = compute_metrics(result, reference)

                print(f"    Max error: {metrics['max_error']:.6e}, Sum: {metrics['sum_predicted']:.6f}")

                # More lenient threshold for edge cases
                passed = (metrics['max_error'] < 1e-2 and
                         abs(metrics['sum_predicted'] - 1.0) < 1e-3)

                if not passed:
                    all_passed = False
                    print(f"    ❌ {name} failed")
                else:
                    print(f"    ✅ {name} passed")

            self.log_result("Edge Cases", all_passed)

            if all_passed:
                print("\n✅ Test 4 PASSED - All edge cases handled")
            else:
                print("\n❌ Test 4 FAILED - Some edge cases failed")

        except Exception as e:
            print(f"\n❌ Test 4 FAILED - Exception: {e}")
            self.log_result("Edge Cases", False, message=str(e))

    def test_5_consistency(self):
        """Test 5: Consistency - same input should give same output."""
        self.print_header("Test 5: Consistency Check")

        try:
            softmax = SoftmaxCKKSOpenFHE(n=128, K=64, scale_factor=8, mult_depth=25)

            test_input = np.random.randn(128)

            # Run multiple times
            print("\n  Running softmax 3 times on same input...")
            results = []
            for i in range(3):
                result = softmax.softmax_encrypted(test_input)
                results.append(result)
                print(f"    Run {i+1}: sum = {np.sum(result):.10f}")

            # Check consistency
            diff_1_2 = np.max(np.abs(results[0] - results[1]))
            diff_2_3 = np.max(np.abs(results[1] - results[2]))

            print(f"\n  Max difference run 1-2: {diff_1_2:.6e}")
            print(f"  Max difference run 2-3: {diff_2_3:.6e}")

            passed = (diff_1_2 < 1e-4 and diff_2_3 < 1e-4)

            self.log_result("Consistency", passed)

            if passed:
                print("\n✅ Test 5 PASSED - Results are consistent")
            else:
                print("\n❌ Test 5 FAILED - Inconsistent results")

        except Exception as e:
            print(f"\n❌ Test 5 FAILED - Exception: {e}")
            self.log_result("Consistency", False, message=str(e))

    def test_6_individual_components(self):
        """Test 6: Test individual components."""
        self.print_header("Test 6: Component Testing")

        try:
            softmax = SoftmaxCKKSOpenFHE(n=128, K=64, scale_factor=8, mult_depth=25)

            # Test 6a: Encryption/Decryption
            print("\n  6a. Testing encryption/decryption")
            test_vec = np.random.randn(128)
            ct = softmax.encrypt_vector(test_vec)
            decrypted = softmax.decrypt_vector(ct)
            enc_error = np.max(np.abs(test_vec - decrypted))
            print(f"      Max error: {enc_error:.6e}")
            test_6a_passed = enc_error < 1e-4
            print(f"      {'✅ PASSED' if test_6a_passed else '❌ FAILED'}")

            # Test 6b: Rotation
            print("\n  6b. Testing rotation")
            ct_rotated = softmax.rotate(ct, 1)
            decrypted_rotated = softmax.decrypt_vector(ct_rotated)
            expected_rotated = np.roll(test_vec, 1)
            rot_error = np.max(np.abs(expected_rotated - decrypted_rotated))
            print(f"      Max error: {rot_error:.6e}")
            test_6b_passed = rot_error < 1e-4
            print(f"      {'✅ PASSED' if test_6b_passed else '❌ FAILED'}")

            # Test 6c: Exponential
            print("\n  6c. Testing exponential computation")
            small_input = np.random.randn(128) * 0.5  # Smaller values for better approximation
            ct_input = softmax.encrypt_vector(small_input)
            ct_exp = softmax.compute_exponential_encrypted(ct_input)
            result_exp = softmax.decrypt_vector(ct_exp)

            # Shift by max like in softmax
            expected_exp = np.exp(small_input - np.max(small_input))
            exp_error = np.max(np.abs(result_exp - expected_exp) / (expected_exp + 1e-10))
            print(f"      Max relative error: {exp_error:.6e}")
            test_6c_passed = exp_error < 0.1  # 10% relative error
            print(f"      {'✅ PASSED' if test_6c_passed else '❌ FAILED'}")

            # Test 6d: Sum with rotation
            print("\n  6d. Testing sum with rotation")
            ct_sum = softmax.sum_with_rotation_encrypted(ct)
            result_sum = softmax.decrypt_vector(ct_sum)
            expected_sum = np.sum(test_vec)
            sum_error = np.max(np.abs(result_sum - expected_sum))
            print(f"      Expected sum in all slots: {expected_sum:.6f}")
            print(f"      Got: {result_sum[0]:.6f}")
            print(f"      Max error: {sum_error:.6e}")
            test_6d_passed = sum_error < 1e-3
            print(f"      {'✅ PASSED' if test_6d_passed else '❌ FAILED'}")

            all_passed = test_6a_passed and test_6b_passed and test_6c_passed and test_6d_passed
            self.log_result("Component Testing", all_passed)

            if all_passed:
                print("\n✅ Test 6 PASSED - All components work correctly")
            else:
                print("\n❌ Test 6 FAILED - Some components failed")

        except Exception as e:
            print(f"\n❌ Test 6 FAILED - Exception: {e}")
            self.log_result("Component Testing", False, message=str(e))

    def test_7_performance_benchmark(self):
        """Test 7: Performance benchmarking."""
        self.print_header("Test 7: Performance Benchmark")

        try:
            print("\n  Benchmarking different configurations...")

            configs = [
                (128, 32, 4, 20, "Fast"),
                (128, 64, 8, 25, "Balanced"),
                (128, 128, 16, 30, "Accurate"),
            ]

            test_input = np.random.randn(128)

            for n, K, scale_factor, mult_depth, name in configs:
                print(f"\n  {name} configuration (K={K}, q={scale_factor}):")

                # Initialization time
                start = time.time()
                softmax = SoftmaxCKKSOpenFHE(n=n, K=K, scale_factor=scale_factor,
                                            mult_depth=mult_depth)
                init_time = time.time() - start
                print(f"    Initialization: {init_time:.3f}s")

                # Run multiple times for average
                times = []
                for _ in range(3):
                    start = time.time()
                    result = softmax.softmax_encrypted(test_input)
                    elapsed = time.time() - start
                    times.append(elapsed)

                avg_time = np.mean(times)
                std_time = np.std(times)
                print(f"    Computation: {avg_time:.3f}s ± {std_time:.3f}s")
                print(f"    Total (init + compute): {init_time + avg_time:.3f}s")

            self.log_result("Performance Benchmark", True,
                          message="Benchmark completed successfully")
            print("\n✅ Test 7 PASSED - Benchmark completed")

        except Exception as e:
            print(f"\n❌ Test 7 FAILED - Exception: {e}")
            self.log_result("Performance Benchmark", False, message=str(e))

    def print_summary(self):
        """Print test summary."""
        self.print_header("TEST SUMMARY")

        print(f"\nTotal tests run: {self.passed + self.failed}")
        print(f"Passed: {self.passed} ✅")
        print(f"Failed: {self.failed} ❌")
        print(f"\nSuccess rate: {100 * self.passed / (self.passed + self.failed):.1f}%")

        print("\n" + "-" * 80)
        print("Individual test results:")
        print("-" * 80)
        for result in self.results:
            status_symbol = "✅" if result['passed'] else "❌"
            print(f"{status_symbol} {result['test']}")
            if result['message']:
                print(f"   → {result['message']}")

        print("\n" + "=" * 80)
        if self.failed == 0:
            print("🎉 ALL TESTS PASSED! 🎉")
        else:
            print(f"⚠️  {self.failed} TEST(S) FAILED")
        print("=" * 80)

    def run_all_tests(self):
        """Run all tests in sequence."""
        print("\n" + "=" * 80)
        print("  COMPREHENSIVE TEST SUITE FOR OPENFHE CKKS SOFTMAX")
        print("=" * 80)

        if not OPENFHE_AVAILABLE:
            print("\n❌ ERROR: OpenFHE-NumPy is not available!")
            print("Please install it following the instructions in README.md")
            return

        print("\n✅ OpenFHE-NumPy is available")

        # Run all tests
        self.test_1_basic_functionality()
        self.test_2_different_inputs()
        self.test_3_parameter_sensitivity()
        self.test_4_edge_cases()
        self.test_5_consistency()
        self.test_6_individual_components()
        self.test_7_performance_benchmark()

        # Print summary
        self.print_summary()


def main():
    """Main entry point."""
    tester = TestSoftmaxOpenFHE()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
