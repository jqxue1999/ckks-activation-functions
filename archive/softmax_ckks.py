"""
CKKS-based Softmax Implementation using Numpy

This implementation follows the algorithm described in solution.md:
1. ExpMinus1: Compute exponential function using power series
2. Sum: Sum all exponentials using rotate-and-add
3. Division: Compute final softmax by dividing by the sum

The implementation uses numpy arrays and can be adapted to work with
actual CKKS ciphertexts by replacing operations with CKKS equivalents.
"""

import numpy as np
from typing import List


class SoftmaxCKKS:
    """
    Softmax implementation designed for CKKS homomorphic encryption.

    Parameters:
    - n: Vector size (must be power of 2, default 128)
    - K: Number of terms in exponential approximation (must be power of 2, default 256)
    - scale_factor: Scaling factor q for exponential computation (default 16)
    """

    def __init__(self, n: int = 128, K: int = 256, scale_factor: int = 16):
        assert n & (n - 1) == 0, "n must be a power of 2"
        assert K & (K - 1) == 0, "K must be a power of 2"

        self.n = n
        self.K = K
        self.scale_factor = scale_factor
        self.N = int(np.log2(K))

    def rotate(self, arr: np.ndarray, positions: int) -> np.ndarray:
        """
        Rotate array elements (simulates CKKS rotation).

        Args:
            arr: Input array
            positions: Number of positions to rotate

        Returns:
            Rotated array
        """
        return np.roll(arr, positions)

    def exp_minus_1(self, z: np.ndarray) -> np.ndarray:
        """
        Compute e^z - 1 using divide-and-conquer algorithm (Algorithm 1).

        Uses power series: e^x - 1 ≈ x/1 + x/1 * x/2 + ... + x/1 * x/2 * ... * x/K

        Args:
            z: Input vector (encrypted or plain)

        Returns:
            Approximation of e^z - 1 for each element
        """
        # Scale input by dividing by scale_factor for better precision
        z_scaled = z / self.scale_factor

        # Initialize T array: T[i] = z / (i+1) for i = 0, ..., K-1
        T = np.zeros((self.K, len(z_scaled)), dtype=np.float64)
        for i in range(self.K):
            T[i] = z_scaled / (i + 1)

        # Algorithm 1: ExpMinus1
        # First loop: process pairs (can be parallelized)
        for i in range(0, self.K, 2):
            T[i + 1] = T[i] * T[i + 1]
            T[i] = T[i] + T[i + 1]

        # Second loop: hierarchical combination
        m = 4
        while m <= self.K:
            for i in range(0, self.K, m):
                T[i + m - 1] = T[i + m - 1] * T[i + m // 2 - 1]
                T[i] = T[i] + T[i + m // 2] * T[i + m // 2 - 1]
            m *= 2

        # At this point, T[0] contains e^(z/scale_factor) - 1
        # Scale back by raising to power scale_factor: (e^(z/q))^q = e^z
        # If y = e^(z/q) - 1, then e^(z/q) = y + 1
        # We need to compute (y+1)^q - 1 = e^z - 1
        result = T[0]  # This is e^(z/q) - 1

        # Square log2(scale_factor) times to get e^z - 1
        # (y+1)^2 - 1 = y^2 + 2y + 1 - 1 = y^2 + 2y = y(y + 2)
        for _ in range(int(np.log2(self.scale_factor))):
            result = result * (result + 2)  # (y+1)^2 - 1 where y is current result

        return result

    def compute_exponential(self, z: np.ndarray) -> np.ndarray:
        """
        Compute e^z by computing e^z - 1 and adding 1.

        Args:
            z: Input vector

        Returns:
            e^z for each element
        """
        # Shift by maximum for numerical stability
        z_max = np.max(z)
        z_shifted = z - z_max

        exp_minus_1 = self.exp_minus_1(z_shifted)
        return exp_minus_1 + 1

    def sum_with_rotation(self, E: np.ndarray) -> np.ndarray:
        """
        Sum all elements using rotate-and-add (Algorithm 2).

        Returns a vector where each slot contains the sum of all elements.

        Args:
            E: Input vector (e.g., exponentials)

        Returns:
            Vector where each element is the sum of all input elements
        """
        S = E.copy()

        # Rotate and add: log2(n) iterations
        i = 1
        n_log = int(np.log2(len(E)))
        for _ in range(n_log):
            S = S + self.rotate(S, i)
            i *= 2

        return S

    def softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Compute softmax function over encrypted data.

        Softmax(z_i) = e^(z_i - max(z)) / sum(e^(z_j - max(z)))

        Args:
            z: Input vector (logits) - must be of size n (default 128)

        Returns:
            Softmax probabilities

        Note:
            For vectors smaller than n, pad with zeros and the result
            will include those padded positions. The algorithm is designed
            for fixed-size vectors to work with CKKS encryption.
        """
        # Ensure input is the right size
        if len(z) != self.n:
            # Pad with zeros if needed
            if len(z) < self.n:
                z_padded = np.zeros(self.n)
                z_padded[:len(z)] = z
                z = z_padded
            else:
                z = z[:self.n]

        # Step 1: Compute exponentials (with max shift for stability)
        E = self.compute_exponential(z)

        # Step 2: Sum all exponentials
        S = self.sum_with_rotation(E)

        # Step 3: Divide to get final softmax
        # In CKKS, this would use EvalDivide
        # S contains the same sum in every slot
        result = E / S

        return result


def test_softmax():
    """Test the CKKS softmax implementation against numpy's softmax."""
    print("Testing CKKS Softmax Implementation")
    print("=" * 70)

    # Numpy reference implementation
    def numpy_softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    # Test 1: Full 128-element vector (optimal use case)
    print("\n1. Testing with full 128-element vector")
    print("-" * 70)
    np.random.seed(42)
    z_large = np.random.randn(128) * 2

    softmax_ckks = SoftmaxCKKS(n=128, K=256, scale_factor=16)
    result_ckks_large = softmax_ckks.softmax(z_large)
    result_numpy_large = numpy_softmax(z_large)

    print(f"Input: 128 random values (mean={np.mean(z_large):.3f}, std={np.std(z_large):.3f})")
    print(f"\nMax absolute error: {np.max(np.abs(result_numpy_large - result_ckks_large)):.2e}")
    print(f"Mean absolute error: {np.mean(np.abs(result_numpy_large - result_ckks_large)):.2e}")
    print(f"Sum of CKKS probabilities: {np.sum(result_ckks_large):.10f}")
    print(f"Sum of Numpy probabilities: {np.sum(result_numpy_large):.10f}")

    # Show first few values
    print(f"\nFirst 5 values comparison:")
    print(f"Numpy:  {result_numpy_large[:5]}")
    print(f"CKKS:   {result_ckks_large[:5]}")

    # Test 2: Testing with different parameters
    print("\n" + "=" * 70)
    print("2. Testing with different approximation parameters")
    print("-" * 70)

    test_vector = np.random.randn(128)

    configs = [
        (128, 64, 8, "Lower precision (K=64, q=8)"),
        (128, 128, 16, "Medium precision (K=128, q=16)"),
        (128, 256, 16, "Higher precision (K=256, q=16)"),
        (128, 512, 32, "Highest precision (K=512, q=32)"),
    ]

    reference = numpy_softmax(test_vector)

    for n, K, q, desc in configs:
        sm = SoftmaxCKKS(n=n, K=K, scale_factor=q)
        result = sm.softmax(test_vector)
        error = np.max(np.abs(reference - result))
        print(f"{desc:35s} -> Max error: {error:.2e}")

    # Test 3: Small vector example (demonstrating padding behavior)
    print("\n" + "=" * 70)
    print("3. Small vector demonstration (n=8, showing padding behavior)")
    print("-" * 70)

    # Create a softmax instance for smaller vectors
    softmax_small = SoftmaxCKKS(n=8, K=64, scale_factor=8)
    z_small = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    result_ckks_small = softmax_small.softmax(z_small)
    result_numpy_small = numpy_softmax(z_small)

    print(f"Input vector: {z_small}")
    print(f"\nNumpy softmax:  {result_numpy_small}")
    print(f"CKKS softmax:   {result_ckks_small}")
    print(f"\nMax absolute error: {np.max(np.abs(result_numpy_small - result_ckks_small)):.2e}")
    print(f"Mean absolute error: {np.mean(np.abs(result_numpy_small - result_ckks_small)):.2e}")


if __name__ == "__main__":
    test_softmax()
