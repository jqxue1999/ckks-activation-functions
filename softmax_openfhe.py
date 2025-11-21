"""
CKKS Softmax Implementation using OpenFHE-NumPy

This implementation uses the openfhe-numpy library to perform
softmax computation on actual CKKS encrypted data.
"""

import numpy as np

try:
    import openfhe_numpy as onp
    from openfhe import *

    OPENFHE_AVAILABLE = True
except ImportError as e:
    OPENFHE_AVAILABLE = False
    print(f"Warning: OpenFHE-NumPy not available. Error: {e}")


class SoftmaxCKKSOpenFHE:
    """
    Softmax implementation using OpenFHE CKKS encryption.

    This class mirrors the structure of SoftmaxCKKS but uses actual
    encrypted operations via OpenFHE.
    """

    def __init__(
        self, n: int = 128, K: int = 256, scale_factor: int = 16, mult_depth: int = 20
    ):
        """
        Initialize CKKS softmax with OpenFHE context.

        Args:
            n: Vector size (must be power of 2)
            K: Number of terms in exponential approximation
            scale_factor: Scaling factor for exponential
            mult_depth: Multiplicative depth for CKKS
        """
        if not OPENFHE_AVAILABLE:
            raise ImportError("OpenFHE-NumPy is required to use this class")

        assert n & (n - 1) == 0, "n must be a power of 2"
        assert K & (K - 1) == 0, "K must be a power of 2"

        self.n = n
        self.K = K
        self.scale_factor = scale_factor
        self.N = int(np.log2(K))

        # Initialize CKKS context
        self.cc, self.keys = self._initialize_ckks(mult_depth)

    def _initialize_ckks(self, mult_depth):
        """Initialize CKKS crypto context and keys."""
        # Set CKKS parameters
        params = CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(mult_depth)
        params.SetScalingModSize(59)
        params.SetFirstModSize(60)
        params.SetScalingTechnique(FIXEDAUTO)
        params.SetSecretKeyDist(UNIFORM_TERNARY)
        params.SetBatchSize(self.n)

        # Generate crypto context
        cc = GenCryptoContext(params)
        cc.Enable(PKESchemeFeature.PKE)
        cc.Enable(PKESchemeFeature.LEVELEDSHE)
        cc.Enable(PKESchemeFeature.ADVANCEDSHE)

        # Generate keys
        keys = cc.KeyGen()
        cc.EvalMultKeyGen(keys.secretKey)
        cc.EvalSumKeyGen(keys.secretKey)

        # Generate rotation keys for all required rotations
        # We need rotations for: 1, 2, 4, 8, ..., n/2
        rotation_indices = []
        i = 1
        while i < self.n:
            rotation_indices.append(i)
            i *= 2
        cc.EvalRotateKeyGen(keys.secretKey, rotation_indices)

        return cc, keys

    def encrypt_vector(self, plaintext_vector):
        """
        Encrypt a plaintext vector.

        Args:
            plaintext_vector: numpy array of size n

        Returns:
            Encrypted ciphertext
        """
        if len(plaintext_vector) != self.n:
            # Pad if needed
            padded = np.zeros(self.n)
            padded[: len(plaintext_vector)] = plaintext_vector
            plaintext_vector = padded

        plaintext = self.cc.MakeCKKSPackedPlaintext(plaintext_vector.tolist())
        return self.cc.Encrypt(self.keys.publicKey, plaintext)

    def decrypt_vector(self, ciphertext):
        """
        Decrypt a ciphertext to plaintext vector.

        Args:
            ciphertext: Encrypted ciphertext

        Returns:
            Decrypted numpy array
        """
        plaintext = self.cc.Decrypt(self.keys.secretKey, ciphertext)
        plaintext.SetLength(self.n)
        result = plaintext.GetRealPackedValue()
        return np.array(result[: self.n])

    def rotate(self, ciphertext, positions):
        """
        Rotate ciphertext elements.

        Args:
            ciphertext: Encrypted ciphertext
            positions: Number of positions to rotate

        Returns:
            Rotated ciphertext
        """
        return self.cc.EvalRotate(ciphertext, positions)

    def exp_minus_1_encrypted(self, z_ct):
        """
        Compute e^z - 1 on encrypted data using Algorithm 1.

        Args:
            z_ct: Encrypted input vector

        Returns:
            Ciphertext containing e^z - 1
        """
        # Scale input by dividing by scale_factor
        # IMPORTANT: Broadcast scalar to all slots!
        scale_factor_pt = self.cc.MakeCKKSPackedPlaintext(
            [1.0 / self.scale_factor] * self.n
        )
        z_scaled_ct = self.cc.EvalMult(z_ct, scale_factor_pt)

        # Initialize T array of ciphertexts: T[i] = z / (i+1)
        T = []
        for i in range(self.K):
            divisor = 1.0 / (i + 1)
            # Broadcast divisor to all slots
            divisor_pt = self.cc.MakeCKKSPackedPlaintext([divisor] * self.n)
            T.append(self.cc.EvalMult(z_scaled_ct, divisor_pt))

        # Algorithm 1: ExpMinus1
        # First loop: process pairs (parallelizable)
        for i in range(0, self.K, 2):
            T[i + 1] = self.cc.EvalMult(T[i], T[i + 1])
            T[i] = self.cc.EvalAdd(T[i], T[i + 1])

        # Second loop: hierarchical combination
        m = 4
        while m <= self.K:
            for i in range(0, self.K, m):
                T[i + m - 1] = self.cc.EvalMult(T[i + m - 1], T[i + m // 2 - 1])
                temp = self.cc.EvalMult(T[i + m // 2], T[i + m // 2 - 1])
                T[i] = self.cc.EvalAdd(T[i], temp)
            m *= 2

        # Scale back by squaring log2(scale_factor) times
        result = T[0]
        # Broadcast 2.0 to all slots
        two_pt = self.cc.MakeCKKSPackedPlaintext([2.0] * self.n)

        for _ in range(int(np.log2(self.scale_factor))):
            # result = result * (result + 2)
            temp = self.cc.EvalAdd(result, two_pt)
            result = self.cc.EvalMult(result, temp)

        return result

    def compute_exponential_encrypted(self, z_ct):
        """
        Compute e^z on encrypted data.

        Args:
            z_ct: Encrypted input vector

        Returns:
            Ciphertext containing e^z
        """
        # Compute e^z - 1
        exp_minus_1_ct = self.exp_minus_1_encrypted(z_ct)

        # Add 1 to get e^z (broadcast 1.0 to all slots)
        one_pt = self.cc.MakeCKKSPackedPlaintext([1.0] * self.n)
        return self.cc.EvalAdd(exp_minus_1_ct, one_pt)

    def sum_with_rotation_encrypted(self, E_ct):
        """
        Sum all elements using rotate-and-add on encrypted data.

        Args:
            E_ct: Encrypted vector

        Returns:
            Ciphertext where each slot contains the sum
        """
        S_ct = E_ct

        # Rotate and add: log2(n) iterations
        i = 1
        n_log = int(np.log2(self.n))
        for _ in range(n_log):
            rotated = self.rotate(S_ct, i)
            S_ct = self.cc.EvalAdd(S_ct, rotated)
            i *= 2

        return S_ct

    def reciprocal_newton(self, a_ct, iterations=4, scale=None):
        """
        Compute 1/a on encrypted data using Newton iteration with scaling.

        Newton iteration for reciprocal:
            x_{n+1} = x_n * (2 - a * x_n)

        Converges to 1/a when starting from x_0 = 1, but only if a is close to 1.
        For other values, we scale first.

        Args:
            a_ct: Encrypted value (ciphertext) - should be positive
            iterations: Number of Newton iterations (default 4)
            scale: Scaling factor (if None, uses self.n for softmax sums)

        Returns:
            Ciphertext containing 1/a
        """
        if scale is None:
            scale = float(self.n)  # Softmax sum is typically close to n

        # Scale input: a_scaled = a / scale
        # This brings the value closer to 1 for better convergence
        scale_pt = self.cc.MakeCKKSPackedPlaintext([1.0 / scale] * self.n)
        a_scaled_ct = self.cc.EvalMult(a_ct, scale_pt)

        # Initial guess: x_0 = 1 (works well for a_scaled near 1)
        one_pt = self.cc.MakeCKKSPackedPlaintext([1.0] * self.n)
        x = self.cc.Encrypt(self.keys.publicKey, one_pt)

        # Prepare constant 2.0 for iteration
        two_pt = self.cc.MakeCKKSPackedPlaintext([2.0] * self.n)

        # Newton iteration: x_{n+1} = x_n * (2 - a_scaled * x_n)
        # This computes 1/a_scaled
        for _ in range(iterations):
            # Compute a_scaled * x_n
            ax = self.cc.EvalMult(a_scaled_ct, x)

            # Compute 2 - a_scaled * x_n
            two_minus_ax = self.cc.EvalSub(two_pt, ax)

            # Compute x_{n+1} = x_n * (2 - a_scaled * x_n)
            x = self.cc.EvalMult(x, two_minus_ax)

        # x now contains 1/a_scaled = 1/(a/scale) = scale/a
        # To get 1/a, divide by scale: (scale/a) / scale = 1/a
        inv_scale_pt = self.cc.MakeCKKSPackedPlaintext([1.0 / scale] * self.n)
        result = self.cc.EvalMult(x, inv_scale_pt)

        return result

    def softmax_encrypted_from_ciphertext(self, z_ct, return_ciphertext=False):
        """
        Compute softmax on ciphertext input (no max-shift).

        Args:
            z_ct: Encrypted input vector (ciphertext)
            return_ciphertext: If True, return ciphertext; else plaintext

        Returns:
            Softmax result
        """
        # Step 1: Compute exponentials (fully encrypted)
        E_ct = self.compute_exponential_encrypted(z_ct)

        # Step 2: Sum all exponentials (fully encrypted)
        S_ct = self.sum_with_rotation_encrypted(E_ct)

        # Step 3: Compute 1/S using Newton iteration (fully encrypted!)
        inv_S_ct = self.reciprocal_newton(S_ct, iterations=4)

        # Step 4: Multiply E * (1/S) to get softmax
        result_ct = self.cc.EvalMult(E_ct, inv_S_ct)

        if return_ciphertext:
            return result_ct
        else:
            # Decrypt final result
            return self.decrypt_vector(result_ct)

    def softmax_encrypted(self, z, return_ciphertext=False):
        """
        Compute softmax on encrypted data using fully homomorphic operations.

        Args:
            z: Plaintext input vector (will be encrypted)
            return_ciphertext: If True, return ciphertext; if False, decrypt and return plaintext

        Returns:
            Softmax result - ciphertext if return_ciphertext=True, else plaintext
        """
        # Shift by maximum for stability
        z_max = np.max(z)
        z_shifted = z - z_max

        # Encrypt the shifted input
        z_ct = self.encrypt_vector(z_shifted)

        # Step 1: Compute exponentials (fully encrypted)
        E_ct = self.compute_exponential_encrypted(z_ct)

        # Step 2: Sum all exponentials (fully encrypted)
        S_ct = self.sum_with_rotation_encrypted(E_ct)

        # Step 3: Compute 1/S using Newton iteration (fully encrypted!)
        inv_S_ct = self.reciprocal_newton(S_ct, iterations=4)

        # Step 4: Multiply E * (1/S) to get softmax
        result_ct = self.cc.EvalMult(E_ct, inv_S_ct)

        if return_ciphertext:
            return result_ct
        else:
            # Decrypt final result
            return self.decrypt_vector(result_ct)


def test_openfhe_softmax():
    """Test OpenFHE CKKS softmax implementation."""
    if not OPENFHE_AVAILABLE:
        print("OpenFHE-NumPy not available. Skipping test.")
        return

    print("Testing OpenFHE CKKS Softmax Implementation")
    print("=" * 70)

    # Reference softmax
    def numpy_softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    # Create test vector
    np.random.seed(42)
    test_vector = np.random.randn(128) * 2

    print(f"\\nInitializing CKKS context...")
    # Note: Using smaller K for faster testing
    softmax = SoftmaxCKKSOpenFHE(n=128, K=64, scale_factor=8, mult_depth=25)

    print(f"Computing encrypted softmax...")
    result_encrypted = softmax.softmax_encrypted(test_vector)
    result_reference = numpy_softmax(test_vector)

    print(f"\\nResults:")
    print(f"First 10 values (encrypted): {result_encrypted[:10]}")
    print(f"First 10 values (reference): {result_reference[:10]}")

    error = np.max(np.abs(result_encrypted - result_reference))
    print(f"\\nMax absolute error: {error:.6f}")
    print(f"Sum of encrypted softmax: {np.sum(result_encrypted):.6f}")
    print(f"Sum of reference softmax: {np.sum(result_reference):.6f}")


if __name__ == "__main__":
    if OPENFHE_AVAILABLE:
        test_openfhe_softmax()
    else:
        print(
            "OpenFHE-NumPy not installed. Please install it following instructions in:"
        )
        print("  openfhe-numpy/README.md")
        print("\\nOr use softmax_ckks.py for numpy-based simulation.")
