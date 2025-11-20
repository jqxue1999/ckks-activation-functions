"""
ReLU Activation Function using OpenFHE CKKS Encryption

ReLU is defined as: f(x) = max(0, x) = { x if x > 0, 0 if x <= 0 }

Since CKKS doesn't support direct comparison, we approximate ReLU using:
1. Polynomial approximation (degree 3 or 5)
2. Smooth approximation methods

This implementation provides multiple approximation methods:
- Polynomial approximation (fast, good for range [-5, 5])
- High-degree polynomial (more accurate)
- Custom polynomial coefficients
"""

import numpy as np

try:
    from openfhe import *
    OPENFHE_AVAILABLE = True
except ImportError:
    OPENFHE_AVAILABLE = False
    print("Warning: OpenFHE not available. Error: OpenFHE module not found")


class ReLUOpenFHE:
    """
    ReLU activation function using CKKS homomorphic encryption.

    Uses polynomial approximation since CKKS doesn't support comparison operations.
    """

    def __init__(
        self,
        n: int = 128,
        mult_depth: int = 10,
        approximation_method: str = "polynomial",
        degree: int = 7
    ):
        """
        Initialize CKKS ReLU.

        Args:
            n: Vector size (must be power of 2)
            mult_depth: Multiplicative depth for CKKS
            approximation_method: "polynomial" or "custom"
            degree: Polynomial degree (3, 5, 7, or 9)
        """
        if not OPENFHE_AVAILABLE:
            raise ImportError("OpenFHE is required to use this class")

        self.n = n
        self.degree = degree
        self.approximation_method = approximation_method

        # Initialize CKKS context
        self.cc, self.keys = self._initialize_ckks(mult_depth)

        # Compute polynomial coefficients for ReLU approximation
        self.coefficients = self._compute_polynomial_coefficients()

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

        return cc, keys

    def _compute_polynomial_coefficients(self):
        """
        Compute polynomial coefficients for ReLU approximation.

        Uses pre-computed least squares fitting over the range [-5, 5].
        These coefficients were computed to minimize ||ReLU(x) - P(x)||^2
        """
        if self.approximation_method == "polynomial":
            # Pre-computed coefficients from least squares fitting
            # Fitted to approximate max(0, x) for x in [-5, 5]
            # Computed using scipy.optimize.curve_fit

            if self.degree == 3:
                # Degree 3: fast, less accurate
                # Max error: ~0.47, Mean error: ~0.15
                return np.array([0.469219, 0.500000, 0.093656, 0.000000])

            elif self.degree == 5:
                # Degree 5: balanced
                # Max error: ~0.29, Mean error: ~0.07
                return np.array([0.293262, 0.500000, 0.163899, -0.000000, -0.003271, 0.000000])

            elif self.degree == 7:
                # Degree 7: accurate (default)
                # Max error: ~0.21, Mean error: ~0.04
                return np.array([0.213837, 0.500000, 0.230484, -0.000000, -0.011246, 0.000000, 0.000233, 0.000000])

            elif self.degree == 9:
                # Degree 9: very accurate but slower
                # Max error: ~0.17, Mean error: ~0.03
                return np.array([0.168397, 0.500000, 0.295788, -0.000000, -0.025584, 0.000000, 0.001226, -0.000000, -0.000021, 0.000000])

            else:
                raise ValueError(f"Unsupported degree: {self.degree}. Use 3, 5, 7, or 9.")

        else:
            raise ValueError(f"Unknown approximation method: {self.approximation_method}")

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
            padded[:len(plaintext_vector)] = plaintext_vector
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
        return np.array(result[:self.n])

    def relu_polynomial_encrypted(self, x_ct):
        """
        Compute polynomial approximation of ReLU on encrypted data.

        Args:
            x_ct: Encrypted input vector

        Returns:
            Encrypted ReLU approximation
        """
        # Evaluate polynomial: sum(coeff[i] * x^i)
        # We'll use Horner's method for better numerical stability
        # But for now, direct evaluation

        result_ct = None

        # Pre-compute powers of x that we need
        x_powers = [None] * len(self.coefficients)
        x_powers[0] = None  # x^0 = 1 (constant, handled separately)
        if len(self.coefficients) > 1:
            x_powers[1] = x_ct  # x^1 = x

        # Compute higher powers iteratively
        for i in range(2, len(self.coefficients)):
            # Always compute powers, even if coefficient is zero (might need for next power)
            x_powers[i] = self.cc.EvalMult(x_powers[i-1], x_ct)

        # Evaluate polynomial
        for i, coeff in enumerate(self.coefficients):
            if abs(coeff) < 1e-10:  # Skip zero coefficients
                continue

            if i == 0:
                # Constant term
                term_pt = self.cc.MakeCKKSPackedPlaintext([coeff] * self.n)
                if result_ct is None:
                    result_ct = self.cc.Encrypt(self.keys.publicKey, term_pt)
                else:
                    result_ct = self.cc.EvalAdd(result_ct, term_pt)
            else:
                # Multiply power by coefficient
                coeff_pt = self.cc.MakeCKKSPackedPlaintext([coeff] * self.n)
                term_ct = self.cc.EvalMult(x_powers[i], coeff_pt)

                if result_ct is None:
                    result_ct = term_ct
                else:
                    result_ct = self.cc.EvalAdd(result_ct, term_ct)

        return result_ct

    def relu_encrypted(self, x):
        """
        Compute ReLU on encrypted data.

        Args:
            x: Plaintext input vector (will be encrypted)

        Returns:
            Decrypted ReLU result
        """
        # Encrypt input
        x_ct = self.encrypt_vector(x)

        # Compute polynomial approximation
        relu_ct = self.relu_polynomial_encrypted(x_ct)

        # Decrypt result
        return self.decrypt_vector(relu_ct)


def numpy_relu(x):
    """Reference ReLU implementation using NumPy."""
    return np.maximum(0, x)


def test_relu_basic():
    """Basic test of ReLU implementation."""
    print("=" * 80)
    print("  Basic ReLU Test")
    print("=" * 80)

    if not OPENFHE_AVAILABLE:
        print("\n❌ OpenFHE not available")
        return

    # Test inputs
    test_cases = [
        ("Positive values", np.array([1.0, 2.0, 3.0, 4.0, 5.0] + [0.0] * 123)),
        ("Negative values", np.array([-1.0, -2.0, -3.0, -4.0, -5.0] + [0.0] * 123)),
        ("Mixed values", np.array([3.0, -2.0, 1.0, -4.0, 2.0] + [0.0] * 123)),
        ("Small values", np.array([0.1, -0.1, 0.5, -0.5, 0.2] + [0.0] * 123)),
    ]

    print("\nInitializing ReLU...")
    relu = ReLUOpenFHE(n=128, mult_depth=10, degree=7)

    for name, test_input in test_cases:
        print(f"\n{name}:")
        print(f"  Input (first 5): {test_input[:5]}")

        # Compute ReLU
        result = relu.relu_encrypted(test_input)
        reference = numpy_relu(test_input)

        # Compare
        error = np.max(np.abs(result[:5] - reference[:5]))
        print(f"  OpenFHE:  {result[:5]}")
        print(f"  NumPy:    {reference[:5]}")
        print(f"  Max error: {error:.4f}")

        if error < 0.5:  # Approximate ReLU, more lenient
            print("  ✅ Acceptable approximation")
        else:
            print("  ⚠️  Large error (polynomial approximation)")


if __name__ == "__main__":
    test_relu_basic()
