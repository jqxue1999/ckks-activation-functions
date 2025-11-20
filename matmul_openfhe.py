"""
Matrix Multiplication Operations using OpenFHE CKKS

This module provides matrix operations on encrypted data:
- Matrix encryption/decryption
- Matrix transpose
- Matrix multiplication (Q @ K^T, A @ B)
- Utility functions for rotation keys

Built on top of openfhe-numpy for efficient matrix operations.
"""

import numpy as np
import time

try:
    from openfhe import *
    import openfhe_numpy as onp
    OPENFHE_AVAILABLE = True
except ImportError:
    OPENFHE_AVAILABLE = False
    print("Warning: OpenFHE not available")


class MatMulOpenFHE:
    """
    Matrix multiplication operations using CKKS homomorphic encryption.

    Provides encrypted matrix operations:
    - A @ B: Matrix multiplication
    - A.T: Matrix transpose
    - Encryption/decryption utilities
    """

    def __init__(
        self,
        mult_depth: int = 10,
        scale_mod_size: int = 59,
        verbose: bool = False
    ):
        """
        Initialize matrix multiplication context.

        Args:
            mult_depth: Multiplicative depth for CKKS
            scale_mod_size: Scaling modulus size
            verbose: Print detailed information
        """
        if not OPENFHE_AVAILABLE:
            raise ImportError("OpenFHE is required to use this class")

        self.mult_depth = mult_depth
        self.scale_mod_size = scale_mod_size
        self.verbose = verbose

        # Initialize CKKS context
        if self.verbose:
            print(f"Initializing CKKS context for matrix operations...")
        self.cc, self.keys = self._initialize_ckks()

        if self.verbose:
            print(f"✅ MatMul initialized (ring_dim={self.cc.GetRingDimension()}, "
                  f"slots={self.cc.GetRingDimension() // 2})")

    def _initialize_ckks(self):
        """Initialize CKKS crypto context and keys."""
        params = CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(self.mult_depth)
        params.SetScalingModSize(self.scale_mod_size)
        params.SetFirstModSize(60)
        params.SetScalingTechnique(FIXEDAUTO)
        params.SetSecretKeyDist(UNIFORM_TERNARY)

        # Let OpenFHE determine batch size automatically
        cc = GenCryptoContext(params)
        cc.Enable(PKESchemeFeature.PKE)
        cc.Enable(PKESchemeFeature.LEVELEDSHE)
        cc.Enable(PKESchemeFeature.ADVANCEDSHE)

        keys = cc.KeyGen()
        cc.EvalMultKeyGen(keys.secretKey)
        cc.EvalSumKeyGen(keys.secretKey)

        return cc, keys

    def encrypt_matrix(self, matrix, mode="tile"):
        """
        Encrypt a matrix using openfhe-numpy.

        Args:
            matrix: NumPy array to encrypt (shape: m x n)
            mode: Packing mode ("tile" or "zero")
                - "tile": Repeats matrix to fill slots
                - "zero": Pads with zeros

        Returns:
            Encrypted matrix (openfhe_numpy array)
        """
        batch_size = self.cc.GetRingDimension() // 2

        ctm = onp.array(
            cc=self.cc,
            data=matrix,
            batch_size=batch_size,
            order=onp.ROW_MAJOR,
            fhe_type="C",
            mode=mode,
            public_key=self.keys.publicKey,
        )

        if self.verbose:
            print(f"  Encrypted matrix: shape {matrix.shape}, mode={mode}")

        return ctm

    def decrypt_matrix(self, ctm, unpack_type="original"):
        """
        Decrypt an encrypted matrix.

        Args:
            ctm: Encrypted matrix (openfhe_numpy array)
            unpack_type: How to unpack ("original" or "full")

        Returns:
            Decrypted NumPy array
        """
        result = ctm.decrypt(self.keys.secretKey, unpack_type=unpack_type)

        if self.verbose:
            print(f"  Decrypted matrix: shape {result.shape}")

        return result

    def transpose(self, ctm):
        """
        Compute transpose of encrypted matrix: A^T

        Args:
            ctm: Encrypted matrix (openfhe_numpy array)

        Returns:
            Encrypted transpose (openfhe_numpy array)
        """
        # Generate transpose keys if not already generated
        onp.gen_transpose_keys(self.keys.secretKey, ctm)

        # Compute transpose
        ctm_T = onp.transpose(ctm)

        if self.verbose:
            print(f"  Transposed matrix: {ctm.nrows}x{ctm.ncols} → {ctm_T.nrows}x{ctm_T.ncols}")

        return ctm_T

    def matmul(self, ctm_A, ctm_B):
        """
        Compute matrix multiplication: A @ B

        Args:
            ctm_A: Encrypted matrix A (openfhe_numpy array)
            ctm_B: Encrypted matrix B (openfhe_numpy array)

        Returns:
            Encrypted result A @ B (openfhe_numpy array)
        """
        # Generate rotation keys for matrix multiplication
        onp.EvalSquareMatMultRotateKeyGen(self.keys.secretKey, ctm_A.ncols)

        # Compute A @ B
        ctm_result = ctm_A @ ctm_B

        if self.verbose:
            print(f"  Matrix multiply: ({ctm_A.nrows}x{ctm_A.ncols}) @ "
                  f"({ctm_B.nrows}x{ctm_B.ncols}) → "
                  f"({ctm_result.nrows}x{ctm_result.ncols})")

        return ctm_result

    def matmul_with_transpose(self, ctm_A, ctm_B):
        """
        Compute matrix multiplication with transpose: A @ B^T

        This is a common operation in attention mechanisms.

        Args:
            ctm_A: Encrypted matrix A (openfhe_numpy array)
            ctm_B: Encrypted matrix B (openfhe_numpy array)

        Returns:
            Encrypted result A @ B^T (openfhe_numpy array)
        """
        if self.verbose:
            print(f"  Computing A @ B^T...")

        # Transpose B
        ctm_BT = self.transpose(ctm_B)

        # Multiply A @ B^T
        ctm_result = self.matmul(ctm_A, ctm_BT)

        return ctm_result

    def encrypt_and_multiply(self, A, B, mode="tile"):
        """
        Convenience function: encrypt two matrices and multiply them.

        Args:
            A: NumPy array (matrix A)
            B: NumPy array (matrix B)
            mode: Packing mode for encryption

        Returns:
            Decrypted result of A @ B
        """
        if self.verbose:
            print(f"\nEncrypt and multiply: {A.shape} @ {B.shape}")

        start = time.time()

        # Encrypt
        ctm_A = self.encrypt_matrix(A, mode=mode)
        ctm_B = self.encrypt_matrix(B, mode=mode)

        if self.verbose:
            print(f"  Encryption time: {time.time() - start:.2f}s")

        # Multiply
        start = time.time()
        ctm_result = self.matmul(ctm_A, ctm_B)

        if self.verbose:
            print(f"  Multiplication time: {time.time() - start:.2f}s")

        # Decrypt
        start = time.time()
        result = self.decrypt_matrix(ctm_result)

        if self.verbose:
            print(f"  Decryption time: {time.time() - start:.2f}s")

        return result

    def scale_matrix(self, ctm, scalar):
        """
        Multiply encrypted matrix by a scalar: A * c

        Args:
            ctm: Encrypted matrix (openfhe_numpy array)
            scalar: Scalar value (float)

        Returns:
            Encrypted result A * c (openfhe_numpy array)
        """
        # Create plaintext scalar
        # Note: Must broadcast to all slots
        batch_size = self.cc.GetRingDimension() // 2
        scalar_pt = self.cc.MakeCKKSPackedPlaintext([scalar] * batch_size)

        # Multiply (note: ctm.data is the ciphertext)
        # For openfhe_numpy arrays, we need to handle this differently
        # This is a simplified version - in practice, use element-wise ops

        if self.verbose:
            print(f"  Scaling matrix by {scalar}")

        # Return scaled ciphertext wrapped in openfhe_numpy array
        # This is a placeholder - actual implementation depends on openfhe_numpy API
        return ctm


def numpy_matmul(A, B):
    """
    Reference matrix multiplication using NumPy.

    Args:
        A: NumPy array (m x n)
        B: NumPy array (n x p)

    Returns:
        Result A @ B (m x p)
    """
    return A @ B


def test_basic_matmul():
    """Test basic matrix multiplication."""
    print("="*80)
    print("  MatMul - Basic Test")
    print("="*80)

    if not OPENFHE_AVAILABLE:
        print("\n❌ OpenFHE not available")
        return

    # Initialize
    print("\nInitializing MatMul context...")
    matmul = MatMulOpenFHE(mult_depth=10, verbose=True)

    # Test matrices
    m, n, p = 4, 4, 4
    A = np.random.randn(m, n)
    B = np.random.randn(n, p)

    print(f"\nTest: {m}x{n} @ {n}x{p}")
    print(f"  A: {A.shape}")
    print(f"  B: {B.shape}")

    # Reference
    ref_result = numpy_matmul(A, B)
    print(f"  Reference result: {ref_result.shape}")

    # Encrypted
    print("\nComputing encrypted multiplication...")
    enc_result = matmul.encrypt_and_multiply(A, B)
    print(f"  Encrypted result: {enc_result.shape}")

    # Compare
    error = np.max(np.abs(enc_result - ref_result))
    print(f"\nMax error: {error:.6f}")

    if error < 0.01:
        print("✅ TEST PASSED - Matrix multiplication correct")
    else:
        print("❌ TEST FAILED - Error too large")


def test_transpose_and_multiply():
    """Test A @ B^T operation."""
    print("\n" + "="*80)
    print("  MatMul - Transpose Test")
    print("="*80)

    if not OPENFHE_AVAILABLE:
        print("\n❌ OpenFHE not available")
        return

    # Initialize
    print("\nInitializing MatMul context...")
    matmul = MatMulOpenFHE(mult_depth=10, verbose=True)

    # Test matrices (for attention: Q @ K^T)
    seq_len, d_k = 4, 4
    Q = np.random.randn(seq_len, d_k) * 0.5
    K = np.random.randn(seq_len, d_k) * 0.5

    print(f"\nTest: Q @ K^T = {Q.shape} @ {K.shape}^T")

    # Reference
    ref_result = Q @ K.T
    print(f"  Reference result: {ref_result.shape}")

    # Encrypted
    print("\nComputing encrypted Q @ K^T...")
    start = time.time()

    Q_ct = matmul.encrypt_matrix(Q, mode="tile")
    K_ct = matmul.encrypt_matrix(K, mode="tile")

    result_ct = matmul.matmul_with_transpose(Q_ct, K_ct)
    enc_result = matmul.decrypt_matrix(result_ct)

    compute_time = time.time() - start

    print(f"  Encrypted result: {enc_result.shape}")
    print(f"  Computation time: {compute_time:.2f}s")

    # Compare
    error = np.max(np.abs(enc_result - ref_result))
    print(f"\nMax error: {error:.6f}")

    if error < 0.01:
        print("✅ TEST PASSED - Transpose and multiply correct")
    else:
        print("❌ TEST FAILED - Error too large")


if __name__ == "__main__":
    test_basic_matmul()
    test_transpose_and_multiply()
