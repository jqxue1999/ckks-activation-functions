"""
Attention Block using OpenFHE CKKS Encryption

Implements scaled dot-product attention:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

where:
- Q: Query matrix (seq_len x d_k)
- K: Key matrix (seq_len x d_k)
- V: Value matrix (seq_len x d_v)
- d_k: Key/Query dimension
- softmax: Applied row-wise

This implementation combines:
1. Matrix operations from openfhe-numpy
2. Row-wise softmax using our softmax_openfhe implementation
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

from softmax_openfhe import SoftmaxCKKSOpenFHE


class AttentionBlockOpenFHE:
    """
    Scaled dot-product attention using CKKS homomorphic encryption.

    Combines openfhe-numpy for matrix operations with custom softmax.
    """

    def __init__(
        self,
        seq_len: int = 8,
        d_k: int = 8,
        d_v: int = 8,
        mult_depth: int = 30,
        scale_mod_size: int = 59,
        softmax_K: int = 64,
        softmax_scale_factor: int = 8
    ):
        """
        Initialize attention block.

        Args:
            seq_len: Sequence length (must be power of 2)
            d_k: Key/Query dimension (must be power of 2)
            d_v: Value dimension (must be power of 2)
            mult_depth: Multiplicative depth for CKKS
            scale_mod_size: Scaling modulus size
            softmax_K: Number of terms for softmax approximation
            softmax_scale_factor: Scaling factor for softmax exponential
        """
        if not OPENFHE_AVAILABLE:
            raise ImportError("OpenFHE is required to use this class")

        self.seq_len = seq_len
        self.d_k = d_k
        self.d_v = d_v
        self.mult_depth = mult_depth
        self.scale_mod_size = scale_mod_size

        # Scaling factor for attention: 1 / sqrt(d_k)
        self.scale = 1.0 / np.sqrt(d_k)

        # Initialize CKKS context
        print(f"Initializing CKKS context (mult_depth={mult_depth})...")
        self.cc, self.keys = self._initialize_ckks()

        # Initialize softmax for each row (d_k dimensions)
        print(f"Initializing softmax for attention scores...")
        self.softmax = SoftmaxCKKSOpenFHE(
            n=d_k,
            K=softmax_K,
            scale_factor=softmax_scale_factor,
            mult_depth=mult_depth
        )

        print("✅ Attention block initialized successfully")

    def _initialize_ckks(self):
        """Initialize CKKS crypto context and keys."""
        params = CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(self.mult_depth)
        params.SetScalingModSize(self.scale_mod_size)
        params.SetFirstModSize(60)
        params.SetScalingTechnique(FIXEDAUTO)
        params.SetSecretKeyDist(UNIFORM_TERNARY)

        # Don't set batch size - let OpenFHE determine it automatically
        # This ensures we have enough slots for matrix operations

        cc = GenCryptoContext(params)
        cc.Enable(PKESchemeFeature.PKE)
        cc.Enable(PKESchemeFeature.LEVELEDSHE)
        cc.Enable(PKESchemeFeature.ADVANCEDSHE)

        keys = cc.KeyGen()
        cc.EvalMultKeyGen(keys.secretKey)
        cc.EvalSumKeyGen(keys.secretKey)

        print(f"  Ring dimension: {cc.GetRingDimension()}")
        print(f"  Slots available: {cc.GetRingDimension() // 2}")

        return cc, keys

    def encrypt_matrix(self, matrix, mode="tile"):
        """
        Encrypt a matrix using openfhe-numpy.

        Args:
            matrix: NumPy array to encrypt
            mode: Packing mode ("tile" or "zero")

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

        return ctm

    def attention_scores_encrypted(self, Q_ct, K_ct):
        """
        Compute attention scores: Q @ K^T / sqrt(d_k)

        Args:
            Q_ct: Encrypted query matrix
            K_ct: Encrypted key matrix

        Returns:
            openfhe_numpy array of attention scores (before softmax)
        """
        # Generate keys for transpose
        onp.gen_transpose_keys(self.keys.secretKey, K_ct)

        # Compute K^T
        KT_ct = onp.transpose(K_ct)

        # Generate keys for matrix multiplication
        onp.EvalSquareMatMultRotateKeyGen(self.keys.secretKey, Q_ct.ncols)

        # Compute Q @ K^T
        scores_ct = Q_ct @ KT_ct

        # Scale by 1/sqrt(d_k)
        # We'll scale after decryption for simplicity
        # (In production, implement homomorphic scaling)

        return scores_ct

    def softmax_rowwise_encrypted(self, scores_ct):
        """
        Apply softmax row-wise to attention scores.

        Args:
            scores_ct: Encrypted scores (openfhe_numpy array)

        Returns:
            NumPy array with softmax applied to each row
        """
        # Decrypt to get scores
        # (In a real production system, you'd implement fully homomorphic softmax)
        scores = scores_ct.decrypt(self.keys.secretKey, unpack_type="original")

        nrows, ncols = scores.shape

        # Scale by 1/sqrt(d_k)
        scores = scores * self.scale

        # Apply softmax to each row
        softmax_output = np.zeros_like(scores)
        for i in range(nrows):
            # Pad to softmax dimension if needed
            row = scores[i, :]
            if len(row) < self.softmax.n:
                row_padded = np.zeros(self.softmax.n)
                row_padded[:len(row)] = row
            else:
                row_padded = row[:self.softmax.n]

            # Apply softmax
            row_softmax = self.softmax.softmax_encrypted(row_padded)
            softmax_output[i, :] = row_softmax[:ncols]

        return softmax_output

    def attention_encrypted(self, Q, K, V):
        """
        Compute full attention: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

        Args:
            Q: Query matrix (seq_len x d_k)
            K: Key matrix (seq_len x d_k)
            V: Value matrix (seq_len x d_v)

        Returns:
            Attention output (seq_len x d_v)
        """
        print("\n" + "="*80)
        print("  Computing Attention Block")
        print("="*80)

        # Validate dimensions
        assert Q.shape == (self.seq_len, self.d_k), f"Q shape mismatch: {Q.shape} vs ({self.seq_len}, {self.d_k})"
        assert K.shape == (self.seq_len, self.d_k), f"K shape mismatch: {K.shape} vs ({self.seq_len}, {self.d_k})"
        assert V.shape == (self.seq_len, self.d_v), f"V shape mismatch: {V.shape} vs ({self.seq_len}, {self.d_v})"

        # Step 1: Encrypt Q, K, V
        print("\n[1/5] Encrypting Q, K, V matrices...")
        start = time.time()
        Q_ct = self.encrypt_matrix(Q, mode="tile")
        K_ct = self.encrypt_matrix(K, mode="tile")
        V_ct = self.encrypt_matrix(V, mode="tile")
        print(f"  Encryption time: {time.time() - start:.2f}s")

        # Step 2: Compute attention scores Q @ K^T
        print("\n[2/5] Computing attention scores (Q @ K^T)...")
        start = time.time()
        scores_ct = self.attention_scores_encrypted(Q_ct, K_ct)
        print(f"  Attention scores time: {time.time() - start:.2f}s")

        # Step 3: Apply scaling and softmax row-wise
        print("\n[3/5] Applying softmax to attention scores (with scaling)...")
        start = time.time()
        attention_weights = self.softmax_rowwise_encrypted(scores_ct)
        print(f"  Softmax time: {time.time() - start:.2f}s")
        print(f"  Attention weights shape: {attention_weights.shape}")

        # Step 4: Encrypt attention weights
        print("\n[4/5] Encrypting attention weights...")
        start = time.time()
        attention_weights_ct = self.encrypt_matrix(attention_weights, mode="tile")
        print(f"  Encryption time: {time.time() - start:.2f}s")

        # Step 5: Compute attention_weights @ V
        print("\n[5/5] Computing final output (attention_weights @ V)...")
        start = time.time()

        # Generate rotation keys for second matmul
        onp.EvalSquareMatMultRotateKeyGen(self.keys.secretKey, attention_weights_ct.ncols)

        output_ct = attention_weights_ct @ V_ct

        # Decrypt result
        output = output_ct.decrypt(self.keys.secretKey, unpack_type="original")
        print(f"  Final computation time: {time.time() - start:.2f}s")
        print(f"  Output shape: {output.shape}")

        return output, attention_weights


def numpy_attention(Q, K, V):
    """
    Reference attention implementation using NumPy.

    Args:
        Q: Query matrix (seq_len x d_k)
        K: Key matrix (seq_len x d_k)
        V: Value matrix (seq_len x d_v)

    Returns:
        Attention output (seq_len x d_v)
    """
    d_k = Q.shape[1]

    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_k)

    # Apply softmax row-wise
    attention_weights = np.zeros_like(scores)
    for i in range(scores.shape[0]):
        row = scores[i, :]
        exp_row = np.exp(row - np.max(row))  # Numerical stability
        attention_weights[i, :] = exp_row / np.sum(exp_row)

    # Compute output
    output = attention_weights @ V

    return output, attention_weights


if __name__ == "__main__":
    print("="*80)
    print("  Attention Block - Basic Test")
    print("="*80)

    if not OPENFHE_AVAILABLE:
        print("\n❌ OpenFHE not available")
        print("Install with: pip install openfhe openfhe_numpy")
        exit(1)

    # Small test case
    seq_len, d_k, d_v = 4, 4, 4

    print(f"\nTest configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Key/Query dimension: {d_k}")
    print(f"  Value dimension: {d_v}")

    # Initialize attention block
    start = time.time()
    attention = AttentionBlockOpenFHE(
        seq_len=seq_len,
        d_k=d_k,
        d_v=d_v,
        mult_depth=30,
        softmax_K=32,  # Smaller for testing
        softmax_scale_factor=4
    )
    init_time = time.time() - start
    print(f"\nInitialization time: {init_time:.2f}s")

    # Generate random Q, K, V
    np.random.seed(42)
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)

    print(f"\nInput matrices:")
    print(f"  Q shape: {Q.shape}")
    print(f"  K shape: {K.shape}")
    print(f"  V shape: {V.shape}")

    # Compute reference
    print("\nComputing reference (NumPy)...")
    ref_output, ref_weights = numpy_attention(Q, K, V)

    # Compute encrypted attention
    print("\nComputing encrypted attention...")
    start = time.time()
    enc_output, enc_weights = attention.attention_encrypted(Q, K, V)
    compute_time = time.time() - start

    print(f"\n{'='*80}")
    print("  Results")
    print("="*80)
    print(f"\nTotal computation time: {compute_time:.2f}s")

    # Compare outputs
    print(f"\nOutput comparison:")
    print(f"  Reference output (first row): {ref_output[0, :3]}")
    print(f"  Encrypted output (first row): {enc_output[0, :3]}")

    output_error = np.max(np.abs(enc_output - ref_output))
    print(f"\n  Max output error: {output_error:.6f}")

    # Compare attention weights
    print(f"\nAttention weights comparison:")
    print(f"  Reference weights (first row): {ref_weights[0, :]}")
    print(f"  Encrypted weights (first row): {enc_weights[0, :]}")

    weights_error = np.max(np.abs(enc_weights - ref_weights))
    print(f"\n  Max weights error: {weights_error:.6f}")

    # Check pass criteria
    if output_error < 0.1 and weights_error < 0.1:
        print("\n✅ TEST PASSED - Attention computation acceptable")
    else:
        print("\n⚠️  TEST WARNING - Large error (check softmax approximation)")
