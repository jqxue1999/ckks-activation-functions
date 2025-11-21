"""
Fully Encrypted Attention Block - Simplified Version

This version operates on single vectors (seq_len=1) to demonstrate
fully encrypted attention without matrix row extraction complexity.

For full seq_len > 1, this shows the building blocks needed.
"""

import numpy as np
import time
from softmax_openfhe import SoftmaxCKKSOpenFHE

try:
    from openfhe import *
    OPENFHE_AVAILABLE = True
except ImportError:
    OPENFHE_AVAILABLE = False


class AttentionFullyEncrypted:
    """
    Simplified attention for single query-key-value vectors.

    Demonstrates fully encrypted attention scores and softmax
    without intermediate decryption.
    """

    def __init__(self, d_k=8, mult_depth=35, softmax_K=64, softmax_scale_factor=8):
        """
        Initialize fully encrypted attention.

        Args:
            d_k: Key/Query/Value dimension
            mult_depth: CKKS multiplicative depth
            softmax_K: Softmax approximation terms
            softmax_scale_factor: Softmax scaling factor
        """
        if not OPENFHE_AVAILABLE:
            raise ImportError("OpenFHE required")

        self.d_k = d_k
        self.scale = 1.0 / np.sqrt(d_k)

        # Initialize softmax
        print(f"Initializing softmax (d_k={d_k})...")
        self.softmax = SoftmaxCKKSOpenFHE(
            n=d_k,
            K=softmax_K,
            scale_factor=softmax_scale_factor,
            mult_depth=mult_depth
        )

        # Use softmax's crypto context
        self.cc = self.softmax.cc
        self.keys = self.softmax.keys

        # Generate rotation keys for broadcasting
        print(f"Generating rotation keys...")
        rotation_indices = []
        for i in range(1, d_k):
            rotation_indices.append(i)
            rotation_indices.append(-i)
        self.cc.EvalRotateKeyGen(self.keys.secretKey, rotation_indices)

        print("✅ Attention initialized")

    def encrypt_vector(self, vec):
        """Encrypt a vector."""
        return self.softmax.encrypt_vector(vec)

    def decrypt_vector(self, ct):
        """Decrypt a vector."""
        return self.softmax.decrypt_vector(ct)

    def attention_single_encrypted(self, q, k, v, return_ciphertext=False):
        """
        Compute attention for single query-key-value vectors.

        Attention(q, k, v) = softmax(q · k / sqrt(d_k)) * v

        This is fully encrypted - NO decryption during computation!

        Args:
            q: Query vector (d_k,) - plaintext
            k: Key vector (d_k,) - plaintext
            v: Value vector (d_k,) - plaintext
            return_ciphertext: If True, return encrypted output

        Returns:
            Attention output - encrypted or decrypted based on flag
        """
        print("\n" + "="*80)
        print("  Fully Encrypted Attention (Single Vector)")
        print("="*80)
        print()

        # Step 1: Encrypt inputs
        print("[1/5] Encrypting Q, K, V...")
        start = time.time()
        q_ct = self.encrypt_vector(q)
        k_ct = self.encrypt_vector(k)
        v_ct = self.encrypt_vector(v)
        print(f"  Time: {time.time() - start:.2f}s")
        print()

        # Step 2: Compute attention score: q · k (dot product)
        print("[2/5] Computing attention score (q · k)...")
        start = time.time()

        # Multiply element-wise: q_ct * k_ct
        qk_ct = self.cc.EvalMult(q_ct, k_ct)

        # Sum all elements to get dot product
        # Use EvalSum to put sum in first slot
        score_ct = self.cc.EvalSum(qk_ct, self.d_k)

        # Broadcast sum to all slots (we need same score for all elements)
        mask = self.cc.MakeCKKSPackedPlaintext([1.0] + [0.0] * (self.d_k - 1))
        score_first = self.cc.EvalMult(score_ct, mask)

        # Broadcast via rotations
        score_broadcast = score_first
        for i in range(1, self.d_k):
            rotated = self.cc.EvalRotate(score_first, -i)
            score_broadcast = self.cc.EvalAdd(score_broadcast, rotated)

        # Scale by 1/sqrt(d_k)
        score_scaled = self.cc.EvalMult(score_broadcast, self.scale)

        print(f"  Time: {time.time() - start:.2f}s")
        print(f"  ✓ Score computed on CIPHERTEXT")
        print()

        # Step 3: Apply softmax to score (fully on ciphertext!)
        print("[3/5] Applying softmax on ciphertext...")
        print("  Using softmax_encrypted_from_ciphertext (NO decryption!)")
        start = time.time()

        # Apply softmax directly on ciphertext - NO DECRYPTION!
        weight_ct = self.softmax.softmax_encrypted_from_ciphertext(score_scaled, return_ciphertext=True)

        print(f"  Time: {time.time() - start:.2f}s")
        print(f"  ✓ Softmax applied (fully encrypted internally)")
        print()

        # Step 4: Apply attention weight to value: weight * v
        print("[4/5] Computing output (weight * v)...")
        start = time.time()

        # Element-wise multiply
        output_ct = self.cc.EvalMult(weight_ct, v_ct)

        print(f"  Time: {time.time() - start:.2f}s")
        print(f"  ✓ Output is CIPHERTEXT")
        print()

        if return_ciphertext:
            print("[5/5] Returning CIPHERTEXT")
            print(f"  ✓ No final decryption")
            return output_ct
        else:
            print("[5/5] Decrypting output for verification...")
            start = time.time()
            output = self.decrypt_vector(output_ct)
            print(f"  Time: {time.time() - start:.2f}s")
            return output


def numpy_attention_single(q, k, v):
    """Reference attention for single vectors."""
    d_k = len(q)
    scale = 1.0 / np.sqrt(d_k)

    # Compute score
    score = np.dot(q, k) * scale

    # Apply softmax (for single score, this is just exp normalization)
    score_vec = np.full(d_k, score)  # Broadcast to all positions
    exp_scores = np.exp(score_vec - np.max(score_vec))
    weights = exp_scores / np.sum(exp_scores)

    # Apply to value
    output = weights * v

    return output, weights


if __name__ == "__main__":
    print("="*80)
    print("  Fully Encrypted Attention - Single Vector Test")
    print("="*80)
    print()

    # Configuration
    d_k = 8

    print(f"Configuration:")
    print(f"  Dimension (d_k): {d_k}")
    print()

    # Initialize
    print("Initializing fully encrypted attention...")
    start = time.time()
    attention = AttentionFullyEncrypted(d_k=d_k, mult_depth=35)
    init_time = time.time() - start
    print(f"Initialization time: {init_time:.2f}s")
    print()

    # Generate test data
    np.random.seed(42)
    q = np.random.randn(d_k) * 0.5
    k = np.random.randn(d_k) * 0.5
    v = np.random.randn(d_k) * 0.5

    print(f"Input:")
    print(f"  Q: {q[:4]}...")
    print(f"  K: {k[:4]}...")
    print(f"  V: {v[:4]}...")
    print()

    # Compute reference
    print("Computing reference (plaintext)...")
    ref_output, ref_weights = numpy_attention_single(q, k, v)
    print(f"  Reference output: {ref_output[:4]}...")
    print(f"  Reference weights: {ref_weights[:4]}...")
    print()

    # Compute encrypted
    print("Computing fully encrypted attention...")
    start = time.time()
    enc_output = attention.attention_single_encrypted(q, k, v, return_ciphertext=False)
    compute_time = time.time() - start

    print()
    print("="*80)
    print("  RESULTS")
    print("="*80)
    print()

    print(f"Encrypted output: {enc_output[:4]}...")
    print(f"Reference output: {ref_output[:4]}...")
    print()

    error = np.max(np.abs(enc_output - ref_output))
    print(f"Max error: {error:.6f}")
    print(f"Total time: {compute_time:.2f}s")
    print()

    if error < 0.1:
        print("✅ FULLY ENCRYPTED ATTENTION TEST PASSED!")
        print()
        print("Note: This demonstrates attention for single vectors.")
        print("For full matrices (seq_len > 1), the same principle applies")
        print("but requires processing each row separately.")
    else:
        print("❌ TEST FAILED - Error too large")
