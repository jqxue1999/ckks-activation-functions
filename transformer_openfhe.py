"""
Transformer Block using OpenFHE CKKS Encryption

Implements a complete transformer encoder block:
    TransformerBlock(x) = LayerNorm(x + MultiHeadAttention(x))
                          + LayerNorm(FFN(x))

where:
- MultiHeadAttention: Self-attention mechanism
- FFN: Feed-forward network (Linear -> ReLU -> Linear)
- LayerNorm: Layer normalization
- Residual connections: x + sublayer(x)

This combines:
1. Attention mechanism (attention_openfhe)
2. ReLU activation (relu_openfhe)
3. Matrix multiplication (matmul_openfhe)
4. Layer normalization (implemented here)
"""

import numpy as np
import time

try:
    from openfhe import *
    OPENFHE_AVAILABLE = True
except ImportError:
    OPENFHE_AVAILABLE = False
    print("Warning: OpenFHE not available")

from relu_openfhe import ReLUOpenFHE
from matmul_openfhe import MatMulOpenFHE

# Note: For fully encrypted transformer, use:
# - attention_fully_encrypted.AttentionFullyEncrypted
# - This file provides: LayerNormOpenFHE and FeedForwardOpenFHE


class LayerNormOpenFHE:
    """
    Layer Normalization for encrypted data using THOR approach.

    LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + epsilon) + beta

    Uses Goldschmidt's algorithm with adaptive Successive Over-Relaxation (aSOR)
    to compute inverse square root homomorphically, based on THOR paper.

    Reference: THOR paper Section 5.2 and Appendix C
    """

    def __init__(self, d_model: int, cc=None, keys=None, epsilon: float = 1e-5, mult_depth: int = 15):
        """
        Initialize layer normalization.

        Args:
            d_model: Model dimension
            cc: Crypto context (optional, will create if not provided)
            keys: Key pair (optional, will create if not provided)
            epsilon: Small constant for numerical stability (default 1e-5)
            mult_depth: Multiplicative depth for CKKS if creating new context
        """
        if not OPENFHE_AVAILABLE:
            raise ImportError("OpenFHE is required for LayerNormOpenFHE")

        self.d_model = d_model
        self.epsilon = epsilon

        # Learnable parameters (in practice, these would be trained)
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

        # Relaxation factors from THOR paper Appendix C for inverse square root
        # These are optimized for fast convergence with aSOR
        self.relaxation_factors = np.array([2.6374, 2.1722, 1.5135, 1.0907])

        # Setup crypto context
        if cc is None or keys is None:
            self.cc, self.keys = self._initialize_ckks(mult_depth)
            self.owns_context = True
        else:
            self.cc = cc
            self.keys = keys
            self.owns_context = False

        # Generate rotation keys for mean/variance computation and broadcasting
        # Only generate if we own the context (otherwise keys should already exist)
        if self.owns_context:
            rotation_indices = []
            # Need all rotations from 1 to d_model-1 (both positive and negative)
            for i in range(1, d_model):
                rotation_indices.append(i)  # Forward rotations for broadcasting
                rotation_indices.append(-i)  # Backward rotations
            self.cc.EvalRotateKeyGen(self.keys.secretKey, rotation_indices)

    def _initialize_ckks(self, mult_depth):
        """Initialize CKKS crypto context and keys."""
        params = CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(mult_depth)
        params.SetScalingModSize(59)
        params.SetFirstModSize(60)
        params.SetScalingTechnique(FIXEDAUTO)
        params.SetSecretKeyDist(UNIFORM_TERNARY)

        cc = GenCryptoContext(params)
        cc.Enable(PKESchemeFeature.PKE)
        cc.Enable(PKESchemeFeature.LEVELEDSHE)
        cc.Enable(PKESchemeFeature.ADVANCEDSHE)

        keys = cc.KeyGen()
        cc.EvalMultKeyGen(keys.secretKey)
        cc.EvalSumKeyGen(keys.secretKey)

        return cc, keys

    def compute_sum(self, ct):
        """
        Compute sum of all elements in a ciphertext.

        Uses OpenFHE's built-in EvalSum which puts the sum in the first slot.

        Args:
            ct: Ciphertext containing vector

        Returns:
            Ciphertext where first slot contains the sum
        """
        return self.cc.EvalSum(ct, self.d_model)

    def inverse_sqrt_goldschmidt(self, x):
        """
        Compute 1/sqrt(x) using Goldschmidt's algorithm with aSOR.

        Based on THOR paper Appendix C:
        Given a_0 = x ∈ (0, 1) and b_0 = 1, iteratively compute:
          a_{i+1} = k_i * a_i * (3 - k_i * a_i)^2 / 4
          b_{i+1} = sqrt(k_i) * b_i * (3 - k_i * a_i) / 2

        where k_i are adaptive relaxation factors.

        Args:
            x: Input value or ciphertext (should be scaled to (0, 1))

        Returns:
            Approximation of 1/sqrt(x) (ciphertext if input is ciphertext)
        """
        # Check if x is a ciphertext
        is_encrypted = hasattr(x, 'GetEncodingType')

        if is_encrypted:
            # Encrypted version - properly track both a and b
            # The Goldschmidt algorithm:
            #   a_{i+1} = k_i * a_i * (3 - k_i * a_i)^2 / 4
            #   b_{i+1} = sqrt(k_i) * b_i * (3 - k_i * a_i) / 2
            # Final result is b (not a)

            a = x
            # Start with b = 1.0 as a ciphertext
            b_pt = self.cc.MakeCKKSPackedPlaintext([1.0] * self.d_model)
            b = self.cc.Encrypt(self.keys.publicKey, b_pt)

            for k in self.relaxation_factors:
                # Compute ka = k * a (plaintext-ciphertext mult is free depth-wise)
                ka = self.cc.EvalMult(a, k)

                # Compute term = 3.0 - ka
                term = self.cc.EvalSub(3.0, ka)

                # Update a: a = ka * term^2 / 4
                term_sq = self.cc.EvalMult(term, term)  # depth +1
                a = self.cc.EvalMult(ka, term_sq)       # depth +1
                a = self.cc.EvalMult(a, 0.25)           # plaintext mult, depth +0

                # Update b: b = sqrt(k) * b * term / 2
                b = self.cc.EvalMult(b, np.sqrt(k))     # plaintext mult, depth +0
                b = self.cc.EvalMult(b, term)           # depth +1
                b = self.cc.EvalMult(b, 0.5)            # plaintext mult, depth +0

            # Return b (the inverse square root approximation)
            return b
        else:
            # Plaintext version (for testing/debugging)
            a = x
            b = 1.0

            for k in self.relaxation_factors:
                ka = k * a
                term = 3.0 - ka
                a = ka * (term ** 2) / 4.0
                b = np.sqrt(k) * b * term / 2.0

            return b

    def normalize(self, x, encrypted=False):
        """
        Apply layer normalization using THOR approach.

        Args:
            x: Input array (d_model,) or (batch, d_model), or ciphertext
            encrypted: Whether input is encrypted (auto-detected if not specified)

        Returns:
            Normalized array or ciphertext
        """
        # Auto-detect if input is encrypted
        if not encrypted:
            encrypted = hasattr(x, 'GetEncodingType')

        if encrypted:
            # Encrypted version using OpenFHE
            return self._normalize_encrypted(x)
        else:
            # Plaintext version for testing/comparison
            return self._normalize_plaintext(x)

    def _normalize_plaintext(self, x):
        """Plaintext version of normalization (for testing)."""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Compute mean
        mean = np.mean(x, axis=-1, keepdims=True)

        # Center the data
        x_centered = x - mean

        # Compute variance: var = mean(x_centered^2)
        var = np.mean(x_centered ** 2, axis=-1, keepdims=True)

        # Add epsilon for numerical stability
        var_eps = var + self.epsilon

        # Scale to (0, 1) for inverse square root approximation
        max_var = np.max(var_eps)
        if max_var > 0:
            scale = 1.0 / max_var
            var_scaled = var_eps * scale
        else:
            var_scaled = var_eps
            scale = 1.0

        # Compute 1/sqrt(var_eps) using Goldschmidt with aSOR
        inv_sqrt_var_scaled = self.inverse_sqrt_goldschmidt(var_scaled)

        # Unscale
        inv_sqrt_var = inv_sqrt_var_scaled * np.sqrt(scale)

        # Normalize
        x_norm = x_centered * inv_sqrt_var

        # Scale and shift
        output = x_norm * self.gamma + self.beta

        return output.squeeze()

    def _broadcast_first_slot(self, ct, n_slots):
        """
        Broadcast the value in the first slot to all other slots.

        Strategy: Extract first slot value, create plaintext vector with that value
        in all positions, but we need to do this homomorphically.

        Correct approach: Rotate forward to spread the value.
        If we have value V in slot 0 and want it in all slots:
        - Start with: [V, ?, ?, ?, ...]
        - Rotate right by 1, get [?, V, ?, ?, ...]
        - Add to original: [V, V, ?, ?, ...]
        - Continue...

        Args:
            ct: Ciphertext with value in first slot
            n_slots: Number of slots to fill

        Returns:
            Ciphertext with first slot value replicated to all slots
        """
        result = ct
        # Create a mask to zero out non-first slots
        mask = [1.0] + [0.0] * (n_slots - 1)
        mask_pt = self.cc.MakeCKKSPackedPlaintext(mask)
        result = self.cc.EvalMult(result, mask_pt)  # Now only first slot has value

        # Now rotate and add to fill all slots
        accumulated = result
        for i in range(1, n_slots):
            rotated = self.cc.EvalRotate(result, -i)  # Negative rotation to move slot[0] forward
            accumulated = self.cc.EvalAdd(accumulated, rotated)

        return accumulated

    def _normalize_encrypted(self, ct_x):
        """
        Fully encrypted version of normalization using OpenFHE.

        This implementation operates entirely on ciphertext without decrypting
        mean or variance statistics. Uses plaintext-ciphertext operations where
        the plaintext is derived from encrypted computations kept as scalars.

        Args:
            ct_x: Encrypted input (ciphertext)

        Returns:
            Encrypted normalized output (ciphertext)
        """
        # Step 1: Compute mean (fully encrypted, no decryption!)
        # EvalSum gives us sum in first slot
        sum_ct = self.compute_sum(ct_x)

        # Divide by n to get mean (as ciphertext with mean in first slot)
        # We'll use this with plaintext-ciphertext operations
        mean_scalar_ct = self.cc.EvalMult(sum_ct, 1.0 / self.d_model)

        # We need mean replicated across all d_model positions
        # Strategy: Use mask and rotations OR use repeated addition
        # For simplicity with d_model slots out of 65536: use plaintext broadcast
        # Extract first slot value by multiplying with mask [1, 0, 0, ...]
        mask_first = self.cc.MakeCKKSPackedPlaintext([1.0] + [0.0] * (self.d_model - 1))
        mean_first_only = self.cc.EvalMult(mean_scalar_ct, mask_first)

        # Broadcast by rotating and adding (use negative rotation to move slot[0] forward)
        mean_broadcast = mean_first_only
        for i in range(1, self.d_model):
            rotated = self.cc.EvalRotate(mean_first_only, -i)  # Negative rotation!
            mean_broadcast = self.cc.EvalAdd(mean_broadcast, rotated)

        # Step 2: Center the data: x_centered = x - mean
        x_centered = self.cc.EvalSub(ct_x, mean_broadcast)

        # Step 3: Compute variance (fully encrypted, no decryption!)
        x_centered_sq = self.cc.EvalMult(x_centered, x_centered)
        sum_sq_ct = self.compute_sum(x_centered_sq)
        var_scalar_ct = self.cc.EvalMult(sum_sq_ct, 1.0 / self.d_model)

        # Add epsilon to variance (broadcast like mean)
        epsilon_vec = self.cc.MakeCKKSPackedPlaintext([self.epsilon] + [0.0] * (self.d_model - 1))
        var_eps_first = self.cc.EvalAdd(var_scalar_ct, epsilon_vec)

        # Mask to get only first slot
        var_eps_first_only = self.cc.EvalMult(var_eps_first, mask_first)

        # Broadcast variance (use negative rotation to move slot[0] forward)
        var_eps_broadcast = var_eps_first_only
        for i in range(1, self.d_model):
            rotated = self.cc.EvalRotate(var_eps_first_only, -i)  # Negative rotation!
            var_eps_broadcast = self.cc.EvalAdd(var_eps_broadcast, rotated)

        # Step 4: Compute 1/sqrt(var) using encrypted Goldschmidt
        # Scale to (0, 1) if needed - use plaintext scaling
        # Note: Goldschmidt works best when input is in (0, 1) range
        scale_factor = 4.0  # Assume variance is reasonable, scale conservatively
        var_scaled_ct = self.cc.EvalMult(var_eps_broadcast, 1.0 / scale_factor)

        # Apply Goldschmidt algorithm on encrypted variance
        # This computes 1/sqrt(var_scaled)
        inv_sqrt_var_scaled_ct = self.inverse_sqrt_goldschmidt(var_scaled_ct)

        # Unscale: 1/sqrt(var) = 1/sqrt(scale * var_scaled) = 1/sqrt(scale) * 1/sqrt(var_scaled)
        inv_sqrt_var_ct = self.cc.EvalMult(inv_sqrt_var_scaled_ct, 1.0 / np.sqrt(scale_factor))

        # Step 5: Normalize: x_norm = x_centered * inv_sqrt_var
        x_norm_ct = self.cc.EvalMult(x_centered, inv_sqrt_var_ct)

        # Step 6: Scale and shift: output = x_norm * gamma + beta
        gamma_pt = self.cc.MakeCKKSPackedPlaintext(self.gamma.tolist())
        x_scaled_ct = self.cc.EvalMult(x_norm_ct, gamma_pt)

        beta_pt = self.cc.MakeCKKSPackedPlaintext(self.beta.tolist())
        output_ct = self.cc.EvalAdd(x_scaled_ct, beta_pt)

        return output_ct

    def encrypt(self, x):
        """
        Encrypt a plaintext vector.

        Args:
            x: NumPy array (d_model,)

        Returns:
            Ciphertext
        """
        if len(x) != self.d_model:
            raise ValueError(f"Input size {len(x)} != d_model {self.d_model}")

        pt = self.cc.MakeCKKSPackedPlaintext(x.tolist())
        ct = self.cc.Encrypt(self.keys.publicKey, pt)
        return ct

    def decrypt(self, ct):
        """
        Decrypt a ciphertext.

        Args:
            ct: Ciphertext

        Returns:
            NumPy array (d_model,)
        """
        pt = self.cc.Decrypt(self.keys.secretKey, ct)
        pt.SetLength(self.d_model)
        result = np.array(pt.GetRealPackedValue()[:self.d_model])
        return result


class FeedForwardOpenFHE:
    """
    Feed-forward network for transformer.

    FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

    Two linear transformations with ReLU in between.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        mult_depth: int = 15,
        relu_degree: int = 7
    ):
        """
        Initialize feed-forward network.

        Args:
            d_model: Model dimension (input/output size)
            d_ff: Hidden dimension (typically 4 * d_model)
            mult_depth: CKKS depth
            relu_degree: ReLU polynomial degree
        """
        self.d_model = d_model
        self.d_ff = d_ff

        # Initialize weights (in practice, these would be trained)
        np.random.seed(42)
        scale = 0.1
        self.W1 = np.random.randn(d_model, d_ff) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale
        self.b2 = np.zeros(d_model)

        # Initialize matrix multiplication module
        self.matmul = MatMulOpenFHE(mult_depth=mult_depth, verbose=False)

        # Initialize ReLU
        # Pad to power of 2 for CKKS
        relu_n = 2 ** int(np.ceil(np.log2(d_ff)))
        self.relu = ReLUOpenFHE(n=relu_n, mult_depth=mult_depth, degree=relu_degree)

    def forward(self, x):
        """
        Forward pass through feed-forward network.

        Args:
            x: Input (d_model,) or (batch, d_model)

        Returns:
            Output (same shape as input)
        """
        original_shape = x.shape
        if x.ndim == 1:
            x = x.reshape(1, -1)

        batch_size = x.shape[0]
        outputs = []

        for i in range(batch_size):
            # First linear layer: x @ W1 + b1
            hidden = x[i] @ self.W1 + self.b1

            # ReLU activation
            # Pad to relu dimension
            if len(hidden) < self.relu.n:
                hidden_padded = np.zeros(self.relu.n)
                hidden_padded[:len(hidden)] = hidden
            else:
                hidden_padded = hidden[:self.relu.n]

            # Apply ReLU on encrypted data
            activated = self.relu.relu_encrypted(hidden_padded)
            activated = activated[:self.d_ff]

            # Second linear layer: activated @ W2 + b2
            output = activated @ self.W2 + self.b2
            outputs.append(output)

        result = np.array(outputs)
        if original_shape == (self.d_model,):
            result = result.squeeze()

        return result


# Note: TransformerBlockOpenFHE has been removed because it uses the old
# attention implementation with decryption points.
# For fully encrypted transformer, use test_full_transformer_encrypted.py which combines:
# - attention_fully_encrypted.AttentionFullyEncrypted (fully encrypted)
# - transformer_openfhe.LayerNormOpenFHE (fully encrypted with Goldschmidt)


if __name__ == "__main__":
    print("="*80)
    print("  LayerNorm and FeedForward - Basic Test")
    print("="*80)

    if not OPENFHE_AVAILABLE:
        print("\n❌ OpenFHE not available")
        print("Install with: pip install openfhe openfhe_numpy")
        exit(1)

    # Test LayerNorm
    d_model = 8
    print(f"\nTesting LayerNorm with d_model={d_model}...")

    layernorm = LayerNormOpenFHE(d_model=d_model, mult_depth=25)

    # Generate test input
    np.random.seed(42)
    x = np.random.randn(d_model) * 0.5

    print(f"Input: {x[:3]}")

    # Test normalization
    output = layernorm.normalize(x)

    print(f"Output: {output[:3]}")
    print(f"Mean: {np.mean(output):.6f}")
    print(f"Variance: {np.var(output):.6f}")

    if abs(np.mean(output)) < 1e-5 and abs(np.var(output) - 1.0) < 0.1:
        print("\n✅ LayerNorm TEST PASSED")
    else:
        print("\n⚠️  LayerNorm TEST WARNING - Check normalization")

    print("\nFor complete fully encrypted transformer, see:")
    print("  python3 test_full_transformer_encrypted.py")
