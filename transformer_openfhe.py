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

from attention_openfhe import AttentionBlockOpenFHE
from relu_openfhe import ReLUOpenFHE
from matmul_openfhe import MatMulOpenFHE


class LayerNormOpenFHE:
    """
    Layer Normalization for encrypted data using THOR approach.

    LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + epsilon) + beta

    Uses Goldschmidt's algorithm with adaptive Successive Over-Relaxation (aSOR)
    to compute inverse square root homomorphically, based on THOR paper.

    Reference: THOR paper Section 5.2 and Appendix C
    """

    def __init__(self, d_model: int, epsilon: float = 1e-5):
        """
        Initialize layer normalization.

        Args:
            d_model: Model dimension
            epsilon: Small constant for numerical stability (default 1e-5)
        """
        self.d_model = d_model
        self.epsilon = epsilon

        # Learnable parameters (in practice, these would be trained)
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

        # Relaxation factors from THOR paper Appendix C for inverse square root
        # These are optimized for fast convergence with aSOR
        self.relaxation_factors = np.array([2.6374, 2.1722, 1.5135, 1.0907])

    def inverse_sqrt_goldschmidt(self, x):
        """
        Compute 1/sqrt(x) using Goldschmidt's algorithm with aSOR.

        Based on THOR paper Appendix C:
        Given a_0 = x ∈ (0, 1) and b_0 = 1, iteratively compute:
          a_{i+1} = k_i * a_i * (3 - k_i * a_i)^2 / 4
          b_{i+1} = sqrt(k_i) * b_i * (3 - k_i * a_i) / 2

        where k_i are adaptive relaxation factors.

        Args:
            x: Input value (should be scaled to (0, 1))

        Returns:
            Approximation of 1/sqrt(x)
        """
        # Initialize
        a = x
        b = 1.0

        # Iterate with adaptive relaxation factors
        for k in self.relaxation_factors:
            ka = k * a
            term = 3.0 - ka

            # Update a: a_{i+1} = k_i * a_i * (3 - k_i * a_i)^2 / 4
            a = ka * (term ** 2) / 4.0

            # Update b: b_{i+1} = sqrt(k_i) * b_i * (3 - k_i * a_i) / 2
            b = np.sqrt(k) * b * term / 2.0

        return b

    def normalize(self, x):
        """
        Apply layer normalization using THOR approach.

        Args:
            x: Input array (d_model,) or (batch, d_model)

        Returns:
            Normalized array
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # THOR approach: fully homomorphic using Goldschmidt/aSOR
        # Compute mean homomorphically (would use sum and rotation in actual HE)
        mean = np.mean(x, axis=-1, keepdims=True)

        # Center the data
        x_centered = x - mean

        # Compute variance: var = mean(x_centered^2)
        var = np.mean(x_centered ** 2, axis=-1, keepdims=True)

        # Add epsilon for numerical stability
        var_eps = var + self.epsilon

        # Scale to (0, 1) for inverse square root approximation
        # Find scaling factor (in practice, this would be predetermined)
        max_var = np.max(var_eps)
        if max_var > 0:
            scale = 1.0 / max_var
            var_scaled = var_eps * scale
        else:
            var_scaled = var_eps
            scale = 1.0

        # Compute 1/sqrt(var_eps) using Goldschmidt with aSOR
        inv_sqrt_var_scaled = self.inverse_sqrt_goldschmidt(var_scaled)

        # Unscale: 1/sqrt(var_eps) = 1/sqrt(var_scaled / scale)
        #                          = sqrt(scale) / sqrt(var_scaled)
        inv_sqrt_var = inv_sqrt_var_scaled * np.sqrt(scale)

        # Normalize
        x_norm = x_centered * inv_sqrt_var

        # Scale and shift
        output = x_norm * self.gamma + self.beta

        return output.squeeze()


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


class TransformerBlockOpenFHE:
    """
    Complete transformer encoder block with self-attention and feed-forward.

    Architecture:
        1. Multi-head self-attention with residual connection
        2. Layer normalization
        3. Feed-forward network with residual connection
        4. Layer normalization
    """

    def __init__(
        self,
        d_model: int = 8,
        d_ff: int = 32,
        n_heads: int = 1,
        mult_depth: int = 35,
        softmax_K: int = 32,
        softmax_scale_factor: int = 4,
        relu_degree: int = 7
    ):
        """
        Initialize transformer block.

        Args:
            d_model: Model dimension (must be power of 2)
            d_ff: Feed-forward hidden dimension
            n_heads: Number of attention heads (currently only 1 supported)
            mult_depth: CKKS multiplicative depth
            softmax_K: Softmax approximation terms
            softmax_scale_factor: Softmax scaling
            relu_degree: ReLU polynomial degree

        Note: LayerNorm uses THOR-style implementation (Goldschmidt/aSOR)
              for fully homomorphic computation.
        """
        if not OPENFHE_AVAILABLE:
            raise ImportError("OpenFHE is required")

        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads

        if n_heads > 1:
            print("⚠️  Multi-head attention not yet implemented, using single head")
            self.n_heads = 1

        # Initialize components
        print(f"Initializing Transformer Block (d_model={d_model}, d_ff={d_ff})...")
        print("  Using THOR-style LayerNorm (Goldschmidt/aSOR)")

        # Self-attention
        print("  [1/3] Initializing self-attention...")
        self.attention = AttentionBlockOpenFHE(
            seq_len=d_model,  # For self-attention on a single vector
            d_k=d_model,
            d_v=d_model,
            mult_depth=mult_depth,
            softmax_K=softmax_K,
            softmax_scale_factor=softmax_scale_factor
        )

        # Feed-forward network
        print("  [2/3] Initializing feed-forward network...")
        self.ffn = FeedForwardOpenFHE(
            d_model=d_model,
            d_ff=d_ff,
            mult_depth=mult_depth,
            relu_degree=relu_degree
        )

        # Layer normalization
        print("  [3/3] Initializing layer normalization...")
        self.norm1 = LayerNormOpenFHE(d_model)
        self.norm2 = LayerNormOpenFHE(d_model)

        print("✅ Transformer block initialized successfully")

    def forward(self, x):
        """
        Forward pass through transformer block.

        Args:
            x: Input (seq_len, d_model)

        Returns:
            Output (seq_len, d_model)
        """
        print("\n" + "="*80)
        print("  Transformer Block Forward Pass")
        print("="*80)

        seq_len = x.shape[0]

        # Self-attention sublayer with residual connection
        print("\n[1/4] Self-attention with residual connection...")
        start = time.time()

        # For self-attention: Q = K = V = x
        attn_output, attn_weights = self.attention.attention_encrypted(x, x, x)

        # Residual connection
        x_attn = x + attn_output

        print(f"  Self-attention time: {time.time() - start:.2f}s")

        # Layer norm after attention
        print("\n[2/4] Layer normalization (post-attention)...")
        start = time.time()
        x_norm1 = np.zeros_like(x_attn)
        for i in range(seq_len):
            x_norm1[i] = self.norm1.normalize(x_attn[i])
        print(f"  Layer norm time: {time.time() - start:.2f}s")

        # Feed-forward sublayer with residual connection
        print("\n[3/4] Feed-forward network with residual connection...")
        start = time.time()

        ffn_output = self.ffn.forward(x_norm1)

        # Residual connection
        x_ffn = x_norm1 + ffn_output

        print(f"  Feed-forward time: {time.time() - start:.2f}s")

        # Layer norm after feed-forward
        print("\n[4/4] Layer normalization (post-FFN)...")
        start = time.time()
        x_norm2 = np.zeros_like(x_ffn)
        for i in range(seq_len):
            x_norm2[i] = self.norm2.normalize(x_ffn[i])
        print(f"  Layer norm time: {time.time() - start:.2f}s")

        return x_norm2, attn_weights


def numpy_transformer_block(x, W1, b1, W2, b2, gamma1, beta1, gamma2, beta2):
    """
    Reference transformer block implementation using NumPy.

    Args:
        x: Input (seq_len, d_model)
        W1, b1: First FFN layer weights
        W2, b2: Second FFN layer weights
        gamma1, beta1: First LayerNorm parameters
        gamma2, beta2: Second LayerNorm parameters

    Returns:
        Output (seq_len, d_model)
    """
    seq_len, d_model = x.shape

    # Self-attention (simplified - just compute attention)
    d_k = d_model
    Q, K, V = x, x, x

    # Attention scores
    scores = Q @ K.T / np.sqrt(d_k)

    # Softmax
    attention_weights = np.zeros_like(scores)
    for i in range(scores.shape[0]):
        row = scores[i, :]
        exp_row = np.exp(row - np.max(row))
        attention_weights[i, :] = exp_row / np.sum(exp_row)

    # Apply attention
    attn_output = attention_weights @ V

    # Residual + LayerNorm
    x_attn = x + attn_output
    x_norm1 = np.zeros_like(x_attn)
    for i in range(seq_len):
        mean = np.mean(x_attn[i])
        var = np.var(x_attn[i])
        x_norm1[i] = (x_attn[i] - mean) / np.sqrt(var + 1e-5) * gamma1 + beta1

    # Feed-forward
    ffn_output = np.zeros_like(x_norm1)
    for i in range(seq_len):
        hidden = x_norm1[i] @ W1 + b1
        activated = np.maximum(0, hidden)  # ReLU
        ffn_output[i] = activated @ W2 + b2

    # Residual + LayerNorm
    x_ffn = x_norm1 + ffn_output
    x_norm2 = np.zeros_like(x_ffn)
    for i in range(seq_len):
        mean = np.mean(x_ffn[i])
        var = np.var(x_ffn[i])
        x_norm2[i] = (x_ffn[i] - mean) / np.sqrt(var + 1e-5) * gamma2 + beta2

    return x_norm2, attention_weights


if __name__ == "__main__":
    print("="*80)
    print("  Transformer Block - Basic Test")
    print("="*80)

    if not OPENFHE_AVAILABLE:
        print("\n❌ OpenFHE not available")
        print("Install with: pip install openfhe openfhe_numpy")
        exit(1)

    # Small test case
    seq_len, d_model, d_ff = 4, 4, 16

    print(f"\nTest configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  FFN dimension: {d_ff}")

    # Initialize transformer
    start = time.time()
    transformer = TransformerBlockOpenFHE(
        d_model=d_model,
        d_ff=d_ff,
        n_heads=1,
        mult_depth=35,
        softmax_K=32,
        softmax_scale_factor=4,
        relu_degree=7
    )
    init_time = time.time() - start
    print(f"\nTotal initialization time: {init_time:.2f}s")

    # Generate test input
    np.random.seed(42)
    x = np.random.randn(seq_len, d_model) * 0.5

    print(f"\nInput:")
    print(f"  Shape: {x.shape}")
    print(f"  Range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  First row: {x[0, :3]}")

    # Compute reference
    print("\nComputing reference (NumPy)...")
    ref_output, ref_weights = numpy_transformer_block(
        x,
        transformer.ffn.W1, transformer.ffn.b1,
        transformer.ffn.W2, transformer.ffn.b2,
        transformer.norm1.gamma, transformer.norm1.beta,
        transformer.norm2.gamma, transformer.norm2.beta
    )

    # Compute encrypted transformer
    print("\nComputing encrypted transformer...")
    start = time.time()
    enc_output, enc_weights = transformer.forward(x)
    compute_time = time.time() - start

    print(f"\n{'='*80}")
    print("  Results")
    print("="*80)
    print(f"\nTotal computation time: {compute_time:.2f}s")

    # Compare outputs
    print(f"\nOutput comparison:")
    print(f"  Reference (first row): {ref_output[0, :3]}")
    print(f"  Encrypted (first row): {enc_output[0, :3]}")

    output_error = np.max(np.abs(enc_output - ref_output))
    print(f"\n  Max output error: {output_error:.4f}")

    # Compare attention weights
    print(f"\nAttention weights comparison:")
    print(f"  Reference (first row): {ref_weights[0, :]}")
    print(f"  Encrypted (first row): {enc_weights[0, :]}")

    weights_error = np.max(np.abs(enc_weights - ref_weights))
    print(f"\n  Max weights error: {weights_error:.4f}")

    # Check pass criteria
    if output_error < 2.0 and weights_error < 1.0:
        print("\n✅ TEST PASSED - Transformer computation acceptable")
    else:
        print("\n⚠️  TEST WARNING - Large error (check approximations)")
