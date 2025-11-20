"""
Example Usage of CKKS Softmax Implementation

This script demonstrates how to use the softmax_ckks module for
computing softmax over encrypted data.
"""

import numpy as np
from softmax_ckks import SoftmaxCKKS


def example_1_basic_usage():
    """Example 1: Basic softmax computation"""
    print("=" * 70)
    print("Example 1: Basic Softmax Computation")
    print("=" * 70)

    # Create softmax instance with default parameters
    # n=128: vector size, K=256: approximation terms, scale_factor=16
    softmax = SoftmaxCKKS(n=128, K=256, scale_factor=16)

    # Create a 128-dimensional input vector (e.g., logits from a neural network)
    np.random.seed(123)
    logits = np.random.randn(128) * 3.0  # Random logits

    # Compute softmax
    probabilities = softmax.softmax(logits)

    print(f"Input logits shape: {logits.shape}")
    print(f"Output probabilities shape: {probabilities.shape}")
    print(f"\nFirst 10 logits: {logits[:10]}")
    print(f"First 10 probabilities: {probabilities[:10]}")
    print(f"\nSum of probabilities: {np.sum(probabilities):.10f}")
    print(f"Max probability: {np.max(probabilities):.6f}")
    print(f"Min probability: {np.min(probabilities):.6f}")


def example_2_attention_mechanism():
    """Example 2: Simulating transformer attention scores"""
    print("\n" + "=" * 70)
    print("Example 2: Transformer Attention Mechanism Simulation")
    print("=" * 70)

    # In transformers, we compute attention weights as softmax of attention scores
    # For a sequence of 128 tokens, we have 128 attention scores

    softmax = SoftmaxCKKS(n=128, K=256, scale_factor=16)

    # Simulated attention scores (query · key^T / sqrt(d_k))
    d_k = 64  # dimension of key vectors
    attention_scores = np.random.randn(128) / np.sqrt(d_k)

    # Compute attention weights
    attention_weights = softmax.softmax(attention_scores)

    print(f"Attention scores (first 10): {attention_scores[:10]}")
    print(f"Attention weights (first 10): {attention_weights[:10]}")
    print(f"\nSum of attention weights: {np.sum(attention_weights):.10f}")

    # The attention weights can now be used to compute weighted sum of values
    # In actual usage: context = sum(attention_weights * values)


def example_3_classification():
    """Example 3: Multi-class classification"""
    print("\n" + "=" * 70)
    print("Example 3: Multi-Class Classification (10 classes)")
    print("=" * 70)

    # For classification with 10 classes, we still use n=128 but only
    # the first 10 values are meaningful
    softmax = SoftmaxCKKS(n=128, K=256, scale_factor=16)

    # Logits for 10 classes (rest padded with zeros)
    logits = np.zeros(128)
    logits[:10] = np.array([2.3, -1.2, 0.5, 3.1, -0.8, 1.7, 0.2, -2.1, 1.0, 0.4])

    probabilities = softmax.softmax(logits)

    # Extract probabilities for the 10 classes
    class_probs = probabilities[:10]

    print(f"Class logits: {logits[:10]}")
    print(f"Class probabilities: {class_probs}")
    print(f"\nPredicted class: {np.argmax(class_probs)}")
    print(f"Confidence: {np.max(class_probs):.4f}")

    # Note: The remaining 118 positions will have very small probabilities
    # due to zero padding, which is expected behavior


def example_4_parameter_tuning():
    """Example 4: Effect of different parameters on accuracy"""
    print("\n" + "=" * 70)
    print("Example 4: Parameter Tuning for Accuracy/Speed Tradeoff")
    print("=" * 70)

    # Reference computation
    test_input = np.random.randn(128)
    reference = np.exp(test_input - np.max(test_input))
    reference = reference / np.sum(reference)

    print(f"Testing different parameter configurations:")
    print(f"{'Config':<30} {'Max Error':<15} {'Mean Error':<15}")
    print("-" * 70)

    configs = [
        {"n": 128, "K": 32, "scale_factor": 4, "name": "Fast (K=32, q=4)"},
        {"n": 128, "K": 64, "scale_factor": 8, "name": "Balanced (K=64, q=8)"},
        {"n": 128, "K": 128, "scale_factor": 16, "name": "Accurate (K=128, q=16)"},
        {
            "n": 128,
            "K": 256,
            "scale_factor": 16,
            "name": "High Precision (K=256, q=16)",
        },
        {
            "n": 128,
            "K": 512,
            "scale_factor": 32,
            "name": "Ultra Precision (K=512, q=32)",
        },
    ]

    for config in configs:
        sm = SoftmaxCKKS(
            n=config["n"], K=config["K"], scale_factor=config["scale_factor"]
        )
        result = sm.softmax(test_input)
        max_error = np.max(np.abs(reference - result))
        mean_error = np.mean(np.abs(reference - result))
        print(f"{config['name']:<30} {max_error:<15.2e} {mean_error:<15.2e}")

    print(
        "\nNote: Higher K and scale_factor increase accuracy but also "
        "computational cost"
    )


def example_5_batch_processing():
    """Example 5: Processing multiple vectors"""
    print("\n" + "=" * 70)
    print("Example 5: Batch Processing Multiple Vectors")
    print("=" * 70)

    softmax = SoftmaxCKKS(n=128, K=256, scale_factor=16)

    # Process a batch of 5 vectors
    batch_size = 5
    batch_logits = np.random.randn(batch_size, 128)

    print(f"Processing batch of {batch_size} vectors...")

    batch_probabilities = np.zeros((batch_size, 128))
    for i in range(batch_size):
        batch_probabilities[i] = softmax.softmax(batch_logits[i])

    print(f"\nBatch input shape: {batch_logits.shape}")
    print(f"Batch output shape: {batch_probabilities.shape}")
    print(f"\nProbability sums for each vector:")
    for i in range(batch_size):
        print(f"  Vector {i}: {np.sum(batch_probabilities[i]):.10f}")


def example_6_ckks_mapping():
    """Example 6: Understanding CKKS encryption mapping"""
    print("\n" + "=" * 70)
    print("Example 6: Mapping to Actual CKKS Encryption")
    print("=" * 70)

    print(
        """
This implementation uses numpy arrays, but maps directly to CKKS operations:

1. ROTATION (in sum_with_rotation):
   - Numpy: np.roll(array, positions)
   - CKKS: EvalRotate(ciphertext, positions)

2. ADDITION:
   - Numpy: array1 + array2
   - CKKS: EvalAdd(ciphertext1, ciphertext2)

3. MULTIPLICATION:
   - Numpy: array1 * array2
   - CKKS: EvalMult(ciphertext1, ciphertext2)

4. DIVISION:
   - Numpy: array1 / array2
   - CKKS: EvalDivide(ciphertext1, ciphertext2)  [using Newton iteration]

5. SCALAR OPERATIONS:
   - Numpy: array / scalar
   - CKKS: EvalMult(ciphertext, 1/scalar) or EvalMultByPlaintext

To adapt this code for actual CKKS:
- Replace np.ndarray with CKKS ciphertext type
- Replace operations with CKKS eval functions
- Handle rescaling and modulus switching appropriately
- The algorithm structure remains identical!

Example with pseudo-CKKS code:
    # Instead of:
    result = array1 + array2

    # Use:
    result = EvalAdd(ciphertext1, ciphertext2)
    result = Rescale(result)  # Manage noise growth
"""
    )


if __name__ == "__main__":
    example_1_basic_usage()
    example_2_attention_mechanism()
    example_3_classification()
    example_4_parameter_tuning()
    example_5_batch_processing()
    example_6_ckks_mapping()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
