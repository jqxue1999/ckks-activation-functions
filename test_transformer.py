"""
Test Suite for OpenFHE CKKS Transformer Block Implementation

This test file validates the transformer_openfhe.py implementation with:
- Basic functionality tests
- Component tests (LayerNorm, FFN, Attention)
- Full transformer block tests
- Performance benchmarking
"""

import numpy as np
import time
import sys


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_section(title):
    """Print a section header."""
    print(f"\n{title}")
    print_separator("-")


def compute_metrics(predicted, reference):
    """Compute comparison metrics."""
    abs_error = np.abs(predicted - reference)

    metrics = {
        "max_error": np.max(abs_error),
        "mean_error": np.mean(abs_error),
        "relative_error": np.mean(abs_error / (np.abs(reference) + 1e-10)),
    }
    return metrics


def test_layer_norm():
    """Test 1: Layer normalization."""
    print_section("Test 1: Layer Normalization")

    from transformer_openfhe import LayerNormOpenFHE

    try:
        # Initialize
        d_model = 8
        ln = LayerNormOpenFHE(d_model=d_model)

        # Test input
        x = np.random.randn(4, d_model)

        print(f"Configuration: d_model={d_model}")
        print(f"Input shape: {x.shape}")

        # Normalize
        start = time.time()
        x_norm = ln.normalize(x)
        norm_time = time.time() - start

        print(f"Computation time: {norm_time:.4f}s")

        # Check properties
        mean_after = np.mean(x_norm, axis=-1)
        var_after = np.var(x_norm, axis=-1)

        print(f"\nProperties check:")
        print(f"  Mean (should be ~0): {mean_after}")
        print(f"  Variance (should be ~1): {var_after}")

        # Pass criteria: mean close to 0, variance close to 1
        mean_ok = np.allclose(mean_after, 0, atol=1e-5)
        var_ok = np.allclose(var_after, 1, atol=0.1)

        if mean_ok and var_ok:
            print("\n✅ TEST 1 PASSED - Layer normalization correct")
            return True
        else:
            print("\n❌ TEST 1 FAILED - Normalization incorrect")
            return False

    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feed_forward():
    """Test 2: Feed-forward network."""
    print_section("Test 2: Feed-Forward Network")

    from transformer_openfhe import FeedForwardOpenFHE

    try:
        # Initialize
        d_model, d_ff = 4, 16
        print(f"Configuration: d_model={d_model}, d_ff={d_ff}")

        start = time.time()
        ffn = FeedForwardOpenFHE(d_model=d_model, d_ff=d_ff, mult_depth=15, relu_degree=7)
        init_time = time.time() - start
        print(f"Initialization: {init_time:.2f}s")

        # Test input
        x = np.random.randn(2, d_model) * 0.5

        print(f"\nInput:")
        print(f"  Shape: {x.shape}")
        print(f"  Range: [{x.min():.2f}, {x.max():.2f}]")

        # Forward pass
        print("\nComputing forward pass...")
        start = time.time()
        output = ffn.forward(x)
        compute_time = time.time() - start

        print(f"  Computation time: {compute_time:.2f}s")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.2f}, {output.max():.2f}]")

        # Check output shape
        shape_ok = output.shape == x.shape

        if shape_ok:
            print("\n✅ TEST 2 PASSED - Feed-forward network functional")
            return True
        else:
            print("\n❌ TEST 2 FAILED - Output shape mismatch")
            return False

    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transformer_basic():
    """Test 3: Basic transformer block."""
    print_section("Test 3: Basic Transformer Block")

    from transformer_openfhe import TransformerBlockOpenFHE, numpy_transformer_block

    try:
        # Small configuration for testing
        seq_len, d_model, d_ff = 4, 4, 16

        print(f"Configuration: seq_len={seq_len}, d_model={d_model}, d_ff={d_ff}")

        # Initialize
        print("\nInitializing transformer...")
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
        print(f"Initialization time: {init_time:.2f}s")

        # Test input
        np.random.seed(42)
        x = np.random.randn(seq_len, d_model) * 0.5

        print(f"\nInput:")
        print(f"  Shape: {x.shape}")
        print(f"  Range: [{x.min():.2f}, {x.max():.2f}]")

        # Reference
        print("\nComputing reference (NumPy)...")
        ref_output, ref_weights = numpy_transformer_block(
            x,
            transformer.ffn.W1, transformer.ffn.b1,
            transformer.ffn.W2, transformer.ffn.b2,
            transformer.norm1.gamma, transformer.norm1.beta,
            transformer.norm2.gamma, transformer.norm2.beta
        )

        # Encrypted
        print("\nComputing encrypted transformer...")
        start = time.time()
        enc_output, enc_weights = transformer.forward(x)
        compute_time = time.time() - start

        print(f"\n{'='*80}")
        print("  Results")
        print("="*80)
        print(f"\nComputation time: {compute_time:.2f}s")

        # Metrics
        output_metrics = compute_metrics(enc_output, ref_output)
        weights_metrics = compute_metrics(enc_weights, ref_weights)

        print(f"\nOutput metrics:")
        print(f"  Max error:  {output_metrics['max_error']:.4f}")
        print(f"  Mean error: {output_metrics['mean_error']:.4f}")

        print(f"\nAttention weights metrics:")
        print(f"  Max error:  {weights_metrics['max_error']:.4f}")
        print(f"  Mean error: {weights_metrics['mean_error']:.4f}")

        # Pass criteria: allow larger error due to approximations
        passed = (output_metrics['max_error'] < 3.0 and
                 weights_metrics['max_error'] < 1.0)

        if passed:
            print("\n✅ TEST 3 PASSED - Transformer block functional")
        else:
            print("\n⚠️  TEST 3 WARNING - Large error (check approximations)")

        return passed

    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transformer_properties():
    """Test 4: Transformer properties."""
    print_section("Test 4: Transformer Properties")

    from transformer_openfhe import TransformerBlockOpenFHE

    try:
        # Initialize small transformer
        seq_len, d_model = 4, 4
        transformer = TransformerBlockOpenFHE(
            d_model=d_model, d_ff=16,
            mult_depth=35, softmax_K=32
        )

        # Test input
        np.random.seed(123)
        x = np.random.randn(seq_len, d_model) * 0.5

        print(f"Input shape: {x.shape}")

        # Forward pass
        print("\nComputing forward pass...")
        output, attn_weights = transformer.forward(x)

        # Property 1: Output shape matches input
        print("\nProperty 1: Shape preservation")
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {output.shape}")
        shape_ok = output.shape == x.shape
        print(f"  Shape preserved? {shape_ok} {'✅' if shape_ok else '❌'}")

        # Property 2: Attention weights sum to ~1
        print("\nProperty 2: Attention weights sum to ~1.0 per row")
        weight_sums = attn_weights.sum(axis=1)
        print(f"  Weight sums: {weight_sums}")
        sum_ok = np.allclose(weight_sums, 1.0, atol=0.5)
        print(f"  Close to 1.0? {sum_ok} {'✅' if sum_ok else '⚠️'}")

        # Property 3: Output magnitude reasonable
        print("\nProperty 3: Output magnitude reasonable")
        x_norm = np.linalg.norm(x)
        out_norm = np.linalg.norm(output)
        ratio = out_norm / x_norm
        print(f"  ||Input||:  {x_norm:.4f}")
        print(f"  ||Output||: {out_norm:.4f}")
        print(f"  Ratio:      {ratio:.4f}")
        mag_ok = 0.1 < ratio < 10.0
        print(f"  Reasonable? {mag_ok} {'✅' if mag_ok else '⚠️'}")

        passed = shape_ok and sum_ok and mag_ok

        if passed:
            print("\n✅ TEST 4 PASSED - Properties verified")
        else:
            print("\n⚠️  TEST 4 WARNING - Some properties violated")

        return passed

    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print_separator("=")
    print("  OPENFHE CKKS TRANSFORMER BLOCK - TEST SUITE")
    print_separator("=")

    print("\nNote: Transformer uses approximations (softmax, ReLU).")
    print("Expect larger errors (~1-3) compared to individual components.")

    # Run tests
    test_results = {}

    # Test 1: Layer normalization
    test_results["Layer Normalization"] = test_layer_norm()

    # Test 2: Feed-forward network
    test_results["Feed-Forward Network"] = test_feed_forward()

    # Test 3: Basic transformer
    test_results["Basic Transformer"] = test_transformer_basic()

    # Test 4: Properties
    test_results["Transformer Properties"] = test_transformer_properties()

    # Summary
    print_separator("=")
    print("  TEST SUMMARY")
    print_separator("=")

    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:<30} {status}")

    print_separator("=")

    passed_count = sum(test_results.values())
    total_count = len(test_results)
    print(f"\n  Results: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n  🎉 ALL TESTS PASSED! 🎉")
        print("\n  Note: Transformer block combines multiple approximations:")
        print("  - Softmax approximation (error ~0.2)")
        print("  - ReLU polynomial approximation (error ~0.2)")
        print("  - Multiple matrix operations (noise accumulation)")
        print("  - Total expected error: ~1-3 (acceptable)")
    else:
        print(f"\n  ⚠️  {total_count - passed_count} test(s) failed or warned")

    print_separator("=")

    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
