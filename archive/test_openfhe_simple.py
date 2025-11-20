"""
Simple test to debug OpenFHE CKKS operations
"""

import numpy as np
from openfhe import *


def test_basic_operations():
    """Test basic CKKS operations"""
    print("Testing Basic CKKS Operations")
    print("=" * 70)

    # Initialize CKKS
    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(10)
    params.SetScalingModSize(59)
    params.SetFirstModSize(60)
    params.SetScalingTechnique(FIXEDAUTO)
    params.SetSecretKeyDist(UNIFORM_TERNARY)
    params.SetBatchSize(8)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)
    cc.EvalRotateKeyGen(keys.secretKey, [1, 2, 4])

    # Test 1: Basic encryption/decryption
    print("\n1. Testing encryption/decryption")
    plaintext_vec = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    pt = cc.MakeCKKSPackedPlaintext(plaintext_vec)
    ct = cc.Encrypt(keys.publicKey, pt)

    decrypted_pt = cc.Decrypt(keys.secretKey, ct)
    decrypted_pt.SetLength(8)
    decrypted = decrypted_pt.GetRealPackedValue()[:8]

    print(f"Original:  {plaintext_vec}")
    print(f"Decrypted: {[f'{x:.6f}' for x in decrypted]}")
    print(f"Error: {np.max(np.abs(np.array(plaintext_vec) - np.array(decrypted))):.2e}")

    # Test 2: Addition
    print("\n2. Testing addition")
    ct2 = cc.EvalAdd(ct, ct)
    decrypted_pt2 = cc.Decrypt(keys.secretKey, ct2)
    decrypted_pt2.SetLength(8)
    decrypted2 = decrypted_pt2.GetRealPackedValue()[:8]

    expected = [2.0 * x for x in plaintext_vec]
    print(f"Expected (2x): {expected}")
    print(f"Decrypted:     {[f'{x:.6f}' for x in decrypted2]}")
    print(f"Error: {np.max(np.abs(np.array(expected) - np.array(decrypted2))):.2e}")

    # Test 3: Multiplication by plaintext
    print("\n3. Testing multiplication by plaintext")
    scalar_pt = cc.MakeCKKSPackedPlaintext([0.5])
    ct3 = cc.EvalMult(ct, scalar_pt)
    decrypted_pt3 = cc.Decrypt(keys.secretKey, ct3)
    decrypted_pt3.SetLength(8)
    decrypted3 = decrypted_pt3.GetRealPackedValue()[:8]

    expected3 = [0.5 * x for x in plaintext_vec]
    print(f"Expected (0.5x): {expected3}")
    print(f"Decrypted:       {[f'{x:.6f}' for x in decrypted3]}")
    print(f"Error: {np.max(np.abs(np.array(expected3) - np.array(decrypted3))):.2e}")

    # Test 4: Ciphertext-ciphertext multiplication
    print("\n4. Testing ciphertext-ciphertext multiplication")
    ct4 = cc.EvalMult(ct, ct)
    decrypted_pt4 = cc.Decrypt(keys.secretKey, ct4)
    decrypted_pt4.SetLength(8)
    decrypted4 = decrypted_pt4.GetRealPackedValue()[:8]

    expected4 = [x * x for x in plaintext_vec]
    print(f"Expected (x^2): {expected4}")
    print(f"Decrypted:      {[f'{x:.6f}' for x in decrypted4]}")
    print(f"Error: {np.max(np.abs(np.array(expected4) - np.array(decrypted4))):.2e}")

    # Test 5: Rotation
    print("\n5. Testing rotation")
    ct5 = cc.EvalRotate(ct, 1)
    decrypted_pt5 = cc.Decrypt(keys.secretKey, ct5)
    decrypted_pt5.SetLength(8)
    decrypted5 = decrypted_pt5.GetRealPackedValue()[:8]

    expected5 = plaintext_vec[1:] + [plaintext_vec[0]]
    print(f"Expected (rotate 1): {expected5}")
    print(f"Decrypted:           {[f'{x:.6f}' for x in decrypted5]}")
    print(f"Error: {np.max(np.abs(np.array(expected5) - np.array(decrypted5))):.2e}")


def test_exponential_approx():
    """Test simple exponential approximation"""
    print("\n" + "=" * 70)
    print("Testing Exponential Approximation")
    print("=" * 70)

    # Test on plaintext first
    x = 0.5
    K = 8
    scale = 2

    # Scale input
    x_scaled = x / scale

    # Compute approximation
    result = 0
    term = 1
    for i in range(1, K + 1):
        term *= x_scaled / i
        result += term

    # Scale back
    for _ in range(int(np.log2(scale))):
        result = result * (result + 2)

    result = result + 1  # Add 1 to get e^x

    expected = np.exp(x)
    print(f"\nInput: {x}")
    print(f"Approximation (K={K}, scale={scale}): {result:.6f}")
    print(f"Numpy exp: {expected:.6f}")
    print(f"Error: {abs(result - expected):.2e}")


if __name__ == "__main__":
    test_basic_operations()
    test_exponential_approx()
