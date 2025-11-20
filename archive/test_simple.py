"""Minimal test for OpenFHE softmax"""
import numpy as np
import time

print("Importing modules...")
from softmax_openfhe import SoftmaxCKKSOpenFHE, OPENFHE_AVAILABLE

if not OPENFHE_AVAILABLE:
    print("OpenFHE not available")
    exit(1)

print("OpenFHE available ✅")

print("\nInitializing (n=128, K=64)...")
start = time.time()
sm = SoftmaxCKKSOpenFHE(n=128, K=64, scale_factor=8, mult_depth=25)
print(f"Init time: {time.time()-start:.2f}s ✅")

print("\nComputing softmax...")
x = np.random.randn(128)
start = time.time()
result = sm.softmax_encrypted(x)
print(f"Compute time: {time.time()-start:.2f}s ✅")

print(f"\nSum: {np.sum(result):.6f}")
print(f"First 5 values: {result[:5]}")
print("\n✅ Test completed!")
