"""Minimal scipy crash test."""
import sys
import os

# Disable all threading before any imports
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
print(f"NumPy: {np.__version__}, from: {np.__file__}")
sys.stdout.flush()

import scipy
print(f"SciPy: {scipy.__version__}, from: {scipy.__file__}")
sys.stdout.flush()

from scipy.signal import butter, filtfilt, lfilter, sosfiltfilt, sosfilt

fs = 200
signal = np.random.randn(2000).astype(np.float64)

# Test 1: butter
print("\n1. butter...", end=" ")
sys.stdout.flush()
b, a = butter(4, [0.5/100, 40.0/100], btype='band')
print("OK")
sys.stdout.flush()

# Test 2: lfilter (single pass - simpler than filtfilt)
print("2. lfilter...", end=" ")
sys.stdout.flush()
try:
    result = lfilter(b, a, signal)
    print(f"OK ({len(result)})")
    sys.stdout.flush()
except Exception as e:
    print(f"FAILED: {e}")

# Test 3: filtfilt
print("3. filtfilt...", end=" ")
sys.stdout.flush()
try:
    result = filtfilt(b, a, signal)
    print(f"OK ({len(result)})")
    sys.stdout.flush()
except Exception as e:
    print(f"FAILED: {e}")

# Test 4: SOS form (alternative implementation path)
print("4. butter sos...", end=" ")
sys.stdout.flush()
sos = butter(4, [0.5/100, 40.0/100], btype='band', output='sos')
print("OK")
sys.stdout.flush()

# Test 5: sosfiltfilt
print("5. sosfiltfilt...", end=" ")
sys.stdout.flush()
try:
    result = sosfiltfilt(sos, signal)
    print(f"OK ({len(result)})")
    sys.stdout.flush()
except Exception as e:
    print(f"FAILED: {e}")

# Test 6: sosfilt (single pass)
print("6. sosfilt...", end=" ")
sys.stdout.flush()
try:
    result = sosfilt(sos, signal)
    print(f"OK ({len(result)})")
    sys.stdout.flush()
except Exception as e:
    print(f"FAILED: {e}")

# Test 7: Check which BLAS/LAPACK scipy uses
print("\n7. SciPy BLAS config:")
sys.stdout.flush()
try:
    from scipy import show_config
    show_config()
except Exception as e:
    print(f"  Could not show config: {e}")

print("\n--- Done ---")
