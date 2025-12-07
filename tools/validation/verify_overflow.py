import numpy as np
import warnings
from peakfit.core.fitting.computation import calculate_amplitudes


def test_overflow_handling():
    # Create inputs that will cause overflow in matmul
    # Using very large numbers
    shapes = np.full((2, 100), 1e200)
    data = np.full((100,), 1e200)

    # Catch warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            result = calculate_amplitudes(shapes, data)
        except Exception as e:
            print(f"FAILED: Exception raised: {e}")
            return

        # Check that no warning was raised (should be caught by errstate)
        overflow_warnings = [str(warn.message) for warn in w if "overflow" in str(warn.message)]
        if overflow_warnings:
            print(f"FAILED: Overflow warnings raised: {overflow_warnings}")
        else:
            print("SUCCESS: No overflow warnings raised")

        # Check result (likely NaNs or Infs, but shouldn't crash)
        print(f"Result: {result}")


if __name__ == "__main__":
    test_overflow_handling()
