import numpy as np
import warnings
from peakfit.core.fitting.computation import calculate_amplitudes


def test_nan_handling():
    # Create inputs with NaNs
    shapes = np.array([[1.0, 2.0], [np.nan, 4.0]])
    data = np.array([1.0, 2.0])

    # Catch warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = calculate_amplitudes(shapes, data)

        # Check that no warning was raised
        if len(w) > 0:
            print(f"FAILED: Warnings raised: {[str(warn.message) for warn in w]}")
        else:
            print("SUCCESS: No warnings raised")

        # Check result is NaN
        if np.all(np.isnan(result)):
            print("SUCCESS: Result contains NaNs as expected")
        else:
            print(f"FAILED: Result should be NaNs, got {result}")


if __name__ == "__main__":
    test_nan_handling()
