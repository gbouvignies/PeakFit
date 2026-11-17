"""The model uses exactly its parameters: none are silently replaced by defaults."""

import numpy as np
import pytest
from builders import peak_pair

from peakfit.engine.domain.params_scalar import Parameters


def _without(params, name):
    kept = Parameters()
    for key, parameter in params.items():
        if key != name:
            kept[key] = parameter
    return kept


@pytest.mark.parametrize(
    ("family", "phase", "missing"),
    [
        ("gaussian", False, "A.F2.lw"),
        ("pvoigt", False, "B.F2.eta"),
        ("no_apod", True, "cluster_1.F2.phase"),
    ],
)
def test_a_missing_model_parameter_raises_instead_of_taking_its_default(family, phase, missing):
    cluster, params = peak_pair(6.0, family=family, phase=phase)
    assert missing in params
    with pytest.raises(ValueError, match=rf"Model parameter {missing} is missing"):
        cluster.evaluate(_without(params, missing))


def test_an_unfitted_phase_is_absent_from_the_model_not_a_default_value():
    unfitted, params = peak_pair(6.0, family="no_apod")
    assert not any(name.endswith(".phase") for name in params)
    shape = unfitted.peaks[0].shapes[0]
    assert "phase" not in shape.get_cluster_parameters(unfitted.peaks, params).extras
    fitted, fitted_params = peak_pair(6.0, family="no_apod", phase=True)
    # A fitted phase at zero is the same model as no phase correction.
    np.testing.assert_array_equal(fitted.evaluate(fitted_params), unfitted.evaluate(params))
