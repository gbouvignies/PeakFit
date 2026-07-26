from typing import Any, cast

import pytest

from peakfit.engine.domain.params_scalar import Parameters


def test_add_parameter_uses_explicit_bound_names() -> None:
    params = Parameters()

    params.add("peak.F2.cs", value=8.1, min_value=7.0, max_value=9.0)

    param = params["peak.F2.cs"]
    assert param.min == 7.0
    assert param.max == 9.0


def test_add_parameter_rejects_unknown_options() -> None:
    params = Parameters()
    add = cast("Any", params.add)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        add("peak.F2.cs", value=8.1, lower=7.0)
