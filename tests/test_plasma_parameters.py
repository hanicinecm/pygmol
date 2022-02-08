import pytest

from pydantic import ValidationError

from pygmol.plasma_parameters import PlasmaParameters


@pytest.fixture
def params_dynamic():
    return {
        "radius": 1.0,
        "length": 1.0,
        "pressure": 1.0,
        "power": [0.0, 1.0],
        "t_power": [0.0, 1.0],
        "feeds": {"Ar": 1.0},
        "t_end": 1.0,
    }


@pytest.fixture
def params_static():
    return {
        "radius": 1.0,
        "length": 1.0,
        "pressure": 1.0,
        "power": 1.0,
        "t_power": None,
        "feeds": {"Ar": 1.0},
        "t_end": 1.0,
    }


def test_params_static(params_static):
    PlasmaParameters(**params_static)


def test_params_dynamic(params_dynamic):
    PlasmaParameters(**params_dynamic)


@pytest.mark.parametrize("radius", [-1, 0, 1])
@pytest.mark.parametrize("length", [-1, 0, 1])
def test_invalid_dimensions(params_static, radius, length):
    params_static["radius"] = radius
    params_static["length"] = length
    if radius <= 0 or length <= 0:
        with pytest.raises(ValidationError):
            PlasmaParameters(**params_static)
    else:
        # no error should be raised
        PlasmaParameters(**params_static)


@pytest.mark.parametrize("pressure", [-1, 0, 1])
def test_invalid_pressure(params_static, pressure):
    params_static["pressure"] = pressure
    if pressure <= 0:
        with pytest.raises(ValidationError):
            PlasmaParameters(**params_static)
    else:
        PlasmaParameters(**params_static)


@pytest.mark.parametrize("power", [-5, [-1, 1]])
def test_invalid_power(params_dynamic, power):
    params_dynamic["power"] = power
    with pytest.raises(ValidationError):
        PlasmaParameters(**params_dynamic)


@pytest.mark.parametrize("power", [0, [0, 1]])
def test_valid_power(params_dynamic, power):
    params_dynamic["power"] = power
    PlasmaParameters(**params_dynamic)


# noinspection PyTypeChecker
@pytest.mark.parametrize(
    "power, t_power",
    [([1], None), ([1, 2], None), ([1, 2], [0]), ([1, 2], [0, 0.5, 1])],
)
def test_invalid_power_series(params_static, power, t_power):
    params_static["power"] = power
    if t_power is not None:
        params_static["t_power"] = t_power
    with pytest.raises(ValidationError):
        PlasmaParameters(**params_static)


@pytest.mark.parametrize("t_end", [-1, 0])
def test_invalid_t_end(params_static, t_end):
    params_static["t_end"] = t_end
    with pytest.raises(ValidationError):
        PlasmaParameters(**params_static)


def test_invalid_feeds(params_static):
    params_static["feeds"]["Ar"] = -1
    with pytest.raises(ValidationError):
        PlasmaParameters(**params_static)
    params_static["feeds"]["Ar"] = 0  # this should be fine
    PlasmaParameters(**params_static)


def test_equivalency_with_dict(params_static, params_dynamic):
    assert dict(PlasmaParameters(**params_static)) == params_static
    assert dict(PlasmaParameters(**params_dynamic)) == params_dynamic
