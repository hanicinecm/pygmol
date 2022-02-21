import pytest

from pygmol.plasma_parameters import (
    PlasmaParametersValidationError,
    plasma_parameters_from_dict,
    validate_plasma_parameters,
    sanitize_power_series,
)
from pygmol.abc import PlasmaParameters

from .resources import DefaultParamsStat, DefaultParamsDyn, DefaultParamsMinimal


@pytest.fixture
def params_dict():
    return {
        "radius": 0.42,
        "length": 0.42,
        "pressure": 0.42,
        "power": [0.0, 420.0],
        "t_power": [0.0, 0.042],
        "feeds": {"Ar": 420.0},
        "temp_e": 0.42,
        "temp_n": 420.0,
        "t_end": 0.042,
    }


def test_params_static_valid():
    validate_plasma_parameters(DefaultParamsStat())


def test_params_dynamic_valid():
    validate_plasma_parameters(DefaultParamsDyn())


@pytest.mark.parametrize("radius", [-1, 0, 1])
@pytest.mark.parametrize("length", [-1, 0, 1])
def test_invalid_dimensions(radius, length):
    params = DefaultParamsStat(radius=radius, length=length)
    if radius <= 0 or length <= 0:
        with pytest.raises(PlasmaParametersValidationError):
            validate_plasma_parameters(params)
    else:
        # no error should be raised
        validate_plasma_parameters(params)


@pytest.mark.parametrize("pressure", [-1, 0, 1])
def test_invalid_pressure(pressure):
    params = DefaultParamsStat(pressure=pressure)
    if pressure <= 0:
        with pytest.raises(PlasmaParametersValidationError):
            validate_plasma_parameters(params)
    else:
        validate_plasma_parameters(params)


@pytest.mark.parametrize("power", [-5, [-1, 1]])
def test_invalid_power(power):
    params = DefaultParamsDyn(power=power)
    with pytest.raises(PlasmaParametersValidationError):
        validate_plasma_parameters(params)


@pytest.mark.parametrize("power", [0, [0, 1]])
def test_valid_power(power):
    params = DefaultParamsDyn(power=power)
    validate_plasma_parameters(params=params)


@pytest.mark.parametrize(
    "power, t_power",
    [([1], None), ([1, 2], None), ([1, 2], [0]), ([1, 2], [0, 0.5, 1])],
)
def test_invalid_power_series(power, t_power):
    if t_power is not None:
        params = DefaultParamsDyn(power=power, t_power=t_power)
    else:
        params = DefaultParamsStat(power=power)
    with pytest.raises(PlasmaParametersValidationError):
        validate_plasma_parameters(params)


def test_valid_t_power():
    params = DefaultParamsDyn(power=[0, 0, 500, 500, 0, 0], t_power=[0, 1, 1, 2, 2, 3])
    validate_plasma_parameters(params)


def test_invalid_t_power():
    params = DefaultParamsDyn(power=[0, 0, 500, 500, 0, 0], t_power=[0, 1, 2, 3, 2, 1])
    with pytest.raises(PlasmaParametersValidationError):
        validate_plasma_parameters(params)


@pytest.mark.parametrize("t_end", [-1, 0])
def test_invalid_t_end(t_end):
    params = DefaultParamsStat(t_end=t_end)
    with pytest.raises(PlasmaParametersValidationError):
        validate_plasma_parameters(params)


def test_invalid_feeds():
    params = DefaultParamsStat()
    params.feeds["Ar"] = -1
    with pytest.raises(PlasmaParametersValidationError):
        validate_plasma_parameters(params)
    params.feeds["Ar"] = 0  # this should be fine
    validate_plasma_parameters(params)


@pytest.mark.parametrize("temp", [-1, 0])
def test_invalid_temperatures(temp):
    params = DefaultParamsStat(temp_e=temp)
    with pytest.raises(PlasmaParametersValidationError):
        validate_plasma_parameters(params)
    params = DefaultParamsStat(temp_n=temp)
    with pytest.raises(PlasmaParametersValidationError):
        validate_plasma_parameters(params)


def test_defaults():
    params = DefaultParamsMinimal()
    assert params.t_power is None
    assert params.feeds == {}
    assert params.temp_e == 1.0
    assert params.temp_n == 300.0
    assert params.t_end == 1.0


def test_sanitize_power_series():
    t_power, power = sanitize_power_series(
        t_power=[50.0, 50.0], power=[0.0, 500.0], t_end=100.0
    )
    assert list(t_power) == [-float("inf"), 49.999, 50.001, float("inf")]
    assert list(power) == [0, 0, 500, 500]

    with pytest.raises(PlasmaParametersValidationError):
        sanitize_power_series([1, 1, 1], [400, 500, 600], 2)

    assert [list(a) for a in sanitize_power_series(None, 42, 0.1)] == [
        [-float("inf"), float("inf")],
        [42, 42],
    ]
    assert [list(a) for a in sanitize_power_series(None, [42], 0.1)] == [
        [-float("inf"), float("inf")],
        [42, 42],
    ]
    assert [list(a) for a in sanitize_power_series([0.05], [42], 0.1)] == [
        [-float("inf"), 0.05, float("inf")],
        [42, 42, 42],
    ]


@pytest.mark.parametrize("power,t_power", [([0, 420], [0, 0.42]), (420, None)])
def test_params_from_dict_valid(power, t_power, params_dict):
    params_dict["power"] = power
    params_dict["t_power"] = t_power
    params = plasma_parameters_from_dict(params_dict)
    assert isinstance(params, PlasmaParameters)
    assert params.radius == 0.42
    assert params.length == 0.42
    assert params.pressure == 0.42
    assert params.power == power
    assert params.t_power == t_power
    assert params.feeds == {"Ar": 420}
    assert params.temp_e == 0.42
    assert params.temp_n == 420
    assert params.t_end == 0.042


@pytest.mark.parametrize("missing_attrib", ["radius", "length", "pressure", "power"])
def test_params_from_dict_invalid(missing_attrib, params_dict):
    params_dict = {
        key: params_dict[key] for key in params_dict if key != missing_attrib
    }
    # params_dict missing one of the abstract attributes, must raise TypeError
    with pytest.raises(TypeError):
        plasma_parameters_from_dict(params_dict)


def test_params_from_dict_defaults():
    params_dict = {"radius": 1, "length": 1, "pressure": 1, "power": 1}
    params_instance = plasma_parameters_from_dict(params_dict)
    assert params_instance.t_power is None
    assert params_instance.feeds == {}
    assert params_instance.temp_e == 1.0
    assert params_instance.temp_n == 300.0
    assert params_instance.t_end == 1.0
