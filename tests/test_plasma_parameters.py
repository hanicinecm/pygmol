import pytest

from pygmol.plasma_parameters import (
    PlasmaParametersValidationError,
    validate_plasma_parameters,
)

from .resources import DefaultParamsStat, DefaultParamsDyn


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
