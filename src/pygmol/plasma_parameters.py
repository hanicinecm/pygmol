"""Module providing a validation function for the concrete subclasses of the
`PlasmaParameters` ABC, as well as a concrete subclass which can be constructed from a
dict.
"""
import numbers
from typing import Union, Sequence, Tuple

import numpy as np
from numpy import ndarray

from .abc import PlasmaParameters


def plasma_parameters_from_dict(plasma_params_dict: dict) -> PlasmaParameters:
    """Concrete PlasmaParameters subclass instance factory.

    Creates a concrete subclass of the `PlasmaParameters` ABC and defines all the
    (key, value) pairs passed down in `plasma_params_dict` as class attributes.
    Returns an *instance* of the new concrete type. If the `plasma_params_dict` omits
    any of the abstract properties of `PlasmaParameters`, an usual TypeError will be
    raised on instantiation.

    Parameters
    ----------
    plasma_params_dict : dict

    Returns
    -------
    PlasmaParameters
        Instance of a dynamically created *concrete subclass* of the `PlasmaParameters`
        ABC.

    Raises
    ------
    TypeError
        If the `plasma_params_dict` does not adhere to the interface defined by
        `PlasmaParameters` ABC.
    """
    concrete_subclass = type(
        "PlasmaParametersFromDict", (PlasmaParameters,), plasma_params_dict
    )
    concrete_instance = concrete_subclass()
    return concrete_instance


class PlasmaParametersValidationError(Exception):
    """A custom exception signaling inconsistent or unphysical data given by the
    concrete `PlasmaParameters` class instance.
    """

    pass


def validate_plasma_parameters(params: PlasmaParameters):
    """Validation function for concrete `PlasmaParameters` subclasses.

    Various inconsistencies are checked for, such as non-physical values of dimensions,
    pressure, feed flows, etc, and inconsistent lengths of `params.power` and
    `params.t_power` if present.

    Parameters
    ----------
    params : PlasmaParameters

    Raises
    ------
    PlasmaParametersValidationError
        If any of the validation checks fails.
    """
    # ensure physical values:
    if params.radius <= 0 or params.length <= 0:
        raise PlasmaParametersValidationError("Plasma dimensions must be positive!")
    if params.pressure <= 0:
        raise PlasmaParametersValidationError("Plasma pressure must be positive!")
    if params.t_end <= 0:
        raise PlasmaParametersValidationError("Simulation end-time must be positive!")
    # power values need to be non-negative:
    if isinstance(params.power, numbers.Number):
        power_vals = [params.power]
    else:
        power_vals = list(params.power)
    if not all(val >= 0 for val in power_vals):
        raise PlasmaParametersValidationError("All power values must be non negative!")
    # all the feed flows must be non-negative:
    if not all(val >= 0 for val in params.feeds.values()):
        raise PlasmaParametersValidationError(
            "All the gas feed flows must be non-negative!"
        )
    # if power is time-dependent, the power and t_power shapes match:
    if not isinstance(params.power, numbers.Number):
        if params.t_power is None or (len(power_vals) != len(params.t_power)):
            raise PlasmaParametersValidationError(
                "The 'power' and 't_power' attributes must have the same length!"
            )
    # t_power needs to be monotonic and rising:
    if params.t_power is not None:
        if list(sorted(params.t_power)) != list(params.t_power):
            raise PlasmaParametersValidationError(
                "The 't_power' values must be monotonic and rising!"
            )
    # temperatures are positive:
    if params.temp_e <= 0 or params.temp_n <= 0:
        raise PlasmaParametersValidationError("Plasma temperature must be positive!")


def sanitize_power_series(
    t_power: Union[None, Sequence[float]],
    power: Union[float, Sequence[float]],
    t_end: float,
) -> Tuple[ndarray, ndarray]:
    """A helper function taking in the possible `t_power` and `power` attributes of the
    `PlasmaParameters` instance and returning two lists of the same length describing
    the power series.

    The lists get sanitized in the way that they will cover the whole time simulation
    domain and the power time series will be continuous in value.

    At this point consistency checks for monotonic behaviour and shapes should already
    have been done.

    Parameters
    ----------
    t_power : Sequence of float, optional
        Times [s] for the power values defined by the `power` sequence.
    power : float or Sequence of float
        Either a single power values in [W] if ``t_power is None``, or the power values
        in [W] for each time in `t_power`
    t_end : float
        The end-of-simulation time in [sec].
    """
    # if the power is constant, return an interval covering -inf to +inf
    if t_power is None:
        if not isinstance(power, numbers.Number):
            power = power[0]
        return np.array([float("-inf"), float("+inf")]), np.array([power, power])
    # for good measures, prepend -inf and append inf to t_power:
    # at this point, guaranteed that both are sequences of the same len
    t_power = [float("-inf")] + list(t_power) + [float("+inf")]
    power = [power[0]] + list(power) + [power[-1]]
    # make it continuous in value - add ramping in discontinuities.
    d_t = 1e-5 * t_end
    for i in range(len(t_power) - 1):
        if t_power[i] == t_power[i + 1]:
            t_power[i] -= d_t
            t_power[i + 1] += d_t
    if list(sorted(t_power)) != list(t_power):
        raise PlasmaParametersValidationError(
            "The 't_power' values must have been ill-defined!"
        )
    return np.array(t_power), np.array(power)
