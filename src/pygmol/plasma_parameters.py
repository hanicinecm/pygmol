"""Module containing the PlasmaParameters wrapper class handling the
validation and sanitisation of parameters passed to the global model.
"""
import numbers

from .abc import PlasmaParameters


# noinspection PyAbstractClass
class PlasmaParametersFromDict(PlasmaParameters):
    """A `PlasmaParameters` subclass built from a dictionary passed.

    The `plasma_params` needs to mirror the interface defined by the
    `PlasmaParameters` ABC.
    """

    def __init__(self, plasma_params, *args, **kwargs):
        for attr, val in plasma_params.items():
            setattr(self, attr, val)
        super().__init__(*args, **kwargs)


class PlasmaParametersValidationError(Exception):
    """A custom exception signaling inconsistent or unphysical data
    given by the concrete `PlasmaParameters` class instance.
    """

    pass


def validate_plasma_parameters(params: PlasmaParameters):
    """Validation function for concrete `PlasmaParameters` subclasses.

    Various inconsistencies are checked for, such as non-physical values
    of dimensions, pressure, feed flows, etc, and inconsistent lengths
    of `params.power` and `params.t_power` if present.

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
    if params.temp_e <= 0 or params.temp_n <= 0:
        raise PlasmaParametersValidationError("Plasma temperature must be positive!")
