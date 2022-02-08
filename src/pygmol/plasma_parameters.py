"""Module containing the PlasmaParameters wrapper class handling the
validation and sanitisation of parameters passed to the global model.
"""
import numbers
from typing import Union, Sequence, Dict

import pydantic


class PlasmaParameters(pydantic.BaseModel):
    # noinspection PyUnresolvedReferences
    """Data class of parameters needed as an input to the global model

    Some basic validation is performed on the parameter values, ensuring
    their physicality and sanity.

    The `PlasmaParameters` instance can be converted to dict by
    ``dict(plasma_parameters)``.

    Attributes
    ----------
    radius, length : float
        Dimensions of cylindrical plasma in [m].
    pressure : float
        Plasma pressure set-point in [Pa].
    power : float or Sequence[float]
        Power deposited into plasma in [W]. Either a single number,
        or a sequence of numbers for time-dependent power
    t_power : None or Sequence[float], default=None
        If `power` passed is a sequence, the `t_power` needs to be
        passed, defining the time-points for the `power` values and
        having the same length.
        Defaults to None, which is fine for time-independent `power`.
    feeds : dict[str, float]
        Dictionary of feed flows in [sccm] for all the species, which
        are fed to plasma. The feed flows are keyed by species IDs
        (distinct names/formulas/... - need to be subset of the values
        returned by `Chemistry.species_ids`).
    t_end : float
        End time of the simulation in [s].

    Raises
    ------
    ValidationError
        If the values passed are inconsistent or unphysical.
    """
    radius: float
    length: float
    pressure: float
    power: Union[float, Sequence[float]]
    t_power: Sequence[float] = None
    feeds: Dict[str, float]
    t_end: float

    @pydantic.validator("radius")
    def radius_positive(cls, value):
        if value <= 0:
            raise ValueError("Plasma dimensions must be positive!")
        return value

    @pydantic.validator("length")
    def length_positive(cls, value):
        if value <= 0:
            raise ValueError("Plasma dimensions must be positive!")
        return value

    @pydantic.validator("pressure")
    def pressure_positive(cls, value):
        if value <= 0:
            raise ValueError("Plasma dimensions must be positive!")
        return value

    @pydantic.validator("power")
    def power_non_negative(cls, value):
        if isinstance(value, Sequence):
            if not all(val >= 0 for val in value):
                raise ValueError("All power values must be non negative!")
        elif value < 0:
            raise ValueError("All power values must be non negative!")
        return value

    @pydantic.validator("t_end")
    def end_time_positive(cls, value):
        if value <= 0:
            raise ValueError("The simulation end time must be positive!")
        return value

    @pydantic.validator("feeds")
    def feeds_non_negative(cls, value):
        if not all(val >= 0 for val in value.values()):
            raise ValueError("All the gas feed flows must be non-negative!")
        return value

    @pydantic.root_validator
    def power_series_consistent(cls, values):
        if not isinstance(values.get("power"), numbers.Number):
            if values.get("t_power") is None or len(values.get("t_power")) != len(
                values.get("power")
            ):
                raise ValueError(
                    "The 'power' and 't_power' attributes must have the same length!"
                )
        return values
