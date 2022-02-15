from typing import Union

from .abc import Chemistry, PlasmaParameters, Equations
from .chemistry import ChemistryFromDict, validate_chemistry
from .plasma_parameters import PlasmaParametersFromDict, validate_plasma_parameters
from .equations import ElectronEnergyEquations


class ModelArgumentsValidationError(Exception):
    """A custom exception signaling the `Chemistry` instance inconsistent with the
    `PlasmaParameters` instance.
    """

    pass


def validate_model_arguments(chemistry: Chemistry, plasma_params: PlasmaParameters):
    """

    Raises
    ------
    ChemistryValidationError
    PlasmaParametersValidationError
    ModelArgumentsValidationError
    """
    validate_chemistry(chemistry)
    validate_plasma_parameters(plasma_params)
    if not set(plasma_params.feeds).issubset(chemistry.species_ids):
        raise ModelArgumentsValidationError(
            "Feed gas species defined in the plasma parameters are inconsistent with "
            "the chemistry!"
        )


class Model:
    def __init__(
        self,
        chemistry: Union[Chemistry, dict],
        plasma_params: Union[PlasmaParameters, dict],
    ):
        """The global model initializer.

        The model instance solves for the equations defined by the
        `ElectronEnergyEquations` class.
        """
        if isinstance(chemistry, dict):
            chemistry = ChemistryFromDict(chemistry_dict=chemistry)
        if isinstance(plasma_params, dict):
            plasma_params = PlasmaParametersFromDict(plasma_params_dict=plasma_params)
        validate_model_arguments(chemistry, plasma_params)
        self.equations = ElectronEnergyEquations(chemistry, plasma_params)

        # placeholder for whatever the selected solver returns:
        self.solution_raw = None
        # placeholder for the array of time samples [sec]:
        self.t = None
        # 2D array of state vectors `y` for all time samples:
        self.solution_primary = None
        # `pandas.DataFrame` of all the final solution values:
        self.sol_secondary = None

        raise NotImplementedError
