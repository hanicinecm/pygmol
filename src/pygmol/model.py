from typing import Union, Mapping

from pandas import DataFrame
import pandas
from numpy import ndarray
from scipy.integrate import solve_ivp

from .abc import Chemistry, PlasmaParameters
from .chemistry import ChemistryFromDict, validate_chemistry
from .plasma_parameters import (
    PlasmaParametersFromDict,
    validate_plasma_parameters,
    PlasmaParametersValidationError,
)
from .equations import ElectronEnergyEquations


class ModelSolutionError(Exception):
    """Custom exception signaling problems with the global model solution."""

    pass


class Model:
    def __init__(
        self,
        chemistry: Union[Chemistry, dict],
        plasma_params: Union[PlasmaParameters, dict],
    ):
        """The global model initializer.

        The model instance solves for the equations defined by the
        `ElectronEnergyEquations` class.

        Raises
        ------
        ChemistryValidationError
            Signals inconsistent chemistry passed.
        PlasmaParametersValidationError
            Signals inconsistent plasma parameters passed
        """
        if isinstance(chemistry, dict):
            chemistry = ChemistryFromDict(chemistry_dict=chemistry)
        if isinstance(plasma_params, dict):
            plasma_params = PlasmaParametersFromDict(plasma_params_dict=plasma_params)

        # validations:
        validate_chemistry(chemistry)
        validate_plasma_parameters(plasma_params)
        if not set(plasma_params.feeds).issubset(chemistry.species_ids):
            raise PlasmaParametersValidationError(
                "Feed gas species defined in the plasma parameters are inconsistent "
                "with the chemistry species ids!"
            )
        self.chemistry = chemistry
        self.plasma_params = plasma_params

        # equations employed by the model:
        self.equations = ElectronEnergyEquations(chemistry, plasma_params)

        # placeholder for whatever the selected low-level solver returns:
        self.solution_raw = None
        # placeholder for the array of time samples [sec]:
        self.t = None
        # 2D array of state vectors `y` for all time samples:
        self.solution_primary = None
        # `pandas.DataFrame` of all the final solution values:
        self.solution = None

    def _reinitialize_equations(self):
        self.equations = ElectronEnergyEquations(self.chemistry, self.plasma_params)

    def _solve(
        self, y0: ndarray = None, method: str = "BDF", reset_equations: bool = False
    ):
        """Runs the low-level solver (`scipy.integrate.solve_ivp`).

        The solver solves for the state vector *y* (see `Equations` documentation) from
        the initial value `y0`. The raw solution from the solver is stored under the
        `solution_raw` instance attribute. If the model instance has already been run
        before (e.g. it's being run for the second time after tweaking the chemistry
        or plasma parameters...), then the `reset_equations` needs to be set to
        ``True``.

        Parameters
        ----------
        y0 : ndarray, optional
            The optional initial guess for the state vector (see the `Equations` docs).
            If not passed, it's built using the `Equations.get_y0_default` method.
        method : str, optional
            The optional solver method forwarded to the low-level solver (see
            `scipy.integrate.solve_ivp`). Defaults to ``"BDF"``.
        reset_equations : bool, optional
            `True` needs to be passed if the model has been run before with different
            chemistry or plasma parameters.
        """
        if reset_equations:
            self._reinitialize_equations()
        if y0 is None:
            y0 = self.equations.get_y0_default()
        func = self.equations.ode_system_rhs

        try:
            self.solution_raw = solve_ivp(
                func, (0, self.plasma_params.t_end), y0, method=method, t_eval=None
            )
        except ValueError as e:
            raise ModelSolutionError(f"solve_ivp raised a ValueError: {e}")

    def _build_solution(self):
        self.t = self.solution_raw.t
        self.solution_primary = self.solution_raw.y.T
        solution_labels = self.equations.final_solution_labels
        self.solution = DataFrame(columns=["t"] + solution_labels)
        for i, (t_i, y_i) in enumerate(zip(self.t, self.solution_primary)):
            self.solution.loc[i, "t"] = t_i
            self.solution.loc[
                i, solution_labels
            ] = self.equations.get_final_solution_values(y_i)

    def success(self):
        return bool(self.solution_raw.success)

    def run(
        self,
        initial_densities: Union[Mapping[str, float], pandas.Series] = None,
        reset_equations: bool = False,
    ):
        if initial_densities is not None:
            y0 = None
            raise NotImplementedError
        else:
            y0 = None
        self._solve(y0=y0, reset_equations=reset_equations)
        if not self.success():
            raise ModelSolutionError(self.solution_raw.message)

    def diagnose(self, quantity, primary_solution=None, totals=False):
        raise NotImplementedError
