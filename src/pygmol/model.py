from typing import Union, Mapping

from pandas import DataFrame
import pandas
from numpy import ndarray
import numpy as np
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
    """The Global Model class.

    Takes instances of `Chemistry` (or dict following the correct interface), and
    `PlasmaParameters` (or a dict following the correct interface), and instantiates
    a concrete `Equations` subclass.

    The model with consistent chemistry and plasma parameters inputs can be run with
    the `run` method, and success checked with the `success` method. Other methods are
    implemented to access the full final solution, reaction rates or wall fluxes, all
    either as function of time, or the final values. Finally, the `diagnose` method
    is provided to extract any partial results defined in the `Equations` subclass
    (see the docstring.)
    """

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
            Signals inconsistent plasma parameters passed.
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

        # placeholder for the equations employed by the model:
        self.equations = None
        # placeholder for whatever the selected low-level solver returns:
        self.solution_raw = None
        # placeholder for the array of time samples [sec]:
        self.t = None
        # 2D array of state vectors `y` for all time samples:
        self.solution_primary = None
        # `pandas.DataFrame` of all the final solution values:
        self.solution = None

    def _initialize_equations(self):
        self.equations = ElectronEnergyEquations(self.chemistry, self.plasma_params)

    def _solve(self, y0: ndarray = None, method: str = "BDF"):
        """Runs the low-level solver (`scipy.integrate.solve_ivp`).

        The solver solves for the state vector *y* (see `Equations` documentation) from
        the initial value `y0`. The raw solution from the solver is stored under the
        `solution_raw` instance attribute. The equations must have been initialized
        already!

        Parameters
        ----------
        y0 : ndarray, optional
            The optional initial guess for the state vector (see the `Equations` docs).
            If not passed, it's built using the `Equations.get_y0_default` method.
        method : str, optional
            The optional solver method forwarded to the low-level solver (see
            `scipy.integrate.solve_ivp`). Defaults to ``"BDF"``.

        Raises
        ------
        ModelSolutionError
            If the `solve_ivp` solver encounters an error, or if it is in any way
            unsuccessful.
        """
        if self.equations is None:
            raise ModelSolutionError("The equations have not yet been initialized!")

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
        """Populates the `solution` instance attribute.

        The `solution_raw` attribute must have already been populated by the `solve`
        method! This method (`build_solution`) will take the raw rows of state vectors
        *y* in time (see `Equations` docs) and turn them into the final solution values
        by the appropriate methods supplied by `Equations` class.

        The final solution will be saved as the `solution` instance attribute and will
        take form of a pandas.DataFrame with columns consisting of ``"t"`` (the first
        column of sampled time in [s]), followed by the
        `equations.final_solution_labels`.

        Raises
        ------
        ModelSolutionError
            If the `solution_raw` is None (not populated yet).
        """
        if self.solution_raw is None:
            raise ModelSolutionError("The solver has not yet been run!")

        self.t = self.solution_raw.t
        self.solution_primary = self.solution_raw.y.T
        solution_labels = self.equations.final_solution_labels
        self.solution = DataFrame(columns=["t"] + solution_labels)
        for i, (t_i, y_i) in enumerate(zip(self.t, self.solution_primary)):
            self.solution.loc[i, "t"] = t_i
            self.solution.loc[
                i, solution_labels
            ] = self.equations.get_final_solution_values(y_i)

    def run(
        self,
        initial_densities: Union[Mapping[str, float], pandas.Series] = None,
    ):
        """Runs the solver on the `Equations` instance (`equations` attribute), and
        builds the final solution out of the solver output.

        Result is the `solution` attribute populated with pandas.DataFrame with
        columns consisting of ``"t"`` (the first column of sampled time in [s]),
        followed by the `equations.final_solution_labels` columns.

        Parameters
        ----------
        initial_densities : dict[str, float] or pandas.Series, optional
            Optional mapping between species ids (consistent with the `chemistry` passed
            to the constructor), and their initial densities, as the initial values
            for the solver. The densities are re-normalized to the total pressure
            downstream, so relative fractions are sufficient.
            If not passed, default is provided by `equations`.
        """
        if initial_densities is not None:
            initial_densities = dict(initial_densities)
            # some consistency checks are needed
            if not set(initial_densities).issubset(self.chemistry.species_ids):
                raise ModelSolutionError(
                    "Initial densities inconsistent with the chemistry!"
                )
            # is there a room for any electrons?
            n0 = np.array(
                initial_densities.get(sp_id, 0.0)
                for sp_id in self.chemistry.species_ids
            )
            if sum(n0 * self.chemistry.species_charges) < 0:
                raise ModelSolutionError(
                    "Total initial charge density is negative! No room for electrons!"
                )

        y0 = self.equations.get_y0_default(initial_densities=initial_densities)
        self._initialize_equations()
        self._solve(y0=y0)
        if not self.success():
            raise ModelSolutionError(self.solution_raw.message)
        self._build_solution()

    def success(self) -> bool:
        """Method checking for success of the solution.

        Returns
        -------
        bool
            True, if the solution was successful.
        """
        return bool(self.solution_raw.success)

    def diagnose(
        self, quantity: str, primary_solution: ndarray = None, totals: bool = False
    ) -> DataFrame:
        """"""
        raise NotImplementedError
