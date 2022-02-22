"""A module providing the `Model` class representing the global model and tying together
all the other classes defined in the `pygmol` package (concrete subclasses of
`Chemistry`, `PlasmaParameters` and `Equations`.)
"""
from typing import Union, Mapping

import numpy as np
import pandas
from numpy import ndarray
from scipy.integrate import solve_ivp

from .abc import Chemistry, PlasmaParameters
from .chemistry import chemistry_from_dict, validate_chemistry
from .equations import ElectronEnergyEquations
from .plasma_parameters import (
    plasma_parameters_from_dict,
    validate_plasma_parameters,
    PlasmaParametersValidationError,
)


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

        Parameters
        ----------
        chemistry : Chemistry
        plasma_params : PlasmaParameters

        Raises
        ------
        ChemistryValidationError
            Signals inconsistent chemistry passed.
        PlasmaParametersValidationError
            Signals inconsistent plasma parameters passed.
        """
        if isinstance(chemistry, dict):
            chemistry = chemistry_from_dict(chemistry)
        if isinstance(plasma_params, dict):
            plasma_params = plasma_parameters_from_dict(plasma_params)

        self.chemistry = chemistry
        self.plasma_params = plasma_params
        self._validate_chemistry_and_plasma_parameters()

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

    def _validate_chemistry_and_plasma_parameters(self):
        """Method running the validation on both chemistry and plasma parameters,
        and checking if they are both consistent with each other.

        Raises
        ------
        ChemistryValidationError
        PlasmaParametersValidationError
        """
        validate_chemistry(self.chemistry)
        validate_plasma_parameters(self.plasma_params)
        if not set(self.plasma_params.feeds).issubset(self.chemistry.species_ids):
            raise PlasmaParametersValidationError(
                "Feed gas species defined in the plasma parameters are inconsistent "
                "with the chemistry species ids!"
            )

    def _initialize_equations(self):
        """Populates the equations instance attribute."""
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

        TODO: This is way too slow, should take around 0.01 s instead of 0.44 s for
              the test case!
        """
        if self.solution_raw is None:
            raise ModelSolutionError("The solver has not yet been run!")

        self.t = self.solution_raw.t
        self.solution_primary = self.solution_raw.y.T
        solution_labels = self.equations.final_solution_labels
        self.solution = pandas.DataFrame(columns=["t"] + solution_labels, dtype=float)
        for i, (t_i, y_i) in enumerate(zip(self.t, self.solution_primary)):
            self.solution.loc[i, "t"] = t_i
            self.solution.loc[
                i, solution_labels
            ] = self.equations.get_final_solution_values(t_i, y_i)

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

        Raises
        ------
        ModelSolutionError
            If there is an error raised by the solver.
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
                [
                    initial_densities.get(sp_id, 0.0)
                    for sp_id in self.chemistry.species_ids
                ]
            )
            if sum(n0 * np.array(self.chemistry.species_charges)) < 0:
                raise ModelSolutionError(
                    "Total initial charge density is negative! No room for electrons!"
                )

        self._initialize_equations()
        y0 = self.equations.get_y0_default(initial_densities=initial_densities)
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

    def diagnose(self, quantity: str, totals: bool = False) -> pandas.DataFrame:
        """Fetch diagnostics of any of the `equations`' partial results for all the time
        samples from the finished primary solution.

        This expects the corresponding `get_{quantity}` getter method existing in the
        `equations` instance and accepting the state vector *y* as the single mandatory
        argument, and returning either a 1D array or a 0D scalar. See the `Equations`
        docs for more documentation.

        Creates a dataframe with the quantities returned by the `equations`' getter
        method evaluated for all the time samples. As an example, if the `equations`
        instance has `get_reaction_rates(y: ndarray) -> ndarray` getter method, which
        returns reaction rates for N reactions,
        then ``diagnose(quantity="reaction_rates")`` will return a `DataFrame` with
        columns ``["t", "col1", "col2", ..., "colN"]``, where values of the first column
        are time samples in [sec] and values of the other columns are what the
        `equations.get_reaction_rates` method returns for each time sample.
        Optionally, the ``totals=True`` flag might be passed, in which case an
        additional column ``"total"`` is appended to the `DataFrame` summing all the
        other columns.

        If the corresponding `get_{quantity}` getter method returns a scalar, the column
        in the resulting dataframe (next to the "t" column) is labeled as the
        `{quantity}` passed, rather than ``"col1"``.

        Parameters
        ----------
        quantity : str
            Needs to correspond to a getter method implemented by the `Equations`
            concrete subclass stored as the `equations` attribute.
        totals : bool
            If True, appends an extra ``"total"`` column summing all the other column
            (apart from time ``"t"``).

        Returns
        -------
        DataFrame
            Columns are:
                - either "t", "col1", ...[, "total"] for equations.get_{quantity}
                  returning a 1D vector,
                - or "t", "{quantity}" [, "total"] for equations.get_{quantity}
                  returning a scalar. In this case the `totals` flag does not make
                  any sense.
            Index of the data frame is arbitrary and irrelevant.
        """
        if self.solution_primary is None:
            raise ModelSolutionError("The solver has not yet been run!")
        equations_method = f"get_{quantity}"
        if not hasattr(self.equations, equations_method):
            raise ModelSolutionError(
                f"The {type(self.equations).__name__} object does not have the "
                f"'{equations_method}' method!"
            )
        diagnostics = np.array(
            [
                getattr(self.equations, equations_method)(y).copy()
                for y in self.solution_primary
            ]
        )
        # if equations.get_{quantity} returns vector, diagnostics is a 2D array.
        # if it returns a scalar, diagnostics needs to be turned into a 2D array:
        if len(diagnostics.shape) < 2:
            diagnostics = diagnostics[np.newaxis].T
            labels = [quantity]
        else:
            labels = [f"col{i}" for i in range(1, diagnostics.shape[1] + 1)]

        df = pandas.DataFrame(np.c_[self.t, diagnostics], columns=["t"] + labels)
        if totals:
            df["total"] = diagnostics.sum(axis=1)
        return df

    def get_solution(self) -> pandas.DataFrame:
        """Method returning the solution of the model in time.

        The `run` method must have been called before this one.

        Returns
        -------
        pandas.DataFrame
            The first column is ``"t"`` for time samples, other columns are determined
            by the `equations` attribute, more specifically by the
            `Equations.final_solution_labels` attribute.
            Each row is for a single time sample.

        Raises
        ------
        ModelSolutionError
            If called before the `run` has been is called.
        """
        if self.solution is None:
            raise ModelSolutionError("The model has not yet been run!")
        return self.solution.copy()

    def get_solution_final(self) -> pandas.Series:
        """Method returning the final row of the solution of the model, corresponding
        to the final time sample.

        The `run` method must have been called before this one.

        Returns
        -------
        pandas.Series
            The first index is ``"t"`` for time sample, other indices are determined
            by the `equations` attribute, more specifically by the
            `Equations.final_solution_labels` attribute.

        Raises
        ------
        ModelSolutionError
            If called before the `run` has been is called.
        """
        return self.get_solution().iloc[-1]

    def get_reaction_rates(self) -> pandas.DataFrame:
        """Method returning the reaction rates in time for the solution of the model.

        The `run` method must have been called before this one.

        Returns
        -------
        pandas.DataFrame
            The first column is ``"t"`` for time samples, the other columns are integer
            reaction ids (see `Chemistry.reactions_ids`). Each row is for a single time
            sample.
        """
        rates = self.diagnose("reaction_rates")
        columns = ["t"]
        columns.extend(self.chemistry.reactions_ids)
        rates.columns = columns
        return rates

    def get_reaction_rates_final(self) -> pandas.Series:
        """Method returning the final reaction rates at the end of the model solution.

        The `run` method must have been called before this one.

        Returns
        -------
        pandas.Series
            The first column is ``"t"`` for the time sample, the other columns are
            integer reaction ids (see `Chemistry.reactions_ids`).
        """
        return self.get_reaction_rates().iloc[-1]

    def get_surface_loss_rates(self) -> pandas.DataFrame:
        """Method returning the surface loss rates [m-3/s] in time for all the heavy
        species in the chemistry.

        The `run` method must have been called before this one.

        Returns
        -------
        pandas.DataFrame
            The first column is ``"t"`` for time samples, the other columns are strings
            of the species ids (see `Chemistry.species_ids`).
            Each row is for a single time sample.
        """
        surf_loss_rates = self.diagnose("surface_loss_rates")
        surf_loss_rates.columns = ["t"] + [
            str(sp_id) for sp_id in self.chemistry.species_ids
        ]
        return surf_loss_rates

    def get_surface_loss_rates_final(self) -> pandas.Series:
        """Method returning the final surface loss rates [m-3/s] at the end of the model
        solution.

        The `run` method must have been called before this one.

        Returns
        -------
        pandas.Series
            The first column is ``"t"`` for the time sample, the others are strings
            of the species ids (see `Chemistry.reactions_ids`).
        """
        return self.get_surface_loss_rates().iloc[-1]

    def get_volumetric_rates_matrix(
        self, t: float = None, annotate: bool = True
    ) -> pandas.DataFrame:
        """Method returning a data frame of all the *volumetric rates of change* of
        each and every *heavy species* due to each and every reaction.

        Each row correspond to a single reaction in the chemistry set and each column
        correspond to a single heavy species in the chemistry set. All the values are
        the volumetric rates of change (positive for volumetric sources / production or
        negative for volumetric sinks / consumption) of the densities of all the
        species. All the values are in [m-3/s] (assuming the
        `equations.get_reaction_rates` method returns results in the same units, as
        it should.)

        The rows are indexed by reaction ids (see `Chemistry.reactions_ids` method of
        the chemistry abstract base class), or, if ``annotate=True`` flag passed, by the
        reaction strings (see `Chemistry.reactions_strings` method of the chemistry
        ABC.)

        The columns are indexed by the species ids (see `Chemistry.species_ids` method
        of the chemistry abstract base class).

        Parameters
        ----------
        t : float, optional
            Time in [sec]. The closest existing time sample from the existing solution
            will be selected, but no interpolation will be performed.
            If not passed, the final time frame is selected.
        annotate : bool, optional, defaults to True
            If True passed, the resulting dataframe will be indexed by the reaction
            strings, rather than reaction ids. The `Chemistry` instance must have the
            `Chemistry.reactions_strings` attribute though.

        Returns
        -------
        pandas.DataFrame
        """
        if self.solution_primary is None:
            raise ModelSolutionError("The solver has not yet been run!")

        stoichiomatrix = np.array(
            self.chemistry.reactions_species_stoichiomatrix_rhs
        ) - np.array(self.chemistry.reactions_species_stoichiomatrix_lhs)

        if t is None:
            rates_frame = self.get_reaction_rates_final()
        else:
            rates = self.get_reaction_rates()
            t_index = abs(rates["t"] - t).idxmin()
            rates_frame = rates.loc[t_index]
        # get rid of the time in rates:
        rates_frame = rates_frame.iloc[1:]

        vol_rates = pandas.DataFrame(
            stoichiomatrix * rates_frame.values[:, np.newaxis],
            index=self.chemistry.reactions_ids,
            columns=self.chemistry.species_ids,
        )
        if annotate:
            vol_rates.index = self.chemistry.reactions_strings

        return vol_rates

    def get_surface_rates_matrix(
        self, t: float = None, annotate: bool = True
    ) -> pandas.DataFrame:
        if self.solution_primary is None:
            raise ModelSolutionError("The solver has not yet been run!")

        return_matrix = np.array(self.chemistry.species_surface_return_matrix)

        if t is None:
            loss_rates_frame = self.get_surface_loss_rates_final()
        else:
            loss_rates = self.get_surface_loss_rates()
            t_index = abs(loss_rates["t"] - t).idxmin()
            loss_rates_frame = loss_rates.loc[t_index]
        loss_rates_frame = loss_rates_frame.iloc[1:]  # remove the time.
        # Loss rates matrix: loc[A, B] is loss rate (< 0) of A due to B sticking to
        # surface. Ny definition, it is a diagonal matrix, A is lost only if A is stuck
        # to surface
        loss_rates_matrix = pandas.DataFrame(
            np.diag(loss_rates_frame),
            columns=self.chemistry.species_ids,
            index=self.chemistry.species_ids,
        )
        # source rates matrix: loc[A, B] is a source rate (> 0) of A due to B sticking
        # to surface and getting returned as A. This will typically not have any
        # diagonal elements (those don't make sense, but are not prohibited).
        source_rate_matrix = pandas.DataFrame(
            return_matrix * -loss_rates_frame.values[np.newaxis, :],
            columns=self.chemistry.species_ids,
            index=self.chemistry.species_ids,
        )
        surface_rates_matrix = (loss_rates_matrix + source_rate_matrix).T
        if annotate:
            # get rid of the zero rows - rows of species which do not get stuck to surf.
            surface_rates_matrix = surface_rates_matrix.loc[
                (surface_rates_matrix != 0).any(axis=1)
            ]
            # swap the species ids as index with the actual surface reaction:
            index = []
            for sp_id in surface_rates_matrix.index:
                raise NotImplementedError  # TODO: Implement this!

        return surface_rates_matrix
