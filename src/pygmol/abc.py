"""Module containing the abstract base classes (ABCs) which define the interfaces of
various objects expected by the ``pygmol`` framework.
"""
from abc import ABC, abstractmethod
from typing import Union, Sequence, Dict, Callable, Mapping

from numpy import ndarray, float64
from pyvalem.formula import FormulaParseError
from pyvalem.stateful_species import StatefulSpecies
from pyvalem.states import StateParseError
from scipy import constants


class Chemistry(ABC):
    # noinspection PyUnresolvedReferences
    """An abstract base class (ABC) defining the interface of objects describing the
    plasma chemistry, expected by other classes of the ``pygmol`` framework.

    A chemistry is a collection of species and reactions, and data attached to them.
    Species can be divided into heavy species (neutral and ions), and *special* species:
    electron and an arbitrary species 'M'.

    The heavy species data required by the model are encoded in the attributes or
    properties starting with ``species_``. The reactions kinetic data (and metadata) are
    encoded in the attributes/properties starting with ``reactions_``. Subset of these
    attributes starting with ``reactions_electron_``, ``reactions_arbitrary`` and
    ``reactions_species`` then handle the relationships between reactions and all the
    species (heavy and special) in the chemistry. See the attributes below.
    Documentation for the mandatory and optional attributes can be found in docstrings
    of the corresponding properties.

    Some of the attributes/properties listed here are implemented in this ABC as useful
    defaults, and therefore must not strictly be re-implemented by a concrete subclass,
    if inheriting from this abstraction (they are not @abstract properties). All of
    the attributes are needed by the `pygmol` package however, so a concrete chemistry
    class which *does not* inherit from this abstraction must implement all the
    attributes/properties below.

    Attributes
    ----------
    species_ids : Sequence[str]
    species_charges : Sequence[int]
        Default provided by the ABC.
    species_masses : Sequence[float]
        Default provided by the ABC.
    species_lj_sigma_coefficients : Sequence[float]
        Default provided by the ABC.
    species_surface_sticking_coefficients : Sequence[float]
        Default provided by the ABC.
    species_surface_return_matrix : Sequence[Sequence[float]]
    reactions_ids : Sequence[str] or Sequence[int]
    reactions_arrh_a : Sequence[float]
    reactions_arrh_b : Sequence[float]
    reactions_arrh_c : Sequence[float]
    reactions_el_energy_losses : Sequence[float]
    reactions_elastic_flags : Sequence[bool]
    reactions_electron_stoich_lhs : Sequence[int]
    reactions_electron_stoich_rhs : Sequence[int]
    reactions_arbitrary_stoich_lhs : Sequence[int]
    reactions_arbitrary_stoich_rhs : Sequence[int]
    reactions_species_stoichiomatrix_lhs : Sequence[Sequence[int]]
    reactions_species_stoichiomatrix_lhs : Sequence[Sequence[int]]
    """

    @property
    @abstractmethod
    def species_ids(self) -> Sequence[str]:
        """Unique ids/names of all the heavy species in the chemistry. This excludes
        electrons and the *arbitrary* species 'M'.

        If pyvalem-compatible formula strings are used as `species_id`, the `Equations`
        ABC will provide defaults for `species_charges` and `species_masses` attributes.
        See the pyvalem package on PyPI.
        """

    @property
    def species_charges(self) -> Sequence[int]:
        """Charges [e] of all the heavy species in the chemistry. This excludes
        electrons and the *arbitrary* species 'M'.

        By default, the species charges are parsed from `species_ids`, if they are in
        pyvalem-compatible format.
        """
        try:
            charges = [
                StatefulSpecies(sp_name).formula.charge for sp_name in self.species_ids
            ]
            return charges
        except (FormulaParseError, StateParseError):
            raise NotImplementedError(
                "Either `species_charges` attribute needs to be implemented, or "
                "the `species_ids` need to be pyvalem-compatible formulas!"
            )

    @property
    def species_masses(self) -> Sequence[float]:
        """Masses [amu] of all the heavy species in the chemistry. This excludes
        electrons and the *arbitrary* species 'M'.

        By default, the species masses are parsed from `species_ids`, if they are in
        pyvalem-compatible format.
        """
        try:
            masses = [
                StatefulSpecies(sp_name).formula.mass for sp_name in self.species_ids
            ]
            return masses
        except (FormulaParseError, StateParseError):
            raise NotImplementedError(
                "Either `species_masses` attribute needs to be implemented, or "
                "the `species_ids` need to be pyvalem-compatible formulas!"
            )

    @property
    def species_lj_sigma_coefficients(self) -> Sequence[float]:
        """Lennard-Jones sigma parameters [Angstrom] of all the heavy species in the
        chemistry. This excludes electrons and the *arbitrary* species 'M'.

        The `Chemistry` ABC provides a useful default where the Lennard-Jones
        coefficients are not available.
        """
        return [3.0 for _ in self.species_ids]

    @property
    def species_surface_sticking_coefficients(self) -> Sequence[float]:
        """Surface sticking coefficients of all the heavy species in the chemistry.
        This excludes electrons and the *arbitrary* species 'M'. The i-th element of
        the sequence denotes what fraction of the i-th species is lost when reaching the
        surface.

        Default is provided by this ABC: All charged species have by default sticking
        coefficient 1.0, while all the neutral species have by default sticking
        coefficient 0.0.
        """
        return [float(bool(sp_charge)) for sp_charge in self.species_charges]

    @property
    @abstractmethod
    def species_surface_return_matrix(self) -> Sequence[Sequence[float]]:
        """A 2D array-like of shape (Ns, Ns), where [i, j] index (i-th row and j-th
        column) denotes the number of i-th species created by each one j-th species
        *STUCK* to the surface. Non-zero values of R[:, j] therefore only make sense if
        ``chemistry.species_sticking_coefficients[j] > 0``.

        Ns refers to the number of heavy species in the chemistry and needs to be
        consistent with `species_ids` property/attribute.
        """

    @property
    @abstractmethod
    def reactions_ids(self) -> Sequence[int]:
        """Unique IDs of all the reactions in the chemistry."""

    @property
    def reactions_strings(self) -> Sequence[str]:
        """Optional human-readable reaction strings for all the reactions in the
        chemistry.

        This is used only for annotating the model solutions.
        """
        return [f"Reaction {r_id}" for r_id in self.reactions_ids]

    @property
    @abstractmethod
    def reactions_arrh_a(self) -> Sequence[float]:
        """First Arrhenius parameters (A, or alpha) for all the reactions in the
        chemistry. The `arrh_a` values are in SI [m3.s-1 / m6.s-1, s-1].
        """

    @property
    @abstractmethod
    def reactions_arrh_b(self) -> Sequence[float]:
        """Second Arrhenius parameters (n, or beta) for all the reactions in the
        chemistry. The `arrh_b` values are unitless.
        """

    @property
    @abstractmethod
    def reactions_arrh_c(self) -> Sequence[float]:
        """Third Arrhenius parameters (E_a, or gamma) for all the reactions in the
        chemistry. The `arrh_c` values are in [eV] for electron collisions and in [K]
        for heavy-species collisions.
        """

    @property
    @abstractmethod
    def reactions_el_energy_losses(self) -> Sequence[float]:
        """Electron energy loss [eV] for all the reactions in the chemistry. Should be
        non-zero only for inelastic electron collisions, estimating average energy loss
        for the electron in each collision.
        """

    @property
    @abstractmethod
    def reactions_elastic_flags(self) -> Sequence[bool]:
        """Boolean flags for all the reactions in the chemistry, evaluating to True for
        elastic collisions only.
        """

    @property
    @abstractmethod
    def reactions_electron_stoich_lhs(self) -> Sequence[int]:
        """Number of electrons on left-hand-side of each reaction in the chemistry."""

    @property
    @abstractmethod
    def reactions_electron_stoich_rhs(self) -> Sequence[int]:
        """Number of electrons on right-hand-side of each reaction in the chemistry."""

    @property
    @abstractmethod
    def reactions_arbitrary_stoich_lhs(self) -> Sequence[int]:
        """Number of arbitrary species 'M' on left-hand-side of each reaction in the
        chemistry.
        """

    @property
    @abstractmethod
    def reactions_arbitrary_stoich_rhs(self) -> Sequence[int]:
        """Number of arbitrary species 'M' on right-hand-side of each reaction in the
        chemistry.
        """

    @property
    @abstractmethod
    def reactions_species_stoichiomatrix_lhs(self) -> Sequence[Sequence[int]]:
        """A 2D array-like of shape (Nr, Ns), where [i, j] index (i-th row and j-th
        column) points to the number of j-th heavy species on the left-hand-side of the
        i-th reaction.

        Nr and Ns refer to the number of reactions and heavy species respectively and
        need to be consistent with `reactions_ids` and `species_ids`
        properties/attributes.
        """

    @property
    @abstractmethod
    def reactions_species_stoichiomatrix_rhs(self) -> Sequence[Sequence[int]]:
        """A 2D array-like of shape (Nr, Ns), where [i, j] index (i-th row and j-th
        column) points to the number of j-th heavy species on the right-hand-side of the
        i-th reaction.

        Nr and Ns refer to the number of reactions and heavy species respectively and
        need to be consistent with `reactions_ids` and `species_ids`
        properties/attributes.
        """


class PlasmaParameters(ABC):
    # noinspection PyUnresolvedReferences
    """Data class of plasma parameters needed as an input to the ``pygmol`` framework.

    See the attributes below. Documentation for the mandatory and optional attributes
    can be found in docstrings of the corresponding properties.

    Time-dependent power is encoded by two same-length sequences: `power` and `t_power'.
    An example of a single 500W pulse for half of the simulation time of 0.2 seconds
    might look like ``power = [500, 500, 0, 0]`` with ``t_power = [0.0, 0.1, 0.1, 0.2]``

    Attributes
    ----------
    radius, length : float
    pressure : float
    power : float or Sequence[float]
    t_power : Sequence[float], optional
    feeds : dict[str, float], optional, default={}
    temp_e : float, optional, default=1.0 [eV]
    temp_n : float, optional, default=300.0 [K]
    t_end : float, optional, default=1.0 [s]
    """

    @property
    @abstractmethod
    def radius(self) -> float:
        """The radius dimension of the cylindrical plasma in [m]."""

    @property
    @abstractmethod
    def length(self) -> float:
        """The length dimension of the cylindrical plasma in [m]."""

    @property
    @abstractmethod
    def pressure(self) -> float:
        """Plasma pressure set-point in [Pa]."""

    @property
    @abstractmethod
    def power(self) -> Union[float, Sequence[float]]:
        """Power deposited into plasma in [W]. Either a single number, or a sequence of
        numbers for time-dependent power.
        """

    @property
    def t_power(self) -> Union[None, Sequence[float]]:
        """If `power` passed is a sequence, the `t_power` needs to be passed, defining
        the time-points for the `power` values and having the same length. Defaults to
        None, which is fine for time-independent `power`.
        """
        return None

    @property
    def feeds(self) -> Dict[str, float]:
        """Dictionary of feed flows in [sccm] for all the species, which are fed to
        plasma. The feed flows are keyed by species IDs (distinct names/formulas/... -
        the keys need to be subset of the values returned by `Chemistry.species_ids`).
        Defaults to the empty dict.
        """
        return {}

    @property
    def temp_e(self) -> float:
        """Electron temperature of the plasma, or, for the model which do resolve the
        electron temperature, its initial value. Defaults to 1.0 eV.
        """
        return 1.0

    @property
    def temp_n(self) -> float:
        """Neutral temperature of the plasma, or, for the model which do resolve the
        temperature, its initial value. Defaults to 300.0 K.
        """
        return 300.0

    @property
    def t_end(self) -> float:
        """End time of the simulation in [s]. Defaults to 1.0 sec."""
        return 1.0


class Equations(ABC):
    # noinspection PyUnresolvedReferences
    """Abstract base class (ABC) representing the system of ODEs.

    The `GlobalModel` (as the highest-level class of the `pygmol` package) solves an
    initial value problem for a system of ordinary differential equations. The system
    being solved (in general) is::

        dy / dt = f(t, y),

    given an initial value::

        y(t0) = y0.

    Here, t is an independent dime variable, and y(t) is an N-D vector-valued function
    (describing the state as a function of time). The N-D vector-valued function f(t, y)
    describes the right-hand side (RHS) of the ODE system and determines the system.

    The concrete `Equations` subclasses need to implement several attributes or
    properties and methods:
    - `ode_system_rhs` attribute/property is expected to give the function *f(t, y)*,
      or the RHS of the ODE system. This function signature must be ``f(t, y)``, where
      `t` is a scalar time and `y` is an N-D state vector of floats.
    - `final_solution_labels` attribute/property (see the abstract property docstring)
    - `get_final_solution_values` method (see the abstract method docstring)
    - `get_y0_default` method (see the abstract property docstring)

    It is advantageous to implement the `ode_system_rhs` as a combination of a series
    of intermediate results implemented as getter methods
    ``get_{quantity}(y: ndarray) -> ndarray | number`` accepting the state vector
    as the only mandatory argument and returning either a vector or a scalar value.
    That way, the intermediate results can be evaluated from the solution returned by
    the solver for every time sample by the `model.Model` class downstream after the
    model is run/solved. See `model.Model.diagnose` method docs. As an example,
    stubs for two intermediate results are prepared in this ABC: `get_reaction_rates(y)`
    and `get_wall_fluxes(y)`. These need to be re-implemented in the concrete subclasses
    to take advantage of all the functionality of the `model.Model` class.

    Attributes
    ----------
    ode_system_rhs : Callable[[float, ndarray], ndarray]
    final_solution_labels : Sequence[str]

    Methods
    -------
    get_final_solution_values
        Accepts the state vector `y` and returns the values of the final solution
        consistent with the `final_solution_labels` attribute.
    get_y0_default
        Builds a consistent initial guess for the state vector `y`, based on chemistry
        and plasma parameters.
    """

    # some constants:
    mtorr = 0.133322  # 1 mTorr in Pa
    sccm = 4.485e17  # 1 sccm in particles/s
    pi = constants.pi
    m_e = constants.m_e
    k = constants.k
    e = constants.e
    epsilon_0 = constants.epsilon_0
    atomic_mass = constants.atomic_mass

    @abstractmethod
    def __init__(self, chemistry: Chemistry, plasma_params: PlasmaParameters):
        """The initializer for the equations class accepting the instances of
        `Chemistry` concrete subclass and `PlasmaParameters` concrete subclass
        """
        self.chemistry = chemistry
        self.plasma_params = plasma_params

    def get_reaction_rates(self, y: ndarray) -> ndarray:
        """Calculates the vector of reaction rates [SI] for all the reactions in the
        chemistry (`self.chemistry`).

        Parameters
        ----------
        y : ndarray
            State vector *y*.

        Returns
        -------
        ndarray
            Reaction rates [m-3/s] for all the reactions in the chemistry set.
        """
        raise NotImplementedError

    def get_surface_loss_rates(self, y: ndarray) -> ndarray:
        """Calculates the vector of wall-loss rates [m-3/s] for all the heavy species in
        the chemistry (`self.chemistry`).

        This *only* covers the species losses due to diffusion *and* sticking to
        surfaces, but does not cover species conversion on the surfaces via the return
        coefficients.

        Parameters
        ----------
        y : ndarray
            State vector *y*.

        Returns
        -------
        ndarray
            The vector of contributions to time derivatives of densities due to the
            diffusion losses for all the heavy species in [m-3/s]. Does not include
            the return species, only the *sticking losses*.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def ode_system_rhs(self) -> Callable[[float, ndarray], ndarray]:
        """A callable returning the time derivative of the state (unknowns being solved
        for) as a function of time and the current state.

        Apart from ``t`` and ``y``, the function dy / dt will also depend on other
        *things* (such as the chemistry set and the plasma conditions). The appropriate
        objects modeling those will normally be passed to the concrete `Equations`
        subclass and this dependence will be coded into the concrete ``ode_system_rhs``
        property.
        """

    @property
    @abstractmethod
    def final_solution_labels(self) -> Sequence[str]:
        """The sequence of `str` labels identifying the final solution expected to be
        built downstream in the `Model`.

        As an example, if the state vector `y` is an array such as
        ``["n_1", ..., "n_N", "rho_e"]`` with densities of the heavy specie and the
        electron energy density, the `final_solution_labels` might be, for example,
        ["n_1", ..., "n_N", "n_e", "T_e"], where the electron density `n_e` is built
        downstream from the heavy species densities (utilizing charge conservation),
        and the electron temperature `T_e` is built downstream from the electron density
        and electron energy density `rho_e`.

        The `final_solution_labels` need to be consistent with the
        `get_final_solution_values` method, which turns the raw
        values of the state vector `y` to the values of the final solution.
        """

    @abstractmethod
    def get_final_solution_values(self, t: float64, y: ndarray) -> ndarray:
        """A method which turns the time [s] and the raw state vector `y` into the
        final solution values consistent with the `final_solution_labels`.

        The method must accept time `t` [sec] and the ndarray `y` (of the dimension of
        the ODE system being solved) and return ndarray of the same length as
        `final_solution_labels`.

        Parameters
        ----------
        t : float64
            Time (within the simulation) in [s].
        y : ndarray
            The instantaneous state vector `y` (dimension as is the dimension of the
            IDE system being solved).

        Returns
        -------
        ndarray
            Array of the final solution values. These are the values of the *top-level*
            solution (such as n_Ar, n_e, T_e, p, ...) constructed from the *lower-level*
            solution, which is the state vector `y`.
        """

    @abstractmethod
    def get_y0_default(self, initial_densities: Mapping[str, float] = None) -> ndarray:
        """A default initial guess of the state vector `y`.

        The values of `y0_default` are built to be consistent with the passed
        `chemistry` and `plasma_parameters` which should be saved as instance
        attributes. For example, the initial guess for `y` should be consistent with
        plasma parameters, such as pressure, electron temperature, neutral temperature,
        etc.

        Parameters
        ----------
        initial_densities : Mapping[str, float], optional
            Mapping between the species ids and the initial densities (or their
            fractions). If not supplied, the densities in the state vector *y* should be
            initialized with some sensible default fractions.

        Returns
        -------
        ndarray
            A consistent initial guess for the state vector `y`.
        """
