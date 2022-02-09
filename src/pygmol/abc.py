"""Module containing the abstract base classes (ABCs) which define the
interfaces of various objects expected by the ``pygmol`` framework.
"""
from abc import ABC, abstractmethod
from typing import Union, Sequence, Dict, Callable

from numpy import ndarray


class Chemistry(ABC):
    # noinspection PyUnresolvedReferences
    """An abstract base class (ABC) defining the interface of objects
    describing the plasma chemistry, expected by other classes of the
    ``pygmol`` framework.

    A chemistry is a collection of species and reactions, and data
    attached to them. Species can be divided into heavy species (neutral
    and ions), and *special* species: electron and an arbitrary species
    'M'.

    The heavy species data required by the model are encoded in the
    attributes/properties starting with ``species_``. The reactions
    kinetic data (and metadata) are encoded in the attributes/properties
    starting with ``reactions_``. Subset of these attributes starting
    with ``reactions_electron_``, ``reactions_arbitrary`` and
    ``reactions_species`` then handle the relationships between
    reactions and all the species (heavy and special) in the chemistry.
    See the attributes below. Documentation for the mandatory and
    optional attributes can be found in docstrings of the corresponding
    properties.

    Attributes
    ----------
    species_ids : Sequence[str]
    species_charges : Sequence[int]
    species_masses : Sequence[float]
    species_lj_sigma_coefficients : Sequence[float]
    species_surface_sticking_coefficients : Sequence[float]
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
        """Unique ids/names of all the heavy species in the chemistry.
        This excludes electrons and the *arbitrary* species 'M'.
        """

    @property
    @abstractmethod
    def species_charges(self) -> Sequence[int]:
        """Charges [e] of all the heavy species in the chemistry.
        This excludes electrons and the *arbitrary* species 'M'.
        """

    @property
    @abstractmethod
    def species_masses(self) -> Sequence[float]:
        """Masses [amu] of all the heavy species in the chemistry.
        This excludes electrons and the *arbitrary* species 'M'.
        """

    @property
    @abstractmethod
    def species_lj_sigma_coefficients(self) -> Sequence[float]:
        """Lennard-Jones sigma parameters [Angstrom] of all the heavy
        species in the chemistry. This excludes electrons and the
        *arbitrary* species 'M'.
        """

    @property
    @abstractmethod
    def species_surface_sticking_coefficients(self) -> Sequence[float]:
        """Surface sticking coefficients of all the heavy species
        in the chemistry. This excludes electrons and the *arbitrary*
        species 'M'.
        The i-th element of the sequence denotes what fraction of the
        i-th species is lost when reaching the surface.
        """

    @property
    @abstractmethod
    def species_surface_return_matrix(self) -> Sequence[Sequence[float]]:
        """A 2D array-like of shape (Ns, Ns), where [i, j] index (i-th
        row and j-th column) denotes the number of i-th species created
        by each one j-th species *STUCK* to the surface. Non-zero values
        of R[:, j] therefore only make sense if
        ``chemistry.species_sticking_coefficients[j] > 0``.

        Ns refers to the number of heavy species in the chemistry and
        needs to be consistent with `species_ids` property/attribute.
        """

    @property
    @abstractmethod
    def reactions_ids(self) -> Union[Sequence[str], Sequence[int]]:
        """Unique IDs of all the reactions in the chemistry."""

    @property
    @abstractmethod
    def reactions_arrh_a(self) -> Sequence[float]:
        """First Arrhenius parameters (A, or alpha) for all the
        reactions in the chemistry. The `arrh_a` values are in
        SI [m3.s-1 / m6.s-1, s-1].
        """

    @property
    @abstractmethod
    def reactions_arrh_b(self) -> Sequence[float]:
        """Second Arrhenius parameters (n, or beta) for all the
        reactions in the chemistry. The `arrh_b` values are unitless.
        """

    @property
    @abstractmethod
    def reactions_arrh_c(self) -> Sequence[float]:
        """Third Arrhenius parameters (E_a, or gamma) for all the
        reactions in the chemistry. The `arrh_c` values are in [eV] for
        electron collisions and in [K] for heavy-species collisions.
        """

    @property
    @abstractmethod
    def reactions_el_energy_losses(self) -> Sequence[float]:
        """Electron energy loss [eV] for all the reactions in the
        chemistry. Should be non-zero only for inelastic electron
        collisions, estimating average energy loss for the electron in
        each collision.
        """

    @property
    @abstractmethod
    def reactions_elastic_flags(self) -> Sequence[bool]:
        """Boolean flags for all the reactions in the chemistry,
        evaluating to True for elastic collisions only.
        """

    @property
    @abstractmethod
    def reactions_electron_stoich_lhs(self) -> Sequence[int]:
        """Number of electrons on left-hand-side of each reaction in
        the chemistry.
        """

    @property
    @abstractmethod
    def reactions_electron_stoich_rhs(self) -> Sequence[int]:
        """Number of electrons on right-hand-side of each reaction in
        the chemistry.
        """

    @property
    @abstractmethod
    def reactions_arbitrary_stoich_lhs(self) -> Sequence[int]:
        """Number of arbitrary species 'M' on left-hand-side of each
        reaction in the chemistry.
        """

    @property
    @abstractmethod
    def reactions_arbitrary_stoich_rhs(self) -> Sequence[int]:
        """Number of arbitrary species 'M' on right-hand-side of each
        reaction in the chemistry.
        """

    @property
    @abstractmethod
    def reactions_species_stoichiomatrix_lhs(self) -> Sequence[Sequence[int]]:
        """A 2D array-like of shape (Nr, Ns), where [i, j] index (i-th
        row and j-th column) points to the number of j-th heavy species
        on the left-hand-side of the i-th reaction.

        Nr and Ns refer to the number of reactions and heavy species
        respectively and need to be consistent with
        `reactions_ids` and `species_ids` properties/attributes.
        """

    @property
    @abstractmethod
    def reactions_species_stoichiomatrix_rhs(self) -> Sequence[Sequence[int]]:
        """A 2D array-like of shape (Nr, Ns), where [i, j] index (i-th
        row and j-th column) points to the number of j-th heavy species
        on the right-hand-side of the i-th reaction.

        Nr and Ns refer to the number of reactions and heavy species
        respectively and need to be consistent with
        `reactions_ids` and `species_ids` properties/attributes.
        """


class PlasmaParameters(ABC):
    # noinspection PyUnresolvedReferences
    """Data class of plasma parameters needed as an input to the
    ``pygmol`` framework.

    See the attributes below. Documentation for the mandatory and
    optional attributes can be found in docstrings of the corresponding
    properties.

    Attributes
    ----------
    radius, length : float
    pressure : float
    power : float or Sequence[float]
    t_power : Sequence[float], optional
    feeds : dict[str, float]
    t_end : float
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
        """Power deposited into plasma in [W]. Either a single number,
        or a sequence of numbers for time-dependent power.
        """

    @property
    def t_power(self) -> Union[None, Sequence[float]]:
        """If `power` passed is a sequence, the `t_power` needs to be
        passed, defining the time-points for the `power` values and
        having the same length.
        Defaults to None, which is fine for time-independent `power`.
        """
        return None

    @property
    @abstractmethod
    def feeds(self) -> Dict[str, float]:
        """Dictionary of feed flows in [sccm] for all the species, which
        are fed to plasma. The feed flows are keyed by species IDs
        (distinct names/formulas/... - the keys need to be subset of the
        values returned by `Chemistry.species_ids`).
        """

    @property
    @abstractmethod
    def t_end(self) -> float:
        """End time of the simulation in [s]."""


class Equations(ABC):
    """"""

    @property
    @abstractmethod
    def ode_system_rhs(self) -> Callable[[float, ndarray], ndarray]:
        """"""


class GlobalModel(ABC):
    """"""

    @abstractmethod
    def solve(self, y0):
        """"""
