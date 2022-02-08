"""Module containing the abstract base class (ABC) `Chemistry`,
defining the interface of the chemistry object needed by the ``pygmol``
framework.
"""
from abc import ABC, abstractmethod
from typing import Union, Sequence


class Chemistry(ABC):
    """An abstract base class (ABC) defining the interface of objects
    describing the plasma chemistry, expected by other classes of the
    ``pygmol`` framework.

    A chemistry is a collection of species and reactions, and data
    attached to them. Species can be divided into heavy species (neutral
    and ions), and *special* species: electron and an arbitrary species
    'M'.

    The heavy species data required by the model are encoded in the
    following attributes/properties, which the `Chemistry` subclasses
    must implement:
        `species_ids`
        `species_charges`
        `species_masses`
        `species_lj_sigma_coefficients`
        `species_surface_sticking_coefficients`
        `species_surface_return_matrix`

    The reactions kinetic data (and metadata) are then encoded in the
    following attributes/properties:
        `reactions_ids`
        `reactions_arrh_a`
        `reactions_arrh_b`
        `reactions_arrh_c`
        `reactions_el_energy_losses`
        `reactions_elastic_flags`

    And finally, the relationship between the reactions and heavy
    species (and *special* species) is encoded in the following
    attributes/properties:
        `reactions_electron_stoich_lhs`
        `reactions_electron_stoich_rhs`
        `reactions_arbitrary_stoich_lhs`
        `reactions_arbitrary_stoich_rhs`
        `reactions_species_stoichiomatrix_lhs`
        `reactions_species_stoichiomatrix_rhs`

    See the abstract attributes below, and their definitions as
    properties for explanation and signatures.
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


class ChemistryValidationError(Exception):
    """A custom exception signaling inconsistent or unphysical data
    given by the concrete `Chemistry` class instance.
    """

    pass


def validate_chemistry(chemistry: Chemistry):
    """Validation function for a concrete `Chemistry` subclass.

    Various inconsistencies are checked for, including uniqueness of
    the `species_ids` and `reactions_ids`, or the correct shapes of
    all the sequences.

    Raises
    ------
    ChemistryValidationError
    """
    if chemistry is None:
        raise ChemistryValidationError("")
