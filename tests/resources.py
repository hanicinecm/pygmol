from dataclasses import dataclass
from typing import Sequence

from pygmol.chemistry import Chemistry


@dataclass
class DefaultChemistry(Chemistry):
    """Default concrete Chemistry subclass for testing."""

    # species: e-, Ar, Ar+
    species_ids: Sequence[str] = ("Ar", "Ar+")
    species_charges: Sequence[int] = (0, 1)
    species_masses: Sequence[float] = (9.948, 39.948)
    species_lj_sigma_coefficients: Sequence[float] = (0.542, 3.542)
    species_surface_sticking_coefficients: Sequence[float] = (0.0, 1.0)
    species_surface_return_matrix: Sequence[Sequence[float]] = ((0, 1), (0, 0))
    # reactions:
    # 3: e- + Ar -> Ar + e-
    # 7: e- + Ar -> Ar+ + e- + e-
    # 48: e- + Ar+ -> Ar+ + e-
    reactions_ids: Sequence[int] = (3, 7, 48)
    reactions_arrh_a: Sequence[float] = (2.66e-07, 3.09e-08, 1.61e-04)
    reactions_arrh_b: Sequence[float] = (-1.28e-02, 4.46e-01, -1.22e00)
    reactions_arrh_c: Sequence[float] = (3.15e00, 1.70e01, 3.82e-02)
    reactions_el_energy_losses: Sequence[float] = (0.0, 15.875, 0.0)
    reactions_elastic_flags: Sequence[bool] = (True, False, True)
    reactions_electron_stoich_lhs: Sequence[int] = (1, 1, 1)
    reactions_electron_stoich_rhs: Sequence[int] = (1, 2, 1)
    reactions_arbitrary_stoich_lhs: Sequence[int] = (0, 0, 0)
    reactions_arbitrary_stoich_rhs: Sequence[int] = (0, 0, 0)
    reactions_species_stoichiomatrix_lhs: Sequence[Sequence[int]] = (
        (1, 0),
        (1, 0),
        (0, 1),
    )
    reactions_species_stoichiomatrix_rhs: Sequence[Sequence[int]] = (
        (1, 0),
        (0, 1),
        (0, 1),
    )
