import pytest

from pygmol.chemistry import validate_chemistry, ChemistryValidationError
import numpy as np

from .resources import DefaultChemistry


def test_default_chemistry():
    validate_chemistry(DefaultChemistry())


@pytest.mark.parametrize(
    "attr",
    [
        "species_ids",
        "species_charges",
        "species_masses",
        "species_lj_sigma_coefficients",
        "species_surface_sticking_coefficients",
        "species_surface_return_matrix",
        "reactions_ids",
        "reactions_arrh_c",
        "reactions_elastic_flags",
        "reactions_electron_stoich_rhs",
        "reactions_arbitrary_stoich_lhs",
        "reactions_species_stoichiomatrix_lhs",
    ],
)
def test_inconsistent_attributes_lengths(attr):
    orig_val = getattr(DefaultChemistry, attr)
    with pytest.raises(ChemistryValidationError):
        validate_chemistry(DefaultChemistry(**{attr: orig_val[:-1]}))
    with pytest.raises(ChemistryValidationError):
        validate_chemistry(DefaultChemistry(**{attr: list(orig_val) + [orig_val[-1]]}))


def test_unique_species_and_reactions_ids():
    orig_species_ids = DefaultChemistry.species_ids
    orig_reactions_ids = DefaultChemistry.reactions_ids
    with pytest.raises(ChemistryValidationError):
        non_unique_species_ids = list(orig_species_ids)
        non_unique_species_ids[-1] = non_unique_species_ids[0]
        validate_chemistry(DefaultChemistry(species_ids=non_unique_species_ids))
    with pytest.raises(ChemistryValidationError):
        non_unique_reactions_ids = list(orig_reactions_ids)
        non_unique_reactions_ids[-1] = non_unique_reactions_ids[0]
        validate_chemistry(DefaultChemistry(reactions_ids=non_unique_reactions_ids))


def test_incorrect_stoichiomatrix_shape():
    orig_stoichiomatrix = np.array(DefaultChemistry.reactions_species_stoichiomatrix_lhs)
    misshapen = orig_stoichiomatrix[:, :-1]
    with pytest.raises(ChemistryValidationError):
        validate_chemistry(
            DefaultChemistry(reactions_species_stoichiomatrix_lhs=misshapen)
        )
    with pytest.raises(ChemistryValidationError):
        validate_chemistry(
            DefaultChemistry(reactions_species_stoichiomatrix_rhs=misshapen)
        )
    misshapen = np.c_[orig_stoichiomatrix, orig_stoichiomatrix[:, 0]]
    with pytest.raises(ChemistryValidationError):
        validate_chemistry(
            DefaultChemistry(reactions_species_stoichiomatrix_lhs=misshapen)
        )
    with pytest.raises(ChemistryValidationError):
        validate_chemistry(
            DefaultChemistry(reactions_species_stoichiomatrix_rhs=misshapen)
        )


def test_incorrect_return_matrix_shape():
    return_matrix = np.array(DefaultChemistry.species_surface_return_matrix)
    misshapen1 = return_matrix[:, :-1]
    misshapen2 = np.c_[return_matrix, return_matrix[:, 0]]
    with pytest.raises(ChemistryValidationError):
        validate_chemistry(
            DefaultChemistry(species_surface_return_matrix=misshapen1)
        )
    with pytest.raises(ChemistryValidationError):
        validate_chemistry(
            DefaultChemistry(species_surface_return_matrix=misshapen2)
        )
