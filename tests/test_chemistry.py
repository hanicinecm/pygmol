import inspect

import numpy as np
import pytest

from pygmol.abc import Chemistry
from pygmol.chemistry import (
    validate_chemistry,
    ChemistryValidationError,
    chemistry_from_dict,
)
from .resources import DefaultChemistry, DefaultChemistryMinimal


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
    orig_stoichiomatrix = np.array(
        DefaultChemistry.reactions_species_stoichiomatrix_lhs
    )
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
        validate_chemistry(DefaultChemistry(species_surface_return_matrix=misshapen1))
    with pytest.raises(ChemistryValidationError):
        validate_chemistry(DefaultChemistry(species_surface_return_matrix=misshapen2))


def test_defaults():
    chem = DefaultChemistryMinimal()
    assert list(chem.reactions_strings) == ["Reaction 3", "Reaction 7", "Reaction 48"]
    assert list(chem.species_charges) == [0, 1]
    assert np.isclose(chem.species_masses, [39.95, 39.95], rtol=0.0001).all()
    assert list(chem.species_lj_sigma_coefficients) == [3, 3]
    assert list(chem.species_surface_sticking_coefficients) == [0, 1]


@pytest.fixture
def chem_dict():
    return {
        attr: val
        for attr, val in inspect.getmembers(DefaultChemistry)
        if not attr.startswith("_")
    }


def test_chemistry_from_dict_valid(chem_dict):
    chemistry = chemistry_from_dict(chem_dict)
    assert isinstance(chemistry, Chemistry)
    for attr, val in inspect.getmembers(DefaultChemistry):
        if not attr.startswith("_"):
            assert chem_dict[attr] == val
            assert getattr(chemistry, attr) == val


@pytest.mark.parametrize(
    "mandatory_attribute",
    [
        "species_ids",
        "reactions_ids",
        "species_surface_return_matrix",
        "reactions_arrh_a",
        "reactions_species_stoichiomatrix_lhs",
    ],
)
def test_chemistry_from_dict_invalid(mandatory_attribute, chem_dict):
    chem_dict = {
        key: val for key, val in chem_dict.items() if key != mandatory_attribute
    }
    # chemistry dict missing one of the abstract attributes, must raise TypeError
    with pytest.raises(TypeError):
        chemistry_from_dict(chem_dict)


def test_chemistry_from_dict_defaults(chem_dict):
    optional_attributes = {
        "species_charges",
        "species_masses",
        "species_lj_sigma_coefficients",
        "species_surface_sticking_coefficients",
        "reactions_strings",
    }
    chem_dict = {
        key: val for key, val in chem_dict.items() if key not in optional_attributes
    }
    chem = chemistry_from_dict(chem_dict)
    assert list(chem.species_charges) == [0, 1]
    assert np.isclose(chem.species_masses, [39.95, 39.95], rtol=0.0001).all()
    assert list(chem.species_lj_sigma_coefficients) == [3, 3]
    assert list(chem.species_surface_sticking_coefficients) == [0, 1]
    assert list(chem.reactions_strings) == ["Reaction 3", "Reaction 7", "Reaction 48"]


def test_incompatible_defaults(chem_dict):
    optional_attributes = {
        "species_charges",
        "species_masses",
        "species_lj_sigma_coefficients",
        "species_surface_sticking_coefficients",
    }
    chem_dict = {
        key: val for key, val in chem_dict.items() if key not in optional_attributes
    }
    chem_dict["species_ids"] = ["Ar", "Arr+"]
    chemistry = chemistry_from_dict(chem_dict)
    with pytest.raises(NotImplementedError):
        _ = chemistry.species_charges
    with pytest.raises(NotImplementedError):
        _ = chemistry.species_masses
    with pytest.raises(NotImplementedError):
        _ = chemistry.species_surface_sticking_coefficients
