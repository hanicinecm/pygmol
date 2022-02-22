"""Module providing a validation function for the concrete subclasses of `Chemistry` ABC
as well as a concrete subclass which can be constructed from a dict.
"""
import numpy as np

from .abc import Chemistry


def chemistry_from_dict(chemistry_dict: dict) -> Chemistry:
    """Concrete Chemistry subclass instance factory.

    Creates a concrete subclass of the `Chemistry` ABC and sets all the
    (key, value) pairs passed down in `chemistry_dict` as class attributes.
    Returns an *instance* of the new concrete type. If the `chemistry_dict` omits
    any of the abstract properties of `Chemistry`, an usual TypeError will be
    raised on instantiation.

    Parameters
    ----------
    chemistry_dict : dict

    Returns
    -------
    Chemistry
        Instance of a dynamically created *concrete subclass* of the `Chemistry` ABC.

    Raises
    ------
    TypeError
        If the `chemistry_dict` does not adhere to the interface defined by the
        `Chemistry` ABC.
    """
    concrete_subclass = type("ChemistryFromDict", (Chemistry,), chemistry_dict)
    concrete_instance = concrete_subclass()
    return concrete_instance


class ChemistryValidationError(Exception):
    """A custom exception signaling inconsistent or unphysical data given by the
    concrete `Chemistry` class instance.
    """

    pass


def validate_chemistry(chemistry: Chemistry):
    """Validation function for a concrete `Chemistry` subclass.

    Various inconsistencies are checked for, including uniqueness of the `species_ids`
    and `reactions_ids`, or the correct shapes of all the sequences.

    The physicality of values in the species and reactions sequences is not currently
    checked for.

    TODO: Implement validation of the values - non-negativity etc.

    Parameters
    ----------
    chemistry : Chemistry
        Object defining the chemistry set.

    Raises
    ------
    ChemistryValidationError
        If any of the validation checks fails.
    """
    sp_attrs = [attr for attr in dir(chemistry) if attr.startswith("species_")]
    sp_attrs_lengths = [len(getattr(chemistry, attr)) for attr in sp_attrs]
    if len(set(sp_attrs_lengths)) != 1:
        raise ChemistryValidationError(
            "All the attributes describing species need to have the same dimension!"
        )
    r_attrs = [attr for attr in dir(chemistry) if attr.startswith("reactions_")]
    r_attrs_lengths = [len(getattr(chemistry, attr)) for attr in r_attrs]
    if len(set(r_attrs_lengths)) != 1:
        raise ChemistryValidationError(
            "All the attributes describing reactions need to have the same dimension!"
        )
    if len(set(chemistry.species_ids)) != len(chemistry.species_ids):
        raise ChemistryValidationError("All the species IDs need to be unique!")
    if len(set(chemistry.reactions_ids)) != len(chemistry.reactions_ids):
        raise ChemistryValidationError("All the reactions IDs need to be unique!")
    # 2D arrays' shapes:
    stoichiomatrix_shape_lhs = np.array(
        chemistry.reactions_species_stoichiomatrix_lhs
    ).shape
    stoichiomatrix_shape_rhs = np.array(
        chemistry.reactions_species_stoichiomatrix_rhs
    ).shape
    for stoich_shape in stoichiomatrix_shape_lhs, stoichiomatrix_shape_rhs:
        if stoich_shape[1] != len(chemistry.species_ids):
            raise ChemistryValidationError(
                "Incorrect shape of the stoichiomatrix arrays, must be (Nr, Ns)!"
            )
    return_matrix_shape = np.array(chemistry.species_surface_return_matrix).shape
    if return_matrix_shape[0] != return_matrix_shape[1]:
        raise ChemistryValidationError(
            "Incorrect shape of the return matrix array, must be (Ns, Ns)!"
        )
