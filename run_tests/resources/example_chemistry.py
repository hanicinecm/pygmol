import inspect
from copy import deepcopy

from pygmol.abc import Chemistry


class ExampleChemistry(Chemistry):
    """An example chemistry for the following

    Species:
    --------
    O, O-, O--, O2, Ar, Ar+, Ar++, Ar*, Ar**

    Reactions:
    ----------
    1: e + O2 > O2 + e
    2: e + Ar > e + Ar
    3: e + Ar+ > Ar+ + e
    4: e + Ar > Ar* + e  # made up for testing purposes...
    6: e + Ar* > Ar** + e
    7: e + Ar* > Ar + e
    8: e + Ar** > Ar + e
    9: e + Ar** > Ar* + e
    10: Ar + e > Ar++ + e + e + e  # made up for testing purposes...
    12: e + Ar* > Ar+ + e + e
    13: e + Ar** > Ar+ + e + e
    14: e + O > O-  # made up for testing purposes...
    15: e + O2 > O- + O
    16: e + e + O > O--  # made up for testing purposes...
    19: e + O2 > e + O + O
    20: e + O- > O--  # made up for testing purposes...
    21: Ar+ + e > Ar++ + e + e
    22: Ar* + Ar* > Ar+ + Ar + e
    23: Ar* + Ar** > Ar+ + Ar + e
    25: Ar + Ar+ > Ar + Ar+
    26: O- + O > O2 + e
    27: Ar+ + O- > Ar + O
    28: Ar+ + O- + M > Ar + O + M  # made up for testing purposes...
    29: Ar++ + O-- + M > Ar + O + M  # made up for testing purposes...
    30: Ar** > Ar*
    31: O-- > e + O-  # made up for testing purposes...
    """

    species_ids = ["O", "O-", "O--", "O2", "Ar", "Ar+", "Ar++", "Ar*", "Ar**"]
    species_charges = [0, -1, -2, 0, 0, 1, 2, 0, 0]
    species_masses = [
        15.9994,
        15.9994,
        15.9994,
        31.9988,
        39.948,
        39.948,
        39.948,
        39.948,
        39.948,
    ]
    species_lj_sigma_coefficients = [
        3.05,
        3.05,
        3.05,
        3.47,
        3.542,
        3.542,
        3.542,
        3.542,
        3.542,
    ]
    # only ions and excited species get stuck to surfaces
    species_surface_sticking_coefficients = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    # singly ionized and excited species return as neutrals
    # doubly ionized and excited species return half as singly, half as neutrals
    species_surface_return_matrix = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    reactions_ids = [
        1,
        2,
        3,
        4,
        6,
        7,
        8,
        9,
        10,
        12,
        13,
        14,
        15,
        16,
        19,
        20,
        21,
        22,
        23,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
    ]
    reactions_strings = [
        "e + O2 -> O2 + e",
        "e + Ar -> e + Ar",
        "e + Ar+ -> Ar+ + e",
        "e + Ar -> Ar* + e",  # made up for testing purposes...
        "e + Ar* -> Ar** + e",
        "e + Ar* -> Ar + e",
        "e + Ar** -> Ar + e",
        "e + Ar** -> Ar* + e",
        "Ar + e -> Ar++ + e + e + e",  # made up for testing purposes...
        "e + Ar* -> Ar+ + e + e",
        "e + Ar** -> Ar+ + e + e",
        "e + O -> O-",  # made up for testing purposes...
        "e + O2 -> O- + O",
        "e + e + O -> O--",  # made up for testing purposes...
        "e + O2 -> e + O + O",
        "e + O- -> O--",  # made up for testing purposes...
        "Ar+ + e -> Ar++ + e + e",
        "Ar* + Ar* -> Ar+ + Ar + e",
        "Ar* + Ar** -> Ar+ + Ar + e",
        "Ar + Ar+ -> Ar + Ar+",
        "O- + O -> O2 + e",
        "Ar+ + O- -> Ar + O",
        "Ar+ + O- + M -> Ar + O + M",  # made up for testing purposes...
        "Ar++ + O-- + M -> Ar + O + M",  # made up for testing purposes...
        "Ar** -> Ar*",
        "O-- -> e + O-",  # made up for testing purposes...
    ]
    reactions_arrh_a = [
        3.93e-14,
        2.660e-13,
        1.61e-10,
        1.00e-11,
        7.91e-13,
        1.96e-15,
        2.26e-14,
        6.86e-13,
        1e-09,
        1.42e-13,
        3.45e-13,
        1e-16,
        6.74e-16,
        1.00e-32,
        1.75e-14,
        1e-16,
        1e-09,
        1e-16,
        1.2e-15,
        5.66e-16,
        3e-16,
        2.7e-13,
        1e-37,
        1e-37,
        100000.0,
        100.0,
    ]
    reactions_arrh_b = [
        0.628,
        -0.0128,
        -1.22,
        0.405,
        0.281,
        0.319,
        0.102,
        0.337,
        0.5,
        0.195,
        -0.0177,
        0.0,
        -1.02,
        0.0,
        -1.28,
        0.0,
        0.5,
        0.0,
        0.5,
        0.5,
        -0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    reactions_arrh_c = [
        -0.0198,
        3.15,
        0.0382,
        12.1,
        1.9,
        0.985,
        0.0441,
        0.102,
        15.0,
        4.38,
        4.11,
        0.0,
        5.78,
        0.0,
        7.38,
        0.0,
        10.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    reactions_el_energy_losses = [
        0.0,
        0.0,
        0.0,
        11.6,
        1.575,
        0.0,
        0.0,
        0.0,
        15.0,
        4.425,
        2.9,
        0.0,
        0.0,
        0.0,
        4.5,
        0.0,
        10.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    reactions_elastic_flags = [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    reactions_electron_stoich_lhs = [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    reactions_electron_stoich_rhs = [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        3,
        2,
        2,
        0,
        0,
        0,
        1,
        0,
        2,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
    ]
    reactions_arbitrary_stoich_lhs = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
    ]
    reactions_arbitrary_stoich_rhs = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
    ]
    reactions_species_stoichiomatrix_lhs = [
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
    ]
    reactions_species_stoichiomatrix_rhs = [
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
    ]

    def __init__(self):
        # save all the class attributes as instance attributes:
        for attr, val in inspect.getmembers(ExampleChemistry):
            if not attr[0].startswith("_"):
                setattr(self, attr, deepcopy(getattr(ExampleChemistry, attr)))
