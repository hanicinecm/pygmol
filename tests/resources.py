from pygmol.abc import Chemistry, PlasmaParameters


class DefaultChemistry(Chemistry):
    """Default concrete Chemistry subclass for testing."""

    # species: e-, Ar, Ar+
    species_ids = ["Ar", "Ar+"]
    species_charges = [0, 1]
    species_masses = [39.948, 39.948]
    species_lj_sigma_coefficients = [0.542, 3.542]
    species_surface_sticking_coefficients = [0.0, 1.0]
    species_surface_return_matrix = [0, 1], [0, 0]
    # reactions:
    # 3: e- + Ar -> Ar + e-
    # 7: e- + Ar -> Ar+ + e- + e-
    # 48: e- + Ar+ -> Ar+ + e-
    reactions_ids = [3, 7, 48]
    reactions_arrh_a = [2.66e-07, 3.09e-08, 1.61e-04]
    reactions_arrh_b = [-1.28e-02, 4.46e-01, -1.22e00]
    reactions_arrh_c = [3.15e00, 1.70e01, 3.82e-02]
    reactions_el_energy_losses = [0.0, 15.875, 0.0]
    reactions_elastic_flags = [True, False, True]
    reactions_electron_stoich_lhs = [1, 1, 1]
    reactions_electron_stoich_rhs = [1, 2, 1]
    reactions_arbitrary_stoich_lhs = [0, 0, 0]
    reactions_arbitrary_stoich_rhs = [0, 0, 0]
    reactions_species_stoichiomatrix_lhs = [
        [1, 0],
        [1, 0],
        [0, 1],
    ]
    reactions_species_stoichiomatrix_rhs = [
        [1, 0],
        [0, 1],
        [0, 1],
    ]

    def __init__(self, **kwargs):
        for attr, val in kwargs.items():
            setattr(self, attr, val)
        super().__init__()


class DefaultChemistryMinimal(Chemistry):
    """Default concrete Chemistry subclass for testing."""

    # species: e-, Ar, Ar+
    species_ids = ["Ar", "Ar+"]
    species_surface_return_matrix = [0, 1], [0, 0]
    # reactions:
    # 3: e- + Ar -> Ar + e-
    # 7: e- + Ar -> Ar+ + e- + e-
    # 48: e- + Ar+ -> Ar+ + e-
    reactions_ids = [3, 7, 48]
    reactions_arrh_a = [2.66e-07, 3.09e-08, 1.61e-04]
    reactions_arrh_b = [-1.28e-02, 4.46e-01, -1.22e00]
    reactions_arrh_c = [3.15e00, 1.70e01, 3.82e-02]
    reactions_el_energy_losses = [0.0, 15.875, 0.0]
    reactions_elastic_flags = [True, False, True]
    reactions_electron_stoich_lhs = [1, 1, 1]
    reactions_electron_stoich_rhs = [1, 2, 1]
    reactions_arbitrary_stoich_lhs = [0, 0, 0]
    reactions_arbitrary_stoich_rhs = [0, 0, 0]
    reactions_species_stoichiomatrix_lhs = [
        [1, 0],
        [1, 0],
        [0, 1],
    ]
    reactions_species_stoichiomatrix_rhs = [
        [1, 0],
        [0, 1],
        [0, 1],
    ]

    def __init__(self, **kwargs):
        for attr, val in kwargs.items():
            setattr(self, attr, val)
        super().__init__()


class DefaultParamsStat(PlasmaParameters):
    """Default concrete PlasmaParameters for testing with static power"""

    radius = 1.0
    length = 2.0
    pressure = 3.0
    power = 400.0
    feeds = {"Ar": 5.0}
    temp_e = 6.0
    temp_n = 700.0
    t_end = 8.0

    def __init__(self, **kwargs):
        for attr, val in kwargs.items():
            setattr(self, attr, val)
        super().__init__()


class DefaultParamsDyn(PlasmaParameters):
    """Default concrete PlasmaParameters for testing with dynamic power"""

    radius = 1.0
    length = 2.0
    pressure = 3.0
    power = [0.0, 400.0]
    t_power = [0.0, 1.0]
    feeds = {"Ar": 5.0}
    temp_e = 6.0
    temp_n = 700.0
    t_end = 8.0

    def __init__(self, **kwargs):
        for attr, val in kwargs.items():
            setattr(self, attr, val)
        super().__init__()


class DefaultParamsMinimal(PlasmaParameters):
    """Default concrete PlasmaParameters for testing, implementing the
    absolute minimum of the values
    """

    radius, length, pressure, power = 1.0, 2.0, 2.0, 400.0
