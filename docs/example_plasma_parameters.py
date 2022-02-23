from pygmol.abc import PlasmaParameters


class ArO2PlasmaParameters(PlasmaParameters):
    radius = 0.000564
    length = 0.03
    pressure = 1.0e5
    power = (0.3, 0.3, 0, 0, 0.3, 0.3, 0, 0, 0.3, 0.3)
    t_power = (0, 3.0e-3, 3.0e-3, 6.0e-3, 6.0e-3, 9.0e-3, 9e-3, 12e-3, 12e-3, 15e-2)
    feeds = {"O2": 0.3, "He": 300.0}
    temp_e = 1.0
    temp_n = 305.0
    t_end = 1.5e-2


class ArgonPlasmaParameters(PlasmaParameters):
    radius = 0.3  # [m]
    length = 0.3  # [m]
    pressure = 133  # [Pa]
    power = [2000, 2000, 0, 0, 2000, 2000]  # [W]
    t_power = [0, 0.001, 0.001, 0.002, 0.002, 0.003]  # [s]
    feeds = {"Ar": 100.0}  # [sccm]
    temp_e = 1.0  # [eV]
    temp_n = 500.0  # [K]
    t_end = 0.003


argon_plasma_parameters = ArgonPlasmaParameters()
argon_oxygen_plasma_parameters = ArO2PlasmaParameters()
