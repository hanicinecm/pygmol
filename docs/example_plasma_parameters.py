from pygmol.abc import PlasmaParameters


class ArO2PlasmaParameters(PlasmaParameters):
    radius = 0.000564  # plasma radius in [m]
    length = 0.03  # plasma length in [m]
    pressure = 1.0e5  # plasma pressure in [Pa]
    power = (0.3, 0.3, 0, 0, 0.3, 0.3, 0, 0, 0.3, 0.3)  # absorbed power in [W]
    # times for the power series in [s]:
    t_power = (0, 0.003, 0.003, 0.006, 0.006, 0.009, 0.009, 0.012, 0.012, 0.015)
    feeds = {"O2": 0.3, "He": 300.0}  # gas inflow feeds in [sccm]
    temp_e = 1.0  # (initial) electron temperature in [eV]
    temp_n = 305.0  # neutral temperature in [K]
    t_end = 0.015  # simulation time in [s]


argon_oxygen_plasma_parameters = ArO2PlasmaParameters()
