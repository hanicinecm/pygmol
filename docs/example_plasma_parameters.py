from pygmol.abc import PlasmaParameters


class ArO2PlasmaParameters(PlasmaParameters):
    radius = 0.000564
    length = 0.03
    pressure = 100_000.0
    power = (0.3, 0.3, 0, 0, 0.3, 0.3, 0, 0, 0.3, 0.3)
    t_power = (0, 0.003, 0.003, 0.006, 0.006, 0.009, 0.009, 0.012, 0.012, 0.015)
    feeds = {"O2": 0.3, "He": 300.0}
    temp_e = 1.0
    temp_n = 305.0
    t_end = 0.015


argon_oxygen_plasma_parameters = ArO2PlasmaParameters()
