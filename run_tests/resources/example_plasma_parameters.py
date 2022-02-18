from pygmol.abc import PlasmaParameters


class ExamplePlasmaParameters(PlasmaParameters):

    radius = 0.1
    length = 0.1
    pressure = 100.0
    power = 1000.0
    feeds = {}
    temp_e = 1.0
    temp_n = 500.0
    t_end = 1.0
