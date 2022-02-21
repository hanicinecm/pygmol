import inspect

from pygmol.abc import PlasmaParameters


class ExamplePlasmaParameters(PlasmaParameters):

    radius = 0.1
    length = 0.1
    pressure = 100.0
    power = 1000.0
    t_power = None
    feeds = {}
    temp_e = 1.0
    temp_n = 500.0
    t_end = 1.0

    def __init__(self):
        # save all the class attributes as instance attributes:
        for attr, val in inspect.getmembers(ExamplePlasmaParameters):
            if not attr[0].startswith("_"):
                setattr(self, attr, getattr(ExamplePlasmaParameters, attr))
