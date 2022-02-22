from pygmol.equations import ElectronEnergyEquations
from .resources import DefaultChemistry, DefaultParamsStat


def test():
    chem = DefaultChemistry()
    pl_params = DefaultParamsStat()
    ElectronEnergyEquations(chem, pl_params)


# TODO: test for the correct instantiation of all the Equations instance attributes!
