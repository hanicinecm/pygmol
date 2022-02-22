import numpy as np

from pygmol.equations import ElectronEnergyEquations
from .resources import DefaultChemistryMinimal, DefaultParamsMinimal


def test():
    chem = DefaultChemistryMinimal()
    params = DefaultParamsMinimal()
    equations = ElectronEnergyEquations(chem, params)

    fun = equations.ode_system_rhs
    assert callable(fun)
    nan_y = fun(np.float64(0), np.array([np.nan, np.nan, np.nan]))
    assert np.isnan(nan_y).all()
    final_labels = equations.final_solution_labels
    assert list(final_labels) == ["Ar", "Ar+", "e", "T_e", "T_n", "p"]


# TODO: this is a dirty hack to get some test coverage, ideally I want to implement
#       unit tests testing all the methods and attributes of ElectronEnergyEquations.
