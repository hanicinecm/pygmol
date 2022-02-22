import pytest

import numpy as np

from .resources import MockEquations, DefaultChemistry, DefaultParamsStat


def test_concrete_equations():

    concrete_equations = MockEquations(DefaultChemistry(), DefaultParamsStat())
    y = np.array([])
    with pytest.raises(NotImplementedError):
        concrete_equations.get_reaction_rates(y)
    with pytest.raises(NotImplementedError):
        concrete_equations.get_wall_fluxes(y)
