import pytest

from pygmol.abc import Equations


# noinspection PyAbstractClass,PyTypeChecker
def test_concrete_equations():
    # noinspection PyMethodOverriding
    class ConcreteEquations(Equations):

        ode_system_rhs = None
        final_solution_labels = None

        def __init__(self, chemistry, plasma_params):
            super().__init__(chemistry, plasma_params)

        def get_final_solution_values(self, y):
            pass

        def get_y0_default(self, initial_densities):
            pass

    concrete_equations = ConcreteEquations(None, None)
    with pytest.raises(NotImplementedError):
        concrete_equations.get_reaction_rates(None)
    with pytest.raises(NotImplementedError):
        concrete_equations.get_wall_fluxes(None)
