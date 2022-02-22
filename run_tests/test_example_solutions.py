from itertools import combinations
import inspect

import numpy as np
from pandas import Series
from pygmol.model import Model
from pygmol.equations import ElectronEnergyEquations
from pytest import fixture

from .resources.example_chemistry import ExampleChemistry
from .resources.example_plasma_parameters import ExamplePlasmaParameters
from .resources.example_solutions import solutions


@fixture
def chemistry():
    return ExampleChemistry()


@fixture
def plasma_params():
    return ExamplePlasmaParameters()


def solutions_match(
    sol_given: Series, sol_expected: Series, verbose: bool = True
) -> bool:
    if verbose:
        print("solution expected:")
        print(",".join(str(v) for v in sol_expected.values))
        print("solution given:")
        print(",".join(str(v) for v in sol_given.values))
    return np.isclose(sol_given.values, sol_expected.values, rtol=1.0e-1).all()


def solution_expected(model: Model, test_case: str) -> bool:
    return solutions_match(model.get_solution_final(), solutions.loc[test_case])


def run(model):
    model.run()


def test_all_example_solutions_unique():
    for index1, index2 in combinations(solutions.index, 2):
        assert not solutions_match(
            solutions.loc[index1], solutions.loc[index2], verbose=False
        )


def test_solution_nominal(chemistry, plasma_params):
    model = Model(chemistry, plasma_params)
    run(model)
    assert solution_expected(model, "nominal")


def test_solution_feeds(chemistry, plasma_params):
    plasma_params.feeds = {"O2": 10000, "Ar": 100}
    model = Model(chemistry, plasma_params)
    run(model)
    assert solution_expected(model, "feeds")


def test_solution_pressure(chemistry, plasma_params):
    plasma_params.pressure = 10
    model = Model(chemistry, plasma_params)
    run(model)
    assert solution_expected(model, "pressure")


def test_solution_power(chemistry, plasma_params):
    plasma_params.power = 100_000  # grossly inflated power to see bigger difference
    model = Model(chemistry, plasma_params)
    run(model)
    assert solution_expected(model, "power")


def test_solution_power_t(chemistry, plasma_params):
    plasma_params.t_power = [0, 0.99, 0.99, 1.0]
    plasma_params.power = [1000, 1000, 0, 0]
    model = Model(chemistry, plasma_params)
    run(model)
    assert solution_expected(model, "power_t")


def test_solution_stick(chemistry, plasma_params):
    chemistry.species_surface_sticking_coefficients = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    model = Model(chemistry, plasma_params)
    run(model)
    assert solution_expected(model, "stick")


def test_solution_surf_and_dim(chemistry, plasma_params):
    chemistry.species_surface_sticking_coefficients = [0, 1, 1, 0, 0, 1, 1, 1, 1]
    chemistry.species_surface_return_matrix = [
        [0.0, 100, 90, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 100, 90, 100, 50],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]  # grossly inflated to see the difference - each 1 stuck species returns 100 other
    model = Model(chemistry, plasma_params)
    run(model)
    assert solution_expected(model, "surf")

    plasma_params.radius *= 10
    plasma_params.length *= 10
    plasma_params.power *= 1000
    run(model)
    assert solution_expected(model, "dim")


def test_solution_arrh_a(chemistry, plasma_params):
    for i in [4, 20, 21, 23]:
        # these are indices of reactions (not reaction ids!) of most significant ones.
        chemistry.reactions_arrh_a[i] /= 100
    model = Model(chemistry, plasma_params)
    run(model)
    assert solution_expected(model, "arrh_a")
