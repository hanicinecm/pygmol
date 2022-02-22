import pytest
from collections import namedtuple

import numpy as np
import pygmol.model
from pygmol.abc import Chemistry, PlasmaParameters, Equations
from pygmol.model import Model, ModelSolutionError
from pygmol.chemistry import ChemistryValidationError, validate_chemistry
from pygmol.plasma_parameters import (
    PlasmaParametersValidationError,
    validate_plasma_parameters,
)

from .resources import DefaultChemistry, DefaultParamsDyn, MockEquations

# noinspection PyUnresolvedReferences
from .test_chemistry import chem_dict

# noinspection PyUnresolvedReferences
from .test_plasma_parameters import params_dict


def test_instantiation_with_classes():
    model = Model(DefaultChemistry(), DefaultParamsDyn())
    assert isinstance(model.chemistry, Chemistry)
    assert isinstance(model.plasma_params, PlasmaParameters)
    assert model.equations is None
    assert model.chemistry.species_ids == ["Ar", "Ar+"]
    assert model.plasma_params.feeds == {"Ar": 5.0}


def test_instantiation_with_dicts(chem_dict, params_dict):
    assert isinstance(chem_dict, dict)
    assert isinstance(params_dict, dict)
    model = Model(chem_dict, params_dict)
    assert isinstance(model.chemistry, Chemistry)
    assert isinstance(model.plasma_params, PlasmaParameters)
    assert model.equations is None
    assert model.chemistry.species_ids == ["Ar", "Ar+"]
    assert model.plasma_params.feeds == {"Ar": 420.0}


def test_instantiation_inconsistent_arguments():
    chemistry = DefaultChemistry(reactions_ids=[1, 2, 3, 4])  # one more than others
    params = DefaultParamsDyn()
    with pytest.raises(ChemistryValidationError):
        Model(chemistry, params)
    chemistry = DefaultChemistry()
    params = DefaultParamsDyn(power=[0, 400, 0])  # one more point than t_power
    with pytest.raises(PlasmaParametersValidationError):
        Model(chemistry, params)
    # now, both chemistry and params are consistent, but inconsistent with each other
    params = DefaultParamsDyn(feeds={"O2": 5.0})
    validate_chemistry(chemistry)
    validate_plasma_parameters(params)
    with pytest.raises(PlasmaParametersValidationError):
        Model(chemistry, params)


def test_initialize_equations():
    model = Model(DefaultChemistry(), DefaultParamsDyn())
    assert model.equations is None
    model._initialize_equations()
    assert isinstance(model.equations, Equations)


def test_premature_solving():
    model = Model(DefaultChemistry(), DefaultParamsDyn())
    with pytest.raises(ModelSolutionError):
        model._solve()
    with pytest.raises(ModelSolutionError):
        model._build_solution()
    with pytest.raises(ModelSolutionError):
        model.diagnose("foo")


def _get_mock_ode_result(success=True, dimension=3, t_samples=10):
    OdeResult = namedtuple("OdeResult", "message,status,success,t,y")
    status = int(not success)
    ode_result = OdeResult(
        message="dummy message",
        status=status,
        success=success,
        t=np.arange(t_samples),
        y=np.arange(t_samples * dimension).reshape((dimension, t_samples)),
    )
    return ode_result


def _get_mock_solve_ivp(success=True, dimension=3, t_samples=10, raises=False):
    # noinspection PyUnusedLocal
    def mock_solve_ivp(*args, **kwargs):
        if raises:
            raise ValueError("Mock solver error!")
        return _get_mock_ode_result(
            success=success, dimension=dimension, t_samples=t_samples
        )

    return mock_solve_ivp


def test_solve(monkeypatch):
    mock_equations = MockEquations(DefaultChemistry(), DefaultParamsDyn())
    mock_equations.get_y0_default = lambda: np.arange(3)
    model = Model(mock_equations.chemistry, mock_equations.plasma_params)
    model.equations = mock_equations
    monkeypatch.setattr(
        pygmol.model,
        "solve_ivp",
        _get_mock_solve_ivp(success=True, dimension=3, t_samples=10),
    )
    model._solve()
    assert model.solution_raw is not None
    assert model.solution_raw.success
    assert len(model.solution_raw.t) == 10
    assert model.solution_raw.y.shape == (3, 10)
    assert model.solution_primary is None
    assert model.solution is None
    monkeypatch.setattr(
        pygmol.model,
        "solve_ivp",
        _get_mock_solve_ivp(success=True, dimension=3, t_samples=10, raises=True),
    )
    with pytest.raises(ModelSolutionError):
        model._solve()


def test_build_solution(monkeypatch):
    mock_equations = MockEquations(DefaultChemistry(), DefaultParamsDyn())
    mock_sol_labels = ["n_Ar", "n_Ar+", "n_e", "T_e", "T_n", "p"]
    mock_equations.final_solution_labels = mock_sol_labels
    mock_equations.get_final_solution_values = lambda y: np.arange(len(mock_sol_labels))
    model = Model(mock_equations.chemistry, mock_equations.plasma_params)
    model.equations = mock_equations
    model.solution_raw = _get_mock_ode_result(success=True, dimension=3, t_samples=10)
    model._build_solution()
    assert model.solution_primary.shape == (10, 3)
    assert model.t.shape == (10,)
    assert list(model.solution.columns) == ["t"] + mock_sol_labels
    assert model.solution.shape == (10, 1 + len(mock_sol_labels))


def test_success():
    model = Model(DefaultChemistry(), DefaultParamsDyn())
    model.solution_raw = _get_mock_ode_result(success=False)
    assert not model.success()


def test_diagnose():
    mock_equations = MockEquations(DefaultChemistry(), DefaultParamsDyn())
    mock_equations.get_foo_1d = lambda y: np.float64(0.42)
    mock_equations.get_foo_3d = lambda y: np.array([0.0, 1.0, 2.0])
    model = Model(mock_equations.chemistry, mock_equations.plasma_params)
    model.equations = mock_equations
    model.solution_primary = np.arange(25).reshape((5, 5))
    model.t = np.arange(5)
    with pytest.raises(ModelSolutionError):
        model.diagnose("non_existing_getter")
    diagnostics_1d = model.diagnose("foo_1d")
    assert diagnostics_1d.shape == (5, 2)
    assert list(diagnostics_1d.columns) == ["t", "foo_1d"]
    assert list(diagnostics_1d["t"]) == list(model.t)
    diagnostics_3d = model.diagnose("foo_3d")
    assert diagnostics_3d.shape == (5, 4)
    assert list(diagnostics_3d.columns) == ["t", "col1", "col2", "col3"]
    assert list(diagnostics_3d["t"]) == list(model.t)
    assert list(diagnostics_3d.iloc[-1]) == [4, 0, 1, 2]
