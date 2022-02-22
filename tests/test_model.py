import pytest

from pygmol.abc import Chemistry, PlasmaParameters, Equations
from pygmol.model import Model, ModelSolutionError
from pygmol.chemistry import ChemistryValidationError, validate_chemistry
from pygmol.plasma_parameters import (
    PlasmaParametersValidationError,
    validate_plasma_parameters,
)

from .resources import DefaultChemistry, DefaultParamsDyn
from .test_chemistry import chem_dict
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
