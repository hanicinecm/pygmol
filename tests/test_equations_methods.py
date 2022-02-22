import numpy as np
import pytest

from pygmol.equations import ElectronEnergyEquations
from .resources import DefaultChemistry, DefaultParamsStat, DefaultParamsDyn

default_chemistry = DefaultChemistry()
default_plasma_params = DefaultParamsStat()

equations = ElectronEnergyEquations(default_chemistry, default_plasma_params)


@pytest.fixture
def y():
    return np.array([1, 2, 3, 4, 5])


def test_get_density_vector(y):
    assert list(equations.get_density_vector(y)) == [1, 2, 3, 4]


def test_get_electron_energy_density(y):
    assert equations.get_electron_energy_density(y) == 5


def test_get_total_density(y):
    assert equations.get_total_density(y) == 10
    assert equations.get_total_density(y, np.array([1, 2])) == 3


def test_get_total_pressure(y, monkeypatch):
    monkeypatch.setattr(equations, "k", 42)
    assert equations.get_total_pressure(y) == 10 * 42 * 700
    assert equations.get_total_pressure(y, n_tot=42.0) == 42 * 42 * 700


def test_get_ion_temperature(y, monkeypatch):
    monkeypatch.setattr(equations, "e", 1)
    monkeypatch.setattr(equations, "k", 1)
    assert equations.get_ion_temperature(y, p=0.0) == 0.5
    monkeypatch.setattr(equations, "get_total_pressure", lambda _: 0.0)
    assert equations.get_ion_temperature(y) == 0.5


def test_get_electron_density(y, monkeypatch):
    n = np.array([1, 2, 3, 4])
    charges = np.array([0, -1, 0, 2])
    monkeypatch.setattr(equations, "sp_charges", charges)
    assert equations.get_electron_density(y, n) == 6.0
    assert equations.get_electron_density(y) == 6.0


def test_get_power_ext():
    params = DefaultParamsDyn(power=[0, 400], t_power=[0, 1])
    eqs = ElectronEnergyEquations(default_chemistry, params)
    assert eqs.get_power_ext(0) == 0
    assert eqs.get_power_ext(1) == 400
    assert eqs.get_power_ext(0.5) == 200
    assert eqs.get_power_ext(-1) == 0
    assert eqs.get_power_ext(2) == 400
    params = DefaultParamsDyn(power=[100], t_power=[0])
    eqs = ElectronEnergyEquations(default_chemistry, params)
    assert eqs.get_power_ext(0) == 100
    assert eqs.get_power_ext(-1) == 100
    assert eqs.get_power_ext(1) == 100
    params = DefaultParamsDyn(power=100, t_power=None)
    eqs = ElectronEnergyEquations(default_chemistry, params)
    assert eqs.get_power_ext(0) == 100
    assert eqs.get_power_ext(-1) == 100
    assert eqs.get_power_ext(1) == 100


# TODO: ALL the methods of the concrete Equations subclass should be unit-tested!
