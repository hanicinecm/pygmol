from pygmol.chemistry import validate_chemistry

from .resources import DefaultChemistry


def test_default_chemistry():
    validate_chemistry(DefaultChemistry())
