from pygmol.model import Model
from run_tests.resources.example_chemistry import ExampleChemistry
from run_tests.resources.example_plasma_parameters import ExamplePlasmaParameters


if __name__ == "__main__":
    model = Model(ExampleChemistry(), ExamplePlasmaParameters())
    model.run()
    # print(model.get_solution())
    print(model.get_volumetric_rates_matrix())
