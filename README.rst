|Tests action| |Run tests action| |Unit-test coverage| |GitHub license| |PyPI version| |PyPI pyversions| |Code style|

.. |Tests action| image:: https://github.com/hanicinecm/pygmol/workflows/unit-tests/badge.svg
   :target: https://github.com/hanicinecm/pygmol/actions
.. |Run tests action| image:: https://github.com/hanicinecm/pygmol/workflows/run-tests/badge.svg
   :target: https://github.com/hanicinecm/pygmol/actions
.. |Unit-test coverage| image:: https://codecov.io/gh/hanicinecm/pygmol/branch/master/graph/badge.svg?token=TNKBDTVGFV
   :target: https://codecov.io/gh/hanicinecm/pygmol
.. |GitHub license| image:: https://img.shields.io/github/license/hanicinecm/pygmol.svg
   :target: https://github.com/hanicinecm/pygmol/blob/master/LICENSE
.. |PyPI version| image:: https://img.shields.io/pypi/v/pygmol.svg
   :target: https://pypi.python.org/pypi/pygmol/
.. |PyPI pyversions| image:: https://img.shields.io/pypi/pyversions/pygmol.svg
   :target: https://pypi.python.org/pypi/pygmol/
.. |Code style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black


**********************
Introduction to PyGMol
**********************

**PyGMol** (the Python Global Model) is a simple-tu-use 0D model, or
global model, for modeling plasma chemistry.

At its current form, the ``pygmol`` package defines the equations belonging to the
system of ordinary differential equations (ODE) which govern the plasma chemistry, and
solves them using the ``scipy.integrate.solve_ivp`` solver.
Quantities resolved presently by the ``pygmol`` model are the densities of all the
heavy species specified by the chemistry input, electron density, and electron
temperature (while heavy-species temperature is treated as a parameter to the model).

The equations being solved for by the model are documented in their full form in the
equations_ document. The package inner workings, structure and some useful examples
are discussed in the documentation_. For further info, refer to the codebase
docstrings.


Installation:
=============

The ``pygmol`` package can be installed either from PyPI_

.. code-block:: bash

    python3 -m pip install pygmol

or from the GitHub_ page

.. code-block:: bash

    python3 -m pip install git+https://github.com/hanicinecm/pygmol.git


For Developers:
===============
It goes without saying that any development should be done in a clean virtual
environment.
After cloning or forking the project from its GitHub_ page, ``pygmol`` can be
installed into the virtual environment in the editable mode by running

.. code-block:: bash

    pip install -e .[dev]

The ``[dev]`` extra installs (apart from the package dependencies) also several
development-related packages, such as ``pytest``, ``coverage``, ``ipython``, or
``black``.
The unit tests and run tests can then be executed by running (from the project root
directory)

.. code-block:: bash

    pytest --cov
    pytest run_tests

respectively.

The project does not have the ``requirements.txt`` file by design, as all the package
dependencies are rather handled by the ``setup.py``.
The package therefore needs to be installed locally to run the tests, which grants the
testing process another layer of usefulness.

Docstrings in the project adhere to the numpydoc_ styling.
The project code is formatted by ``black``.


.. _equations: https://github.com/hanicinecm/pygmol/blob/master/docs/equations.pdf
.. _documentation: https://github.com/hanicinecm/pygmol/tree/master/docs/index.rst
.. _GitHub: https://github.com/hanicinecm/pygmol
.. _PyPI: https://pypi.org/project/pygmol/
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html