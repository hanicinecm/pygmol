***************************
``Equations`` Documentation
***************************

The ``pygmol.abc.Equations`` is the abstract base class (ABC) to be inherited from
(or the interface mirrored) by *concrete* equations class used by the
global ``pygmol.model.Model``.

The best introduction to the interface is a dive into the
`source code <https://github.com/hanicinecm/pygmol/blob/master/src/pygmol/abc.py>`_
of he ``Equations`` abstraction.

The ``pygmol.equations`` module provides a native
concrete equations class ``ElectronEnergyEquations(Equations)``, which is
currently hard-coded into the ``Model``. This will be discussed further, but first,
a short excursion into the ``Equations`` abstraction might be in order.

The ``Model`` class' main goal is to find a numerical solution to a system of ordinary
differential equations (ODE) that governs the behaviour of whatever plasma observables
we want the model to resolve, given their initial values.
Solving this *initial value problem* is offloaded to ``scipy.integrate.solve_ivp``
solver under the hood of the ``Model`` class.

The ODE system looks in general like this::

    dy / dt = f(t, y),

given an initial value::

    y(t0) = y0.

Here, ``t`` is an independent time variable and ``y(t)`` is an N-D vector-valued function
describing the *state vector y* as a function of time.

The ``solve_ivp`` expects (apart from the initial value of the state vector) the
right-hand-side of the ODE system, ``f(t, y)`` in the form of a vector-valued function.
The function signature must be ``f(t: float, y: n-d array) -> n-d array``.
The main responsibility of any concrete ``Equations`` subclass is to provide this
function for the solver. The is given by the ``Equations.ode_system_rhs`` attribute.
This will typically be a dynamically build ``@property`` constructed based on the
`Chemistry <doc_chemistry.rst>`_ and  `PlasmaParameters <doc_plasma_parameters.rst>`_
instances passed to the ``Model`` (and further on to ``Equations``).

Other abstract methods and properties are defined by the ``Equations`` abstraction,
which need to be overridden by a concrete subclass, such as ``final_solution_labels``,
``get_final_solution_values``, and ``get_y0_default`` (see the
`source code <https://github.com/hanicinecm/pygmol/blob/master/src/pygmol/abc.py>`_
for further information).


``ElectronEnergyEquations``
===========================

As mentioned above, ``pygmol`` provides a concrete ``Equations`` subclass, currently
used by the global model, the ``ElectronEnergyEquations`` inside the ``equations``
module
(`source <https://github.com/hanicinecm/pygmol/blob/master/src/pygmol/equations.py>`_).

This equations class resolves densities of all the heavy species in the chemistry (which
is kind of *given* for any global model of plasma chemistry), as well as the
*electron energy density* (hence the class name). It's ``ode_system_rhs`` works on the
state vector which is ``[n_1, n_2, ..., n_N, rho_e]`` (where ``N`` is the number of
heavy species in the ``chemistry``). Those are the quantities which are resolved by the
solver on a lower level. On a higher level, those are then converted to the values
of the ``final_solution_labels`` by the ``get_final_solution_values`` method of the
concrete ``Equations`` class.

A short demonstration should make everything clear.

Firstly, some maintenance is in order, to help with doc-testing the following code
snippets:

.. code-block:: pycon

    >>> import sys
    >>> from pathlib import Path
    >>> from numpy import array

    >>> # I will be importing some example chemistry and plasma parameters objects
    >>> # from the documentation directory (not part of the package), so it goes to the
    >>> # system path:
    >>> sys.path.append(str(Path(".") / "docs"))

Now, let us instantiate the ``Model`` with the same example chemistry and plasma
parameters as used int the `Model <doc_index.rst>`_ documentation:

.. code-block:: pycon

    >>> from pygmol.model import Model
    >>> from example_chemistry import argon_oxygen_chemistry
    >>> from example_plasma_parameters import argon_oxygen_plasma_parameters

    >>> model = Model(argon_oxygen_chemistry, argon_oxygen_plasma_parameters)

The ``ElectronEnergyEquations`` is instantiated as the ``equations`` attribute to the
model, wherever the ``run`` method is called:

.. code-block:: pycon

    >>> model.run()

    >>> type(model.equations)
    <class 'pygmol.equations.ElectronEnergyEquations'>

After a successful ``Model`` run, all the state vectors *y(t)* are stored as
the ``solution_primary`` attribute ``numpy.ndarray``, in this case with several
thousand rows (each for a single time step) and 25 columns (for densities of 24
heavy species and the electron energy density):

.. code-block:: pycon

    >>> type(model.solution_primary)
    <class 'numpy.ndarray'>

    >>> model.solution_primary.shape[1]
    25

Let us now see, how the ``equations`` object works:

.. code-block:: pycon

    >>> equations = model.equations

    >>> # the final (last) state vector:
    >>> y = array([2.37231337e+25, 2.10846582e+15, 8.57126911e+12, 2.01183854e+13,
    ...            1.45857406e+13, 1.71508621e+21, 5.65338119e+17, 3.08500654e+16,
    ...            2.23303476e+15, 3.00187971e+16, 2.12734223e+22, 9.12458352e+20,
    ...            6.28684944e+13, 6.42392705e+20, 1.44619515e+15, 1.75817604e+15,
    ...            8.73664736e+16, 3.17005006e+15, 2.45284068e+19, 1.16724944e+17,
    ...            2.02079492e+11, 6.20627690e+15, 9.37287931e+15, 6.95883253e+13,
    ...            1.49559385e+17])

    >>> # the time derivative of the final state vector, based on itself and the final
    >>> # time t = 0.015 s
    >>> ode_rhs = equations.ode_system_rhs
    >>> dy_over_dt = ode_rhs(t=0.015, y=y)  # this is the bit used by the solver

    >>> for val in dy_over_dt:
    ...     print(f"{val:.1e}")
    -9.8e+22
    -3.9e+16
    -3.9e+14
    ...
    -5.7e+17
    -8.1e+15
    2.4e+17


Finally, this is how the state vector *y* for each time step gets converted to the final
solution values (this happens under the hood of the ``Model.run`` method):

.. code-block:: pycon

    >>> for quantity in equations.final_solution_labels:
    ...     print(quantity)
    He
    He*
    ...
    O4+
    O4-
    e
    T_e
    T_n
    p
    P

    >>> for quantity_value in equations.get_final_solution_values(t=0.015, y=y):
    ...     print(f"{quantity_value:.1e}")
    2.4e+25
    2.1e+15
    ...
    9.4e+15
    7.0e+13
    6.0e+16
    1.7e+00
    3.0e+02
    1.0e+05
    3.0e-01
