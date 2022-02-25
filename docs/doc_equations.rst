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

