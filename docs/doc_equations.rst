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

Here, ``t`` is an independent time variable and ``y(t)`` is an N-D vector-values function
describing the *state vector y* as a function of time.

The ``solve_ivp`` expects (apart the initial value of the state vector) the
right-hand-side of the ODE system, ``f(t, y)`` in the form of a vector-valued function.
The function signature must be ``f(t: float, y: n-d array) -> n-d array``.
