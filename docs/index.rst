********************
PyGMol Documentation
********************
**PyGMol** (the Python Global Model) is a simple-to-use 0D model of plasma chemistry.

At its current form, the ``pygmol`` package defines the
system of ordinary differential equations (ODE) which govern the plasma chemistry, and
solves them using the ``scipy.integrate.solve_ivp`` solver.
Quantities resolved presently by the ``pygmol`` model are the densities of all the
heavy species specified by the chemistry input, electron density, and electron
temperature (while heavy-species temperature is treated as a parameter to the model).

The equations being solved for by the model are documented in their full form in the
`equations math`_ document.

The following text describes their implementation as the ``pygmol`` package.


Package structure:
==================
To be added...


The ``Model``:
==============
This section shows the basic example of usage of the ``model.Model`` class for modeling
plasma chemistry.

Firstly, some maintenance is in order, to help with doc-testing the following code
snippets. This following code block is not normally necessary:

.. code-block:: pycon

    >>> import sys
    >>> from pathlib import Path
    >>> # I will be importing some example chemistry and plasma parameters objects
    >>> # from the documentation directory (not part of the package), so it goes to the
    >>> # system path:
    >>> sys.path.append(str(Path(".") / "docs"))
    >>> # I have prepared a function which prints pandas.DataFrames (object of choice
    >>> # for the model outputs) in a controlled way. This helps with doc-testing:
    >>> from utils import print_dataframe


The ``Model`` class takes two inputs:

- ``chemistry`` - an instance of any concrete subclass of the ``pygmol.abc.Chemistry``
  abstract base class

- ``plasma_parameters`` - an instance of any concrete subclass of the
  ``pygmol.abc.PlasmaParameters`` abstract base class.

For the purpose of this demonstration, I have prepared an example argon_oxygen_chemistry_
describing an Argon/Oxygen pulsed plasma, with the example argon_oxygen_plasma_parameters_.

Both inputs are based on Turner [1]_.
Again, these are not part of the ``pygmol`` package, but rather only live for this
documentation:

.. code-block:: pycon

    >>> from example_chemistry import argon_oxygen_chemistry
    >>> from example_plasma_parameters import argon_oxygen_plasma_parameters


In fact, the ``Model`` class constructor accepts also ``dict`` as both parameters, if
they adhere to the exact interface defined by the abstract ``Chemistry`` and
``PlasmaParameters`` classes. So the following ``dict`` input is equivalent to
``argon_oxygen_plasma_parameters``:

.. code-block:: pycon

    >>> argon_oxygen_params_dict = {
    ...     "radius": 0.000564,  # [m]
    ...     "length": 0.03,  # [m]
    ...     "pressure": 1e5,  # [Pa]
    ...     "power": (0.3, 0.3, 0, 0, 0.3, 0.3, 0, 0, 0.3, 0.3),  # [W]
    ...     "t_power": (0, 0.003, 0.003, 0.006, 0.006, 0.009, 0.009, 0.012, 0.012, 0.015),  # [s]
    ...     "feeds": {"O2": 0.3, "He": 300.0},  # [sccm]
    ...     "temp_e": 1.0,  # [eV]
    ...     "temp_n": 305.0,  # [K]
    ...     "t_end": 0.015  # [s]
    ... }


Both inputs to the ``Model`` class have their own documentation pages explaining them in
detail: `Chemistry <chemistry.rst>`_, `PlasmaParameters <plasma_parameters.rst>`_.

One note is in order: Any fast glance at the argon_oxygen_chemistry_ makes it very clear that
this is a *terrible* format for defining static chemistry data. Instead, the intention
is that in real situation, the ``chemistry`` passed to the ``Model`` will be an instance
of much more powerful class (coded responsibly by the user either inheriting from
``pygmol.abc.Chemistry`` or mirroring the interface exactly), which defines the
attributes needed by the model as dynamic ``@properties``, rather than static class
attributes as used in the example. Such a user-defined class might hold instances of
classes representing species and reactions, it might have some species or reactions-oriented
*reduction* functionality built in, or it might be a class already in use in another modeling
framework, or a class representing a ``django`` model in an online chemistry database, just
augmented with the properties expected by ``pygmol``.

With that out of the way, let us instantiate our model:

.. code-block:: pycon

    >>> from pygmol.model import Model

    >>> model = Model(
    ...     chemistry=argon_oxygen_chemistry,
    ...     plasma_params=argon_oxygen_params_dict
    ... )

and run it (hopefully with success):

.. code-block:: pycon

    >>> model.run()

    >>> model.success()
    True

Note: If the solution is *not* successful, the ``ModelSolutionError`` will be raised and
all the info returned by the ``scipy.integrate.solve_ivp`` will be stored under
``model.solution_raw``.


Solution
--------

In the case of a successful solution, we can access it (in the final, post-processed
form) as a ``pandas.DataFrame`` (index of the dataframe is irrelevant and not printed
out):

.. code-block:: pycon

    >>> solution = model.get_solution()
    >>> print_dataframe(solution)
             t      He     He*     He+    He2*  ...       e     T_e     T_n       p       P
       0.0e+00 2.4e+25 2.4e+10 2.0e+10 2.4e+10  ... 2.4e+10 1.0e+00 3.0e+02 1.0e+05 3.0e-01
       2.9e-15 2.4e+25 2.4e+10 2.0e+10 2.4e+10  ... 2.4e+10 6.0e+00 3.0e+02 1.0e+05 3.0e-01
       5.7e-15 2.4e+25 2.4e+10 2.0e+10 2.4e+10  ... 2.4e+10 1.1e+01 3.0e+02 1.0e+05 3.0e-01
       2.5e-14 2.4e+25 2.4e+10 2.0e+10 2.4e+10  ... 2.4e+10 4.5e+01 3.0e+02 1.0e+05 3.0e-01
       4.5e-14 2.4e+25 2.4e+10 2.0e+10 2.4e+10  ... 2.4e+10 7.8e+01 3.0e+02 1.0e+05 3.0e-01
    ...
       1.4e-02 2.4e+25 2.1e+15 8.7e+12 2.0e+13  ... 5.9e+16 1.7e+00 3.0e+02 1.0e+05 3.0e-01
       1.4e-02 2.4e+25 2.1e+15 8.7e+12 2.0e+13  ... 5.9e+16 1.7e+00 3.0e+02 1.0e+05 3.0e-01
       1.5e-02 2.4e+25 2.1e+15 8.6e+12 2.0e+13  ... 5.9e+16 1.7e+00 3.0e+02 1.0e+05 3.0e-01
       1.5e-02 2.4e+25 2.1e+15 8.6e+12 2.0e+13  ... 6.0e+16 1.7e+00 3.0e+02 1.0e+05 3.0e-01
       1.5e-02 2.4e+25 2.1e+15 8.6e+12 2.0e+13  ... 6.0e+16 1.7e+00 3.0e+02 1.0e+05 3.0e-01
    ...

The columns of the solution dataframe are controlled by the ``Equations`` instance used
by the ``Model``, see the `Equations <equations.rst>`_ documentation. For the native
``ElectronEnergyEquations``
(`source code <https://github.com/hanicinecm/pygmol/blob/master/src/pygmol/equations.py>`_)
class, those are (apart time ``"t"``) the heavy
species names (``chemistry.species_ids``) for their densities in [SI], ``"e"`` for
the electron density, and ``["T_e", "T_n", "p", "P"]`` for electron and neutral
temperatures (in eV, and K respectively), pressure [Pa], and finally power [W].
The neutral temperature is treated as a constant parameter by ``ElectronEnergyEquations``
and stays therefore at it's initial value as defined by the ``plasma_parameters`` passed
to the ``Model``.

A number of additional data extracted from a successful solution are provided by the
``Model``:

Reaction rates
--------------
Reaction rates in time (in m-3/s) of all the reactions specified by the ``chemistry``,
identified by their IDs as the dataframe columns (``chemistry.reactions_ids``).
The index of the dataframe is irrelevant (and not printed out).

.. code-block:: pycon

    >>> reaction_rates = model.get_reaction_rates()
    >>> print_dataframe(reaction_rates)
             t       1       2       3       4  ...     369     370     371     372     373
       0.0e+00 1.9e-08 1.8e-07 2.8e+07 2.8e+07  ... 2.1e+06 1.5e+07 7.5e+05 6.2e+07 6.7e+07
       2.9e-15 6.1e-12 1.4e-10 3.2e+05 3.2e+05  ... 2.1e+06 1.5e+07 7.5e+05 6.2e+07 6.7e+07
       5.7e-15 4.0e-13 1.2e-11 7.0e+04 7.0e+04  ... 2.1e+06 1.5e+07 7.5e+05 6.2e+07 6.7e+07
       2.5e-14 7.2e-16 4.5e-14 2.1e+03 2.1e+03  ... 2.1e+06 1.5e+07 7.5e+05 6.2e+07 6.7e+07
       4.5e-14 6.3e-17 5.1e-15 5.4e+02 5.3e+02  ... 2.1e+06 1.5e+07 7.5e+05 6.2e+07 6.7e+07
    ...
       1.4e-02 5.2e+06 1.1e+08 8.4e+15 1.4e+16  ... 4.0e+14 1.5e+15 5.8e+17 1.3e+16 1.6e+14
       1.4e-02 5.2e+06 1.1e+08 8.4e+15 1.4e+16  ... 3.9e+14 1.5e+15 5.8e+17 1.2e+16 1.6e+14
       1.5e-02 5.2e+06 1.1e+08 8.4e+15 1.4e+16  ... 3.8e+14 1.5e+15 5.7e+17 1.2e+16 1.5e+14
       1.5e-02 5.2e+06 1.1e+08 8.3e+15 1.4e+16  ... 3.7e+14 1.5e+15 5.7e+17 1.2e+16 1.5e+14
       1.5e-02 5.2e+06 1.1e+08 8.3e+15 1.4e+16  ... 3.7e+14 1.5e+15 5.7e+17 1.2e+16 1.5e+14
    ...

Rates of change of species densities
------------------------------------
The rates of change of species densities (in m-3/s) can be accessed for any given time
``t`` by the ``get_rates_matrix_total`` method, and will show the values for each heavy
species (excluding electrons and the arbitrary heavy species ``"M"``), and per each
volumetric reaction or surface reaction process and for the closest time frame to ``t``.
This time, the dataframe is indexed by the human-readable reaction strings, if supplied
by the ``chemistry`` (``chemistry.reactions_strings``).

.. code-block:: pycon

    >>> rates_matrix = model.get_rates_matrix_total(t=0.015)  # at the end of the time domain
    >>> print_dataframe(rates_matrix, max_cols=6, hide_index=False)
                                              He     He*     He+  ...     O3-     O4+     O4-
    He + O2(v) -> He + O2 (R272)         0.0e+00 0.0e+00 0.0e+00  ... 0.0e+00 0.0e+00 0.0e+00
    O(1D) + O2 -> O + O2(b1Su+) (R112)   0.0e+00 0.0e+00 0.0e+00  ... 0.0e+00 0.0e+00 0.0e+00
    O2(b1Su+) + O3 -> O + O2 + O2 (R137) 0.0e+00 0.0e+00 0.0e+00  ... 0.0e+00 0.0e+00 0.0e+00
    e + O2 -> e + O + O(1D) (R22)        0.0e+00 0.0e+00 0.0e+00  ... 0.0e+00 0.0e+00 0.0e+00
    e + O2 -> e + O2(a1Du) (R32)         0.0e+00 0.0e+00 0.0e+00  ... 0.0e+00 0.0e+00 0.0e+00
    ...
    e + He -> e + He (R5)                0.0e+00 0.0e+00 0.0e+00  ... 0.0e+00 0.0e+00 0.0e+00
    e + O2(b1Su+) -> e + O2(b1Su+) (R61) 0.0e+00 0.0e+00 0.0e+00  ... 0.0e+00 0.0e+00 0.0e+00
    e + O2(b1Su+) -> e + O2(b1Su+) (R62) 0.0e+00 0.0e+00 0.0e+00  ... 0.0e+00 0.0e+00 0.0e+00
    e + O2(b1Su+) -> e + O2(b1Su+) (R69) 0.0e+00 0.0e+00 0.0e+00  ... 0.0e+00 0.0e+00 0.0e+00
    e + O2(a1Du) -> e + O2(a1Du) (R43)   0.0e+00 0.0e+00 0.0e+00  ... 0.0e+00 0.0e+00 0.0e+00
    ...

Admittedly, not much is happening in the previous example, so lets limit the scope to
just 3 selected species and just the processes which affect their densities:

.. code-block:: pycon

    >>> selected = rates_matrix[["O", "O2(a1Du)", "O3"]]
    >>> selected = selected.loc[(selected!=0).any(axis=1)]
    >>> print_dataframe(selected, hide_index=False)
                                               O  O2(a1Du)       O3
    O(1D) + O2 -> O + O2(b1Su+) (R112)   3.8e+23   0.0e+00  0.0e+00
    O2(b1Su+) + O3 -> O + O2 + O2 (R137) 2.4e+23   0.0e+00 -2.4e+23
    e + O2 -> e + O + O(1D) (R22)        3.3e+23   0.0e+00  0.0e+00
    e + O2 -> e + O2(a1Du) (R32)         0.0e+00   4.3e+23  0.0e+00
    O2(a1Du) + surf. -> surf. + O2       0.0e+00  -2.9e+23  0.0e+00
    ...                                      ...       ...      ...
    e + O+ + O2 -> O + O2 (R147)         2.6e+10   0.0e+00  0.0e+00
    O + O3 -> O + O + O2 (R108)          3.0e+09   0.0e+00 -3.0e+09
    e + e + O+ -> e + O (R142)           1.6e+09   0.0e+00  0.0e+00
    O3 + O3 -> O + O2 + O3 (R140)        5.2e+07   0.0e+00 -5.2e+07
    O2 + O2 -> O + O + O2 (R124)         5.7e-54   0.0e+00  0.0e+00
    ...

General diagnostics
-------------------
Finally, a general diagnostics method is provided, returning the time dependence of any
intermediate result defined by the concrete ``Equations`` class used by the model.
For example, the *Debye length* can be requested in time by

    >>> debye_length = model.diagnose("debye_length")
    >>> print_dataframe(debye_length)
             t  debye_length
       0.0e+00       4.8e-02
       2.9e-15       1.2e-01
       5.7e-15       1.6e-01
       2.5e-14       3.2e-01
       4.5e-14       4.2e-01
    ...
       1.4e-02       3.9e-05
       1.4e-02       3.9e-05
       1.5e-02       3.9e-05
       1.5e-02       3.9e-05
       1.5e-02       3.9e-05
    ...

assuming that ``model.equations`` has the ``get_debye_length`` method accepting the
state vector *y* (see `Equations <equations.rst>`_).

Other functionality
-------------------
The examples above only cover the selected functionality of the ``Model``. Other
useful methods might include

- ``get_surface_loss_rates``, ``get_rates_matrix_volume``, ``get_rates_matrix_surface``,
  ``get_{*}_final``.

And, of course, reading through the source code will provide much more insight into the
package than any documentation ever will.

So dive in ...


.. _`equations math`: https://github.com/hanicinecm/pygmol/blob/master/docs/math.pdf
.. _argon_oxygen_chemistry: https://github.com/hanicinecm/pygmol/blob/master/docs/example_chemistry.py
.. _argon_oxygen_plasma_parameters: https://github.com/hanicinecm/pygmol/blob/master/docs/example_plasma_parameters.py


.. [1] Miles M Turner 2015 *Plasma Sources Sci. Technol.* **24** 035027