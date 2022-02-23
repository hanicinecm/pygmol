********************
PyGMol Documentation
********************
**PyGMol** (the Python Global Model) is a simple-to-use 0D model, or
global model, for modeling plasma chemistry.

At its current form, the ``pygmol`` package defines the
system of ordinary differential equations (ODE) which govern the plasma chemistry, and
solves them using the ``scipy.integrate.solve_ivp`` solver.
Quantities resolved presently by the ``pygmol`` model are the densities of all the
heavy species specified by the chemistry input, electron density, and electron
temperature (while heavy-species temperature is treated as a parameter to the model).

The equations being solved for by the model are documented in their full form in the
equations_ document. This documentation covers the inner workings of the package, its
structure, and adds some useful examples of usage.


Package structure:
==================
To be added...


The ``Model``:
==============
To be added...

.. code-block:: pycon

    >>> import sys
    >>> from pathlib import Path
    >>> sys.path.append(str(Path(".") / "docs"))
    >>> import pandas as pd
    >>> pd.options.display.float_format = "{:.1e}".format
    >>> pd.set_option("display.max_rows", 10)
    >>> pd.set_option("display.max_columns", 10)
    >>> pd.set_option("display.expand_frame_repr", False)
    >>> def print_table(df: pd.DataFrame):
    ...     df = df.copy()
    ...     df.index = [""] * len(df)
    ...     print(df)

    >>> from pygmol.model import Model

    >>> from example_chemistry import argon_oxygen_chemistry
    >>> from example_plasma_parameters import argon_oxygen_plasma_parameters

    >>> model = Model(argon_oxygen_chemistry, argon_oxygen_plasma_parameters)

    >>> model.run()

    >>> model.success()
    True

    >>> solution = model.get_solution()
    >>> print_table(solution)
    ...
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

    >>> reaction_rates = model.get_reaction_rates()
    >>> print_table(reaction_rates)
    ...
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

    >>> rates_matrix = model.get_rates_matrix_total()
    >>> rates_matrix
    ...
                                              He     He*     He+    He2*    He2+  ...   O3(v)     O3+     O3-     O4+     O4-
    He + O2(v) -> He + O2 (R272)         0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00  ... 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00
    O(1D) + O2 -> O + O2(b1Su+) (R112)   0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00  ... 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00
    O2(b1Su+) + O3 -> O + O2 + O2 (R137) 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00  ... 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00
    e + O2 -> e + O + O(1D) (R22)        0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00  ... 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00
    e + O2 -> e + O2(a1Du) (R32)         0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00  ... 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00
    ...                                      ...     ...     ...     ...     ...  ...     ...     ...     ...     ...     ...
    e + He -> e + He (R5)                0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00  ... 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00
    e + O2(b1Su+) -> e + O2(b1Su+) (R61) 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00  ... 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00
    e + O2(b1Su+) -> e + O2(b1Su+) (R62) 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00  ... 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00
    e + O2(b1Su+) -> e + O2(b1Su+) (R69) 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00  ... 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00
    e + O2(a1Du) -> e + O2(a1Du) (R43)   0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00  ... 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00
    ...

    >>> selected_rates_matrix = rates_matrix[["O", "O2(a1Du)", "O3"]]
    >>> selected_rates_matrix.loc[(selected_rates_matrix!=0).any(axis=1)]
    ...
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

    >>> debye_length = model.diagnose("debye_length")
    >>> print_table(debye_length)
    ...
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


.. _equations: https://github.com/hanicinecm/pygmol/blob/master/docs/equations.pdf