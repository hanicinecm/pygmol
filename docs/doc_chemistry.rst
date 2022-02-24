***************************
``Chemistry`` Documentation
***************************

The ``pygmol.abc.Chemistry`` defines the abstract base class (ABC) to be inherited from
(or the interface mirrored by) by any *concrete* chemistry class instance needed as a
parameter to the global model.

The best way into understanding the interface expected by the global model is to read
the `source code <https://github.com/hanicinecm/pygmol/blob/master/src/pygmol/abc.py>`_
of the abstraction itself, or to see the example_chemistry_, compiled based on
Turner [1]_.

One note is in order: Any fast glance at the example_chemistry_ example makes it very clear that
this is a *terrible* format for defining static chemistry data. Instead, the intention
is that in real situation, the ``chemistry`` passed to the global model will be an instance
of much more powerful class (coded responsibly by the user either inheriting from
``pygmol.abc.Chemistry`` or mirroring the interface exactly), which will define the
attributes needed by the model as dynamic ``@properties``, rather than static class
attributes as used in the example. Such a user-defined class might hold instances of
classes representing species and reactions, it might have some species or reactions-oriented
*reduction* functionality built in, or it might be a class already in use in another modeling
framework, or a class representing a ``django`` model in an online chemistry database, just
augmented with the properties expected by ``pygmol``.

With that out of the way, let's look at the example_chemistry_ in some detail.

First, some maintenance is needed, so the following snippets can be doc-tested:

.. code-block:: pycon

    >>> import sys
    >>> from pathlib import Path
    >>> # add the docs directory into the system path:
    >>> sys.path.append(str(Path(".") / "docs"))

    >>> from example_chemistry import argon_oxygen_chemistry
    >>> import pygmol

Now, the example chemistry:

.. code-block:: pycon

    >>> type(argon_oxygen_chemistry)
    <class 'example_chemistry.ArO2Chemistry'>

    >>> isinstance(argon_oxygen_chemistry, pygmol.abc.Chemistry)
    True

    >>> for sp_id in argon_oxygen_chemistry.species_ids:
    ...     print(sp_id)
    He
    He*
    He+
    ...
    O3-
    O4+
    O4-

    >>> for r_str in argon_oxygen_chemistry.reactions_strings:
    ...     print(r_str)
    e + e + He+ -> He* + e
    e + e + He2+ -> He* + He + e
    e + He + He+ -> He* + He
    ...
    He2+ + O3 -> He + He + O+ + O2
    He2+ + O3- -> He + He + O3
    He2+ + O4- -> He + He + O2 + O2


The data required by the model for each species include, among others, charge, mass, or
sticking coefficients:

.. code-block:: pycon

    >>> argon_oxygen_chemistry.species_ids[7]
    'O(1S)'
    >>> argon_oxygen_chemistry.species_charges[7]
    0
    >>> argon_oxygen_chemistry.species_masses[7]
    16.0
    >>> argon_oxygen_chemistry.species_surface_sticking_coefficients[7]
    1

The ``species_surface_sticking_coefficients`` represent probabilities of each species
getting stuck to the surface (and removed from the system) *after it reaches the surface by diffusion*.
The surface processes are further encoded in the ``species_surface_return_matrix`` attribute,
a matrix of return coefficients with i-th row and j-th column specifying the "number"
of i-th species returned to plasma per a *single* j-th species *stuck* to surface.

The following snippet shows that each ``'He2*'`` species reaching the surface will stick
and return to the system as two ``'He'`` neutrals:


.. code-block:: pycon

    >>> argon_oxygen_chemistry.species_ids[3]
    'He2*'
    >>> argon_oxygen_chemistry.species_surface_sticking_coefficients[3]
    1
    >>> argon_oxygen_chemistry.species_ids[0]
    'He'
    >>> argon_oxygen_chemistry.species_surface_return_matrix[0][3]
    2

The reactions kinetics is parametrized by the Arrhenius formula (see the
`equations math`_). The following snipped shows, that the reaction

.. raw:: html

    O + O(<sup>1</sup>S) → O + O

has the rate coefficient of

.. raw:: html

    <i>k</i> = 2.5x10<sup>-17</sup> (<i>T</i><sub>n</sub>/300K)<sup>0</sup> exp(-300/<i>T</i><sub>n</sub>)

.. _example_chemistry: https://github.com/hanicinecm/pygmol/blob/master/docs/example_chemistry.py
.. _`equations math`: https://github.com/hanicinecm/pygmol/blob/master/docs/math.pdf

.. [1] Miles M Turner 2015 *Plasma Sources Sci. Technol.* **24** 035027