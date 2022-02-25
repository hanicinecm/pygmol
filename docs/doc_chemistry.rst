***************************
``Chemistry`` Documentation
***************************

The ``pygmol.abc.Chemistry`` is the abstract base class (ABC) to be inherited from
(or the interface mirrored) by any *concrete* chemistry class instance needed as a
parameter to the global model.

The best way into understanding the interface expected by the global model is to read
the `source code <https://github.com/hanicinecm/pygmol/blob/master/src/pygmol/abc.py>`_
of the abstraction itself, or to see the annotated example_chemistry_, compiled based on
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

    >>> from example_chemistry import ArO2Chemistry
    >>> argon_oxygen_chemistry = ArO2Chemistry()
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
`equations math`_). The following snippet shows, that the reaction (id 99)

.. raw:: html

    O + O(<sup>1</sup>S) → O + O

has the rate coefficient (in SI) of

.. raw:: html

    <i>k<sub>99</sub></i> = 2.5x10<sup>-17</sup> (<i>T</i><sub>n</sub>/300K)<sup>0</sup> exp(-300/<i>T</i><sub>n</sub>)<br>
    <i>k<sub>99</sub></i> = 2.5x10<sup>-17</sup> exp(-300/<i>T</i><sub>n</sub>)

.. code-block:: pycon

    >>> argon_oxygen_chemistry.reactions_ids[98]
    99
    >>> argon_oxygen_chemistry.reactions_strings[98]
    'O + O(1S) -> O + O'
    >>> argon_oxygen_chemistry.reactions_arrh_a[98]
    2.5e-17
    >>> argon_oxygen_chemistry.reactions_arrh_b[98]
    0
    >>> argon_oxygen_chemistry.reactions_arrh_c[98]
    300

Several other properties of the reactions need to be given in the chemistry, such as
electron energy losses, or the boolean array flagging all the elastic collisions, see the
source code.

The species and the reactions are related via the ``reactions_species_stoichiomatrix``
and the ``reactions_electron_stoich`` parameters (one of each for left-hand and
right-hand-sides of reactions). The following shows, that the first
reaction has two electrons and He+ as reactants, and one electron and He* as products:

.. code-block:: pycon

    >>> argon_oxygen_chemistry.reactions_strings[0]
    'e + e + He+ -> He* + e'
    >>> argon_oxygen_chemistry.species_ids[1]
    'He*'
    >>> argon_oxygen_chemistry.species_ids[2]
    'He+'

    >>> # left-hand-side stoichiometries:
    >>> argon_oxygen_chemistry.reactions_species_stoichiomatrix_lhs[0][2]
    1
    >>> argon_oxygen_chemistry.reactions_electron_stoich_lhs[0]
    2

    >>> # right-hand-side stoichiometries:
    >>> argon_oxygen_chemistry.reactions_species_stoichiomatrix_rhs[0][1]
    1
    >>> argon_oxygen_chemistry.reactions_electron_stoich_rhs[0]
    1


Finally, the ``chemistry`` module also provides a function for validation of ``Chemistry``
instances (this is used under the hood by the global model).

.. code-block:: pycon

    >>> from pygmol.chemistry import validate_chemistry
    >>> validate_chemistry(chemistry=argon_oxygen_chemistry)

This will raise an appropriate custom error if the attributes/properties of the
``chemistry`` instance passed is inconsistent in some way, e.g. if length of the species
attributes do not match:

.. code-block:: pycon

    >>> len(argon_oxygen_chemistry.species_ids)
    24
    >>> len(argon_oxygen_chemistry.species_charges)
    24

    >>> argon_oxygen_chemistry.species_charges = argon_oxygen_chemistry.species_charges[:-1]
    >>> len(argon_oxygen_chemistry.species_charges)
    23

    >>> validate_chemistry(chemistry=argon_oxygen_chemistry)
    Traceback (most recent call last):
      ...
    pygmol.chemistry.ChemistryValidationError: All the attributes describing species need to have the same dimension!


As ever, reading through the source code will provide much more insight into the
package than any documentation ever will. I have tried my best to keep all the docstrings
as informative as possible and up-to-date.

So dive in ...

.. _example_chemistry: https://github.com/hanicinecm/pygmol/blob/master/docs/example_chemistry.py
.. _`equations math`: https://github.com/hanicinecm/pygmol/blob/master/docs/math.pdf

.. [1] Miles M Turner 2015 *Plasma Sources Sci. Technol.* **24** 035027