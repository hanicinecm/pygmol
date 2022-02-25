**********************************
``PlasmaParameters`` Documentation
**********************************

To ``pygmol.abc.PlasmaParameters`` is the abstract base class (ABC) defining the
interface of the *plasma parameters* input required by the global model.

The best way to introduce the plasma parameters interface is to read the
`source code <https://github.com/hanicinecm/pygmol/blob/master/src/pygmol/abc.py>`_
of the abstraction itself, or to see the annotated example_plasma_parameters_ for
a pulsed micro-jet plasma in Argon/Oxygen mixture, based on Turner [1]_.

One thing should be mention up front: It would be very impractical for each model run
to *actually* define a new concrete subclass of the ``PlasmaParameters`` ABC and passing
its instance into global model. Instead, a simple ``dict`` can be used mirroring exactly
the ABC's interface. The ``PlasmaParameters`` abstraction is there mostly to formally
define the form of the input data expected by the model, and is used behind the scenes
by the model to insure that the plasma parameters passed to the model are consistent and
physical (the ``pygmol.plasma_parameters.validate_plasma_parameters`` function also
comes into play here, also under the hood of the model) and that they follow the
interface.

That said, subclasses of ``abc.PlasmaParameters``, or any other objects defining the same
interface (such as an appropriate ``dataclass`` or ``namedtuple``), can easily be used
as inputs for the global model.

With that in mind, let us jump to some examples.

First, some maintenance is needed, so the following snippets can be doc-tested:

.. code-block:: pycon

    >>> import sys
    >>> from pathlib import Path
    >>> # add the docs directory into the system path:
    >>> sys.path.append(str(Path(".") / "docs"))

The following simple dictionary is exactly equivalent (as a model input) to the
example_plasma_parameters_:

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

Most of the parameters are self explanatory. The dimensions refer to an ICP plasma of
cylindrical shape.

The pressure parameter serves not only as the initial value (normalising the initial
densities), but also as the pressure *set-point*. The ODE system describing the plasma
behaviour includes a pressure correction term, compensating for species surface losses
or species number densities' changes due to dissociative and recombination processes
and the incoming feed flows. This pressure correction has a physical analogy in the
regulating gate valve to a pump which will be in use in almost any plasma processing
reactor.

The power absorbed by the plasma can be in general time dependent and
is defined as a series of points ``[t_power[i], power[i]]``. The example above shows
three pulses of 0.3 W, each 3 microseconds wide, followed by 3 microseconds of no power.
A time-constant power can also be specified, e.g. by ``"power": 300.0, "t_power": None``.

The ``feeds`` parameter specifies feed flows for certain species, all of which must be
among the ``chemistry.species_ids`` (where ``chemistry`` is another input passed to
the global model, see its `documentation page <doc_chemistry.rst>`_).

The ``temp_e`` and ``temp_n`` parameters describe the initial values for electron and
neutral temperatures. Depending on the ``Equations`` subclass, used by the model, (see
the `Equations <doc_equations.rst>`_ documentation page), these will or will not be
solved for, depending on if the equations resolve the electron energy density or neutral
energy density.

An finally, the ``t_end`` parameter is simply the simulation time, in this case covering
all three power pulses.

The global ``pygmol.model.Model`` can be instantiated with this dictionary and with a
``Chemistry`` instance, such as

.. code-block:: pycon

    >>> from pygmol.model import Model
    >>> from example_chemistry import argon_oxygen_chemistry

    >>> model = Model(
    ...     chemistry=argon_oxygen_chemistry,
    ...     plasma_params=argon_oxygen_params_dict
    ... )

The appropriate errors will be raised upon instantiation, if the plasma parameters are
not self-inconsistent:

.. code-block:: pycon

    >>> params = argon_oxygen_params_dict.copy()
    >>> params["t_power"] = params["t_power"][:-1]

    >>> Model(argon_oxygen_chemistry, params)
    Traceback (most recent call last):
      ..
    pygmol.plasma_parameters.PlasmaParametersValidationError: The 'power' and 't_power' attributes must have the same length!

unphysical:

.. code-block:: pycon

    >>> params = argon_oxygen_params_dict.copy()
    >>> params["radius"] = 0.0

    >>> Model(argon_oxygen_chemistry, params)
    Traceback (most recent call last):
      ..
    pygmol.plasma_parameters.PlasmaParametersValidationError: Plasma dimensions must be positive!

.. _example_plasma_parameters: https://github.com/hanicinecm/pygmol/blob/master/docs/example_plasma_parameters.py

not adhering to the interface required:

.. code-block:: pycon

    >>> params = argon_oxygen_params_dict.copy()
    >>> del(params["length"])

    >>> Model(argon_oxygen_chemistry, params)
    Traceback (most recent call last):
      ..
    TypeError: Can't instantiate abstract class PlasmaParametersFromDict with abstract methods length

or inconsistent with the chemistry:

.. code-block:: pycon

    >>> params = argon_oxygen_params_dict.copy()
    >>> params["feeds"]["N2"] = 42.0

    >>> Model(argon_oxygen_chemistry, params)
    Traceback (most recent call last):
      ..
    pygmol.plasma_parameters.PlasmaParametersValidationError: Feed gas species defined in the plasma parameters are inconsistent with the chemistry species ids!


.. [1] Miles M Turner 2015 *Plasma Sources Sci. Technol.* **24** 035027