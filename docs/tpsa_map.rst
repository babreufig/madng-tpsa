TpsaMap
=======

:class:`~madng_tpsa.maps.TpsaMap` is a lightweight vector-valued TPSA map. All
components share one descriptor and, after construction, are either all real
TPSAs or all complex TPSAs. Mixed input is promoted to a complex map.

Construction
------------

.. code-block:: python

   descriptor = Descriptor(variables=['x', 'px'], order=4)
   x, px = descriptor.vars()

   map_ = TpsaMap({
       'x': x + px + 0.2*x**2,
       'px': px - 0.3*x,
   })

A map can contain fewer outputs than descriptor variables, but operations which
perform a complete substitution require a full map.

Identity and complex promotion
------------------------------

.. code-block:: python

   identity = TpsaMap.identity(descriptor)
   complex_identity = identity.to_complex()

Coordinate names and variable labels
------------------------------------

``coord_names`` label map outputs. ``descriptor.var_labels`` label independent
input variables. Algebra is positional; names do not reorder components.

Inspection
----------

.. code-block:: python

   map_[0]
   map_['x']
   map_.x
   map_.const_part
   map_.jacobian()
   map_.param_jacobian()
   map_.sensitivity('x', 'kqf')
   map_.monomial_coeffs('x')
   map_.coefficient('x', (2, 0))

In-place setters
----------------

.. code-block:: python

   map_.set_const_part([1.0, -2.0])
   map_.set_jacobian([[1.0, 2.0], [3.0, 4.0]])
   map_.set_param_jacobian([[0.1, 0.2], [0.3, 0.4]])
   map_.set_coefficient('x', (2, 0), 0.25)

Variable and parameter first-order blocks are independent.

Evaluation
----------

.. code-block:: python

   map_.evaluate([1e-3, 2e-3])
   map_.evaluate([1e-3, 2e-3], parameters=[0.1, -0.2])
   map_.jacobian([1e-3, 2e-3], parameters=[0.1, -0.2])

Composition
-----------

``left @ right`` means ``left(right(z))``.

.. code-block:: python

   result = left @ right
   result = left.compose(right)

Descriptor parameters are appended automatically as identity substitutions.

Scalar pullback
---------------

For a scalar TPSA :math:`f` and a full map :math:`M`, the pullback is
:math:`f\circ M`:

.. code-block:: python

   transformed = map_.pullback(f)

or

.. code-block:: python

   from madng_tpsa import pullback
   transformed = pullback(f, map_)

This is especially useful in homological equations and normal-form work.

Translation and inversion
-------------------------

.. code-block:: python

   shifted = map_.translate([1.0, -2.0])
   inverse = centred_map.inverse()
   partial = centred_map.partial_inverse([True, False])

The Python API requires a zero constant part before full or partial inversion.
Re-centre an affine map first.

Order manipulation
------------------

.. code-block:: python

   cubic = map_.homogeneous(3)
   through_cubic = map_.truncate(3)
   without_quadratic = map_.clear_order(2)

Hamiltonian and Lie operations
------------------------------

Hamiltonian operations assume descriptor variables are ordered in canonical
pairs ``(q1, p1, q2, p2, ...)``. Variable names are not inspected to determine
canonicality.

``vector_to_field(f)`` mirrors MAD-NG ``vec2fld`` and returns
:math:`G=-J\nabla f`. ``field_to_vector(G)`` performs the reverse conversion for
a Hamiltonian field. The additive constant of a scalar generator cannot be
recovered from its field and is fixed to zero.

Exponential Poisson maps
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   descriptor = Descriptor(variables=['q', 'p'], order=5)
   q, p = descriptor.vars()
   f = q**3 / 3

   lie_map = TpsaMap.identity(descriptor).exp_poisson(f)

With the MAD-NG high-level convention this gives, through the retained order,
:math:`q'=q` and :math:`p'=p-q^2`.

Map logarithms
~~~~~~~~~~~~~~

``log_poisson()`` returns the logarithmic vector field.
``log_generator()`` converts it to a scalar generator. The optional
``initial_guess`` is an algorithmic initial guess for MAD-NG's logarithm
iteration, not a physical reference map.

Truncation and symplecticity
----------------------------

A map available through order :math:`N` can generally only be verified to be
symplectic through order :math:`N-1`: taking derivatives lowers polynomial
order, so the order-:math:`N` part of a Poisson bracket can depend on unknown
order-:math:`N+1` map coefficients.

Normal-form scope
-----------------

``TpsaMap`` provides the generic algebraic building blocks for nonlinear
normal-form calculations: composition, inversion, homogeneous extraction,
pullback, real/complex promotion, coefficient access, Lie exponentials and
logarithms, and parameter dependence.

Eigenvector normalisation, resonance classification, homological-equation
solving, action-angle interpretation, detuning extraction and RDT extraction
belong in a higher-level normal-form layer.

Subclass behaviour
------------------

Generic map operations return plain :class:`TpsaMap` objects. Arbitrary
subclasses may carry external state whose transformation under map algebra is
undefined.

API reference
-------------

.. autoclass:: madng_tpsa.maps.TpsaMap
   :members:

Functional map API
------------------

.. autofunction:: madng_tpsa.maps.compose
.. autofunction:: madng_tpsa.maps.pullback
.. autofunction:: madng_tpsa.maps.inverse
.. autofunction:: madng_tpsa.maps.partial_inverse
.. autofunction:: madng_tpsa.maps.evaluate
.. autofunction:: madng_tpsa.maps.translate
.. autofunction:: madng_tpsa.maps.vector_to_field
.. autofunction:: madng_tpsa.maps.field_to_vector
.. autofunction:: madng_tpsa.maps.lie_bracket
.. autofunction:: madng_tpsa.maps.exp_poisson
.. autofunction:: madng_tpsa.maps.log_poisson
.. autofunction:: madng_tpsa.maps.log_generator
