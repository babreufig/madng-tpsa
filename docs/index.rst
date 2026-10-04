madng-tpsa
==========

``madng-tpsa`` provides Python bindings to the Generalised Truncated Power
Series Algebra (GTPSA) engine used by MAD-NG.

The package has four main building blocks:

* :class:`~madng_tpsa.descriptor.Descriptor` defines the algebraic space:
  variables, optional parameters, and truncation orders.
* :class:`~madng_tpsa.tpsa.Tpsa` represents a real truncated power series.
* :class:`~madng_tpsa.complex_tpsa.ComplexTpsa` represents the corresponding
  complex-coefficient series.
* :class:`~madng_tpsa.maps.TpsaMap` groups several series into a vector-valued
  Taylor map and provides composition, inversion, Lie-map operations, and other
  map algebra.

A TPSA stores Taylor coefficients. For a multi-index :math:`\alpha`,

.. math::

   f(\mathbf{x}) = \sum_{\alpha} c_{\alpha}\,\mathbf{x}^{\alpha},
   \qquad
   c_{\alpha} = \frac{1}{\alpha!}
   \left.\partial^{\alpha}f\right|_{\mathbf{x}=0}.

First-order coefficients therefore coincide with derivatives at the expansion
point; higher-order coefficients include the corresponding factorial.

Quick start
-----------

.. code-block:: python

   from madng_tpsa import Descriptor, TpsaMap

   descriptor = Descriptor(variables=['x', 'y'], order=4)
   x, y = descriptor.vars()

   f = 2 + 3 * x - y + 4 * x * y + x**3
   print(f.const_part)
   print(f.grad())
   print(f.monomial_coeffs())

   map_ = TpsaMap({
       'x': x + y + 0.2 * x**2,
       'y': y - 0.3 * x,
   })

   print(map_.const_part)
   print(map_.jacobian())

Composition follows mathematical ordering:

.. code-block:: text

   left @ right == left(right(z))

Hamiltonian and Lie operations assume descriptor variables are ordered in
canonical pairs,

.. math::

   (q_1,p_1,q_2,p_2,\ldots).

Descriptor parameters are not canonical variables.

.. toctree::
   :maxdepth: 2
   :caption: User guide
   :hidden:

   tpsa
   complex_tpsa
   tpsa_map
   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   api
