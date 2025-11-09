QEP Modeling with Derivatives
===============================================

Here are some examples of QEP modeling with derivative information in QPyTorch.


QEPs with Derivatives
----------------------

Derivative information can be used by QEPs to accelerate Bayesian optimization, and solve PDEs.
See the `1D derivatives QEP example`_,  the `2D derivatives QEP example`_, and the `variational QEP with 2nd order derivatives`_ 
for examples on using QEPs with derivative information.

Due to the enhanced capability to handle inhomogeneity, QEP can be used to solve and learn partial differential equations (PDEs).
We demonstrate in [Yu and Lan (2025)](https://openreview.net/pdf?id=WjhS0EpJH7) QEP model with q=1 as a superior probabilistic compared with GP.
See this `solving PDE with QEP`_ example for a demonstration with Burgers'' equation.

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   Simple_QEP_Regression_Derivative_Information_1d.ipynb
   Simple_QEP_Regression_Derivative_Information_2d.ipynb
   SVQEP_Regression_Derivative_Information_2d.ipynb
   Solve_PDE_with_QEP.ipynb


.. _1D derivatives QEP example:
  Simple_QEP_Regression_Derivative_Information_1d.ipynb

.. _2D derivatives QEP example:
  Simple_QEP_Regression_Derivative_Information_2d.ipynb

.. _variational QEP with 2nd order derivatives:
  SVQEP_Regression_Derivative_Information_2d.ipynb

.. _solving PDE with QEP:
  Solve_PDE_with_QEP.ipynb
