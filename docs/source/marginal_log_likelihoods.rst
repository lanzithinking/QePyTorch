.. role:: hidden
    :class: hidden-section

qpytorch.mlls
===================================

These are modules to compute (or approximate/bound) the marginal log likelihood
(MLL) of the GP/QEP model when applied to data.  I.e., given a GP (QEP) :math:`f \sim
\mathcal{GP}(\mu, K)` (:math:`f \sim \mathcal{QEP}(\mu, K)`), 
and data :math:`\mathbf X, \mathbf y`, these modules compute/approximate

.. math::

   \begin{equation*}
      \mathcal{L} = p_f(\mathbf y \! \mid \! \mathbf X)
      = \int p \left( \mathbf y \! \mid \! f(\mathbf X) \right) \: p(f(\mathbf X) \! \mid \! \mathbf X) \: d f
   \end{equation*}

This is computed exactly when the GP/QEP inference is computed exactly (e.g. regression w/ a Gaussian/Q-exponential likelihood).
It is approximated/bounded for GP/QEP models that use approximate inference.

These models are typically used as the "loss" functions for GP/QEP models (though note that the output of
these functions must be negated for optimization).
All of them are ported from GPyTorch excepted those :hlmod:`highlighted`.

.. automodule:: qpytorch.mlls
.. currentmodule:: qpytorch.mlls


Exact Inference
-----------------------------

These are MLLs for use with :obj:`~qpytorch.models.ExactGP` (:obj:`~qpytorch.models.ExactQEP`) modules. They compute the MLL exactly.

:hidden:`ExactMarginalLogLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ExactMarginalLogLikelihood
   :members:

:hidden:`LeaveOneOutPseudoLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LeaveOneOutPseudoLikelihood
   :members:


Approximate Inference
-----------------------------------

These are MLLs for use with :obj:`~qpytorch.models.ApproximateGP` (obj:`~qpytorch.models.ApproximateQEP`) modules. They are designed for
when exact inference is intractable (either when the likelihood is non-Gaussian likelihood, or when
there is too much data for an ExactGP/ExactQEP model).

:hidden:`VariationalELBO`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: VariationalELBO
   :members:

:hidden:`PredictiveLogLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PredictiveLogLikelihood
   :members:

:hidden:`GammaRobustVariationalELBO`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GammaRobustVariationalELBO
   :members:

:hidden:`DeepApproximateMLL`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DeepApproximateMLL
   :members:


Modifications to Objective Functions
---------------------------------------

.. autoclass:: AddedLossTerm
   :members:

:hidden:`InducingPointKernelAddedLossTerm`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: InducingPointKernelAddedLossTerm
   :members:

:hidden:`KLGaussianAddedLossTerm`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: KLGaussianAddedLossTerm
   :members:

:hlmod:`KLQExponentialAddedLossTerm`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: KLQExponentialAddedLossTerm
   :members:
