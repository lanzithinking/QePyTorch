.. role:: hidden
    :class: hidden-section

qpytorch.likelihoods
===================================

.. automodule:: qpytorch.likelihoods
.. currentmodule:: qpytorch.likelihoods


QPyTorch likelihood objects are ported from GPyTorch excepted those :hlmod:`highlighted`.


Likelihood
--------------------

.. autoclass:: Likelihood
   :special-members: __call__
   :members:


One-Dimensional Likelihoods
-----------------------------

Likelihoods for GPs/QEPs that are distributions of scalar functions.
(I.e. for a specific :math:`\mathbf x` we expect that :math:`f(\mathbf x) \in \mathbb{R}`.)

One-dimensional likelihoods should extend :obj:`qpytorch.likelihoods._OneDimensionalLikelihood` to
reduce the variance when computing approximate GP/QEP objective functions.
(Variance reduction is accomplished by using 1D Gauss-Hermite quadrature rather than MC-integration).


:hidden:`GaussianLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GaussianLikelihood
   :members:

:hidden:`GaussianLikelihoodWithMissingObs`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GaussianLikelihoodWithMissingObs
   :members:

:hidden:`FixedNoiseGaussianLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FixedNoiseGaussianLikelihood
   :members:


:hidden:`DirichletClassificationLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DirichletClassificationLikelihood
   :members:


:hlmod:`QExponentialLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: QExponentialLikelihood
   :members:

:hlmod:`QExponentialLikelihoodWithMissingObs`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: QExponentialLikelihoodWithMissingObs
   :members:

:hlmod:`FixedNoiseQExponentialLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FixedNoiseQExponentialLikelihood
   :members:


:hlmod:`QExponentialDirichletClassificationLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: QExponentialDirichletClassificationLikelihood
   :members:


:hidden:`BernoulliLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BernoulliLikelihood
   :members:


:hidden:`BetaLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BetaLikelihood
   :members:


:hidden:`LaplaceLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LaplaceLikelihood
   :members:


:hidden:`StudentTLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: StudentTLikelihood
   :members:


Multi-Dimensional Likelihoods
-----------------------------

Likelihoods for GPs/QEPs that are distributions of vector-valued functions.
(I.e. for a specific :math:`\mathbf x` we expect that :math:`f(\mathbf x) \in \mathbb{R}^t`,
where :math:`t` is the number of output dimensions.)


:hidden:`MultitaskGaussianLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MultitaskGaussianLikelihood
   :members:


:hidden:`HadamardGaussianLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: HadamardGaussianLikelihood
   :members:


:hlmod:`MultitaskQExponentialLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MultitaskQExponentialLikelihood
   :members:


:hlmod:`HadamardQExponentialLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: HadamardQExponentialLikelihood
   :members:


:hidden:`SoftmaxLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SoftmaxLikelihood
   :members:
