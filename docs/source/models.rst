.. role:: hidden
    :class: hidden-section

qpytorch.models
===================================

QPyTorch models are ported from GPyTorch excepted those :hlmod:`highlighted`.

.. automodule:: qpytorch.models
.. currentmodule:: qpytorch.models


Models for Exact Inference
-----------------------------

:hidden:`ExactGP`
~~~~~~~~~~~~~~~~~

.. autoclass:: ExactGP
   :members:


:hlmod:`ExactQEP`
~~~~~~~~~~~~~~~~~

.. autoclass:: ExactQEP
   :members:


Models for Approximate Inference
-----------------------------------

:hidden:`ApproximateGP`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ApproximateGP
   :members:


:hlmod:`ApproximateQEP`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ApproximateQEP
   :members:


Deep Probabilistic Models
-----------------------------------

:hidden:`deep_gps.DeepGP`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: gpytorch.models.deep_gps.DeepGP
   :members:

:hidden:`deep_gps.DeepGPLayer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: gpytorch.models.deep_gps.DeepGPLayer
   :members:


:hlmod:`deep_qeps.DeepQEP`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: qpytorch.models.deep_qeps.DeepQEP
   :members:

:hlmod:`deep_qeps.DeepQEPLayer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: qpytorch.models.deep_qeps.DeepQEPLayer
   :members:


Gaussian Process Latent Variable Models (GPLVM)
-------------------------------------------------------

:hidden:`gplvm.BayesianGPLVM`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: gpytorch.models.gplvm.BayesianGPLVM
   :members:

:hidden:`gplvm.PointLatentVariable`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: gpytorch.models.gplvm.PointLatentVariable
   :members:

:hidden:`gplvm.MAPLatentVariable`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: gpytorch.models.gplvm.MAPLatentVariable
   :members:

:hidden:`gplvm.VariationalLatentVariable`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: gpytorch.models.gplvm.VariationalLatentVariable
   :members:


:hlmod:`Q-Exponential Process Latent Variable Models (QEPLVM)`
---------------------------------------------------------------

:hlmod:`qeplvm.BayesianQEPLVM`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: qpytorch.models.qeplvm.BayesianQEPLVM
   :members:

:hlmod:`qeplvm.PointLatentVariable`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: qpytorch.models.qeplvm.PointLatentVariable
   :members:

:hlmod:`qeplvm.MAPLatentVariable`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: qpytorch.models.qeplvm.MAPLatentVariable
   :members:

:hlmod:`qeplvm.VariationalLatentVariable`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: qpytorch.models.qeplvm.VariationalLatentVariable
   :members:


Models for integrating with Pyro
-----------------------------------

:hidden:`PyroGP`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: gpytorch.models.pyro.PyroGP
   :members:

:hlmod:`PyroQEP`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: qpytorch.models.pyro.PyroQEP
   :members:


:hlmod:`Q-Exponential Process PDE Solver`
---------------------------------------------

:hlmod:`PDESolver`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: qpytorch.models.PDESolver
   :members:
