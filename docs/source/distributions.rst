.. role:: hidden
    :class: hidden-section

qpytorch.distributions
===================================

QPyTorch distribution objects are essentially the same as GPyTorch distribution objects excepted those :hlmod:`highlighted`.
For the most part, QPyTorch relies on torch's distribution library.
However, we offer two custom distributions.

We implement a custom :obj:`~qpytorch.distributions.MultivariateQExponential` that accepts
:obj:`~linear_operator.operators.LinearOperator` objects for covariance matrices. This allows us to use custom
linear algebra operations, which makes this more efficient.

In addition, we implement a :obj:`~qpytorch.distributions.MultitaskMultivariateQExponential` which
can be used with multi-output Q-exponential process models.

.. note::

  If Pyro is available, all GPyTorch distribution objects inherit Pyro's distribution methods
  as well.

.. automodule:: qpytorch.distributions
.. currentmodule:: qpytorch.distributions


Distribution
-----------------------------

.. autoclass:: Distribution
   :members:


Delta
----------------------------------

.. class:: Delta(v, log_density=0.0, event_dim=0, validate_args=None)

  (Borrowed from Pyro.) Degenerate discrete distribution (a single point).

  Discrete distribution that assigns probability one to the single element in
  its support. Delta distribution parameterized by a random choice should not
  be used with MCMC based inference, as doing so produces incorrect results.

  :param v: The single support element.
  :param log_density: An optional density for this Delta. This is useful to
    keep the class of Delta distributions closed under differentiable
    transformation.
  :param event_dim: Optional event dimension, defaults to zero.
  :type v: torch.Tensor
  :type log_density: torch.Tensor
  :type event_dim: int


MultivariateNormal
-----------------------------

.. autoclass:: MultivariateNormal
   :members:
   :special-members: __getitem__


MultitaskMultivariateNormal
----------------------------------

.. autoclass:: MultitaskMultivariateNormal
   :members:
   :special-members: __getitem__


:hlmod:`QExponential`
-----------------------------

.. autoclass:: QExponential
   :members:
   :special-members: __getitem__


:hlmod:`MultivariateQExponential`
----------------------------------

.. autoclass:: MultivariateQExponential
   :members:
   :special-members: __getitem__


:hlmod:`MultitaskMultivariateQExponential`
-------------------------------------------

.. autoclass:: MultitaskMultivariateQExponential
   :members:
   :special-members: __getitem__


:hlmod:`Power`
-----------------------------

.. autoclass:: Power
   :members:
   :special-members: data