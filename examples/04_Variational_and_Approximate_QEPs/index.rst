Variational and Approximate QEPs
================================

Variational and approximate q-exponential processes are used in a variety of cases:

- When the likelihood is non-Gaussian (e.g. for classification).
- To scale up QEP regression (by using stochastic optimization).
- To use QEPs as part of larger probabilistic models.

With QPyTorch it is possible to implement various types approximate QEP models.
All approximate models consist of the following 3 composable objects:

- :obj:`VariationalDistribution`, which define the form of the approximate inducing value
  posterior :math:`q(\mathbf u)`.
- :obj:`VarationalStrategies`, which define how to compute :math:`q(\mathbf f(\mathbf X))` from
  :math:`q(\mathbf u)`.
- :obj:`~qpytorch.mlls._ApproximateMarginalLogLikelihood`, which defines the objective function
  to learn the approximate posterior (e.g. variational ELBO).

(See the `strategy/distribution comparison`_ for examples of the different classes.)
The variational documentation has more information on how to use these objects.
Here we provide some examples which highlight some of the common use cases:

- **Large-scale regression** (when exact methods are too memory intensive): see the `stochastic variational regression example`_.
- **Variational inference with natural gradient descent** (for faster/better optimization): see the `ngd example`_.
- **Variational inference with contour integral quadrature** (for large numbers of inducing points): see the `ciq example`_.
- **Variational inference with nearest neighbor approximation** (for large numbers of inducing points): see the `vnnqep example`_.
- **Variational distribution options** for different scalability/expressiveness: see the `strategy/distribution comparison`_.
- **Alternative optimization objectives** (not fully tested) for the QEP's predictive distribution: see the `approximate QEP objective functions notebook`_.
  This example compares and contrasts the variational ELBO with the predictive log likelihood of Jankowiak et al., 2020.
- **Classification**: see the `non-Gaussian likelihood notebook`_ and the `Polya-Gamma Binary Classification notebook`_.
- **Multi-output variational QEPs** (when exact methods are too memory intensive): see the `variational QEPs with multiple outputs example`_.
- **Uncertain inputs**: see the `QEPs with uncertain inputs example`_.

.. toctree::
   :maxdepth: 1
   :hidden:

   SVQEP_Regression_CUDA.ipynb
   Modifying_the_variational_strategy_and_distribution.ipynb
   Natural_Gradient_Descent.ipynb
   SVQEP_CIQ.ipynb
   VNNQEP.ipynb
   Approximate_QEP_Objective_Functions.ipynb
   Non_Gaussian_Likelihoods.ipynb
   PolyaGamma_Binary_Classification.ipynb
   SVQEP_Multitask_QEP_Regression.ipynb
   QEP_Regression_with_Uncertain_Inputs.ipynb

.. _strategy/distribution comparison:
  ./Modifying_the_variational_strategy_and_distribution.ipynb

.. _stochastic variational regression example:
  ./SVQEP_Regression_CUDA.ipynb

.. _ngd example:
  ./Natural_Gradient_Descent.ipynb

.. _ciq example:
  ./SVQEP_CIQ.ipynb

.. _vnnqep example:
  ./VNNQEP.ipynb

.. _approximate QEP objective functions notebook:
  ./Approximate_QEP_Objective_Functions.ipynb

.. _non-Gaussian likelihood notebook:
  ./Non_Gaussian_Likelihoods.ipynb

.. _Polya-Gamma Binary Classification notebook:
  ./PolyaGamma_Binary_Classification.ipynb

.. _variational QEPs with multiple outputs example:
  ./SVQEP_Multitask_QEP_Regression.ipynb

.. _QEPs with uncertain inputs example:
  ./QEP_Regression_with_Uncertain_Inputs.ipynb
