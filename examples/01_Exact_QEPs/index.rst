Exact QEPs (Regression)
========================

Regression with a q-exponential noise model is the canonical example of Q-exponential processes.
These examples will work for small to medium sized datasets (~2,000 data points).
All examples here use exact QEP inference.

- `Simple QEP Regression`_ is the basic tutorial for regression in QPyTorch.
- `Spectral Mixture Regression`_ extends on the previous example with a more complex kernel.
- `Fully Bayesian QEP Regression`_ demonstrates how to perform fully Bayesian inference by sampling the QEP hyperparameters
  using NUTS. (This example requires Pyro to be installed).
- `Distributional QEP Regression`_ is an example of how to take account of uncertainty in inputs.
- `Dirichlet Classification`_ is an example of how to perform regression on classification labels via an approximate
  likelihood.

.. toctree::
   :maxdepth: 1
   :hidden:

   Simple_QEP_Regression.ipynb
   Spectral_Delta_QEP_Regression.ipynb
   Spectral_Mixture_QEP_Regression.ipynb
   QEP_Regression_Fully_Bayesian.ipynb
   QEP_Regression_DistributionalKernel.ipynb
   QEP_Regression_on_Classification_Labels.ipynb

.. _Simple QEP Regression:
  ./Simple_QEP_Regression.ipynb

.. _Spectral Mixture Regression:
  ./Spectral_Mixture_QEP_Regression.ipynb

.. _Fully Bayesian QEP Regression:
  ./QEP_Regression_Fully_Bayesian.ipynb

.. _Distributional QEP Regression:
  ./QEP_Regression_DistributionalKernel.ipynb

.. _Dirichlet Classification:
  ./QEP_Regression_on_Classification_Labels.ipynb
