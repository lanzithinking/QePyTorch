Basic Usage
==============

This folder contains notebooks for basic usage of the package, e.g. things like dealing with hyperparameters,
parameter constraints and priors, and saving and loading models.


Before checking these out, you may want to check out the `introduction to Q-Exponential Process`_ and
our `simple QEP regression tutorial`_ that details the anatomy of a QPyTorch model.

* Check out our `Tutorial on Hyperparameters`_ for information on things like raw versus actual
  parameters, constraints, priors and more.
* The `Saving and Loading Models`_ notebook details how to save and load QPyTorch models
  on disk.
* The `Kernels with Additive or Product Structure`_ notebook describes how to compose kernels additively or multiplicatively,
  whether for expressivity, sample efficiency, or scalability.
* The `Implementing a Custom Kernel`_ notebook details how to write your own custom kernel in QPyTorch.
* The `Tutorial on Metrics`_ describes various metrics provided by QPyTorch for assessing the generalization of QEP models.

.. toctree::
   :maxdepth: 1
   :hidden:

   Hyperparameters.ipynb
   Saving_and_Loading_Models.ipynb
   kernels_with_additive_or_product_structure.ipynb
   Implementing_a_custom_Kernel.ipynb
   Metrics.ipynb

.. _introduction to Q-Exponential Process:
  Introduction_to_QExponential_Process.ipynb

.. _simple QEP regression tutorial:
  ../01_Exact_QEPs/Simple_QEP_Regression.ipynb

.. _Tutorial on Hyperparameters:
  Hyperparameters.ipynb

.. _Saving and Loading Models:
  Saving_and_Loading_Models.ipynb

.. _Kernels with Additive or Product Structure:
  kernels_with_additive_or_product_structure.ipynb

.. _Implementing a custom Kernel:
  Implementing_a_custom_Kernel.ipynb

.. _Tutorial on Metrics:
  Metrics.ipynb
