.. QPyTorch documentation master file, created by
   sphinx-quickstart on Wed Jun 11 00:09:48 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/lanzithinking/QePyTorch

QPyTorch's documentation
====================================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tutorials:

   examples/00_Basic_Usage/Introduction_to_QExponential_Process.ipynb
   examples/01_Exact_QEPs/Simple_QEP_Regression.ipynb

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Examples:

   examples/**/index

.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   models
   likelihoods
   kernels
   keops_kernels
   means
   marginal_log_likelihoods
   metrics
   constraints
   distributions
   priors
   variational
   optim

.. toctree::
   :maxdepth: 1
   :caption: Settings and Beta Features

   settings
   beta_features

.. toctree::
   :maxdepth: 1
   :caption: Advanced Package Reference

   module
   functions
   utils



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Projects
==================

* Q-Exponential Process    `[Q-EXP] <https://github.com/lanzithinking/Q-EXP>`_
* Regularization of Latent Representation    `[Reg_Rep] <https://github.com/lanzithinking/Reg_Rep>`_
* Deep Q-Exponential Processes    `[DeepQEP] <https://github.com/lanzithinking/DeepQEP>`_
* Solving PDE with Q-Exponential Processes    `[Diff_QEP] <https://github.com/lanzithinking/Diff_QEP>`_

Research references
======================

* Li, Shuyi, Michael O'Connor, and Shiwei Lan. "Bayesian Learning via Q-Exponential Process." In Advances in NIPS (2023).
* Obite, Chukwudi P., Zhi Chang, Keyan Wu, and Shiwei Lan.  "Bayesian Regularization on Latent Representation." In ICLR (2025).
* Chang, Zhi, Chukwudi P. Obite, Shuang Zhou, and Shiwei Lan.  "Deep Q-Exponential Processes. " In AABI (2025).
* Yu, Guangting and Shiwei Lan. "Solving and Learning Partial Differential Equations with Variational Q-Exponential Processes." In NIPS (2025).
