Pyro Integration
===================

QPyTorch can optionally work with the Pyro probabilistic programming language.
This makes it possible to use Pyro's advanced inference algorithms, or to incorporate QEPs as part of larger probabilistic models.
QPyTorch offers two ways of integrating with Pyro:

High-level Pyro Interface (for predictive models)
--------------------------------------------------

The high-level interface provides a simple wrapper around :obj:`~qpytorch.models.ApproximateQEP` that makes it
possible to use Pyro's inference tools with QPyTorch models.
It is best designed for:

- Developing models that will be used for predictive tasks
- QEPs with likelihoods that have additional latent variables

The `Pyro + QPyTorch High-Level Introduction`_ gives an overview of the high-level interface.
For a more in-depth example that shows off the power of the integration, see the `Clustered Multitask QEP Example`_.

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   Pyro_QPyTorch_High_Level.ipynb
   Clustered_Multitask_QEP_Regression.ipynb


Low-level Pyro Interface (for latent function inference)
----------------------------------------------------------

The low-level interface simply provides tools to compute QEP latent functions, and requires users to write their own :meth:`model` and :meth:`guide` functions.
It is best designed for:

- Performing inference on probabilistic models that involve QEPs
- Models with complicated likelihoods

The `Pyro + QPyTorch Low-Level Introduction`_ gives an overview of the low-level interface.
The `Cox Process Example`_ is a more in-depth example of a model that can be built using this interface.

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   Pyro_QPyTorch_Low_Level.ipynb
   Cox_Process_Example.ipynb

.. _Pyro + QPyTorch High-Level Introduction:
  Pyro_QPyTorch_High_Level.ipynb

.. _Clustered Multitask QEP Example:
  Clustered_Multitask_QEP_Regression.ipynb

.. _Pyro + QPyTorch Low-Level Introduction:
  Pyro_QPyTorch_Low_Level.ipynb

.. _Cox Process Example:
  Cox_Process_Example.ipynb
