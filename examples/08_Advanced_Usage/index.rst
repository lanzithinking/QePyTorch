Advanced Usage
===============================================

Here are some examples highlighting QPyTorch's more advanced features.

Batch QEPs
-----------

QPyTorch makes it possible to train/perform inference with a batch of q-exponential processes in parallel.
This can be useful for a number of applications:

 - Modeling a function with multiple (uncorrelated) outputs
 - Performing efficient cross-validation
 - Parallel acquisition function sampling for Bayesian optimization
 - And more!

Here we highlight a number of common batch QEP scenarios and how to construct them in QPyTorch.

- **Multi-output functions (with uncorrelated outputs).** Batch QEPs are extremely efficient at modelling multi-output functions, when each of the output functions
  are **independent**. See the `Batch Uncorrelated Multioutput QEP`_ example for more details.

- **For cross validation**, or for some BayesOpt applications, it may make sense to evaluate the QEP on different batches of test data.
  This can be accomplished by using a standard (non-batch) QEP model.
  At test time, feeding a `b x n x d` tensor into the model will then return `b` batches of `n` test points.
  See the `Batch Mode Regression`_ example for more details.

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   ../03_Multitask_Exact_QEPs/Batch_Uncorrelated_Multioutput_QEP.ipynb
   Simple_Batch_Mode_QEP_Regression.ipynb

.. _Batch Independent Multioutput QEP:
  ../03_Multitask_Exact_QEPs/Batch_Uncorrelated_Multioutput_QEP.ipynb

.. _Batch Mode Regression:
  Simple_Batch_Mode_QEP_Regression.ipynb


Variational Fantasization
----------------------------------
We also include an example of how to perform fantasy modelling (e.g. efficient, closed form updates) for variational
q-exponential process models, enabling their usage for lookahead optimization. See the `Variational fantasization`_ example.

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   SVQEP_Model_Updating.ipynb

.. _Variational fantasization:
  SVQEP_Model_Updating.ipynb

Converting Models to TorchScript
----------------------------------

In order to deploy QEPs in production code, it can be desirable to avoid using PyTorch directly for performance reasons.
Fortunately, PyTorch offers a mechanism called TorchScript to aid in this. In these example notebooks, we'll demonstrate
how to convert both an exact QEP and a variational QEP to a ScriptModule that can then be used for example in LibTorch.

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   TorchScript_Exact_Models.ipynb
   TorchScript_Variational_Models.ipynb
