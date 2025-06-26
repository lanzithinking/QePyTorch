Multitask/Multioutput QEPs with Exact Inference
================================================

Exact QEPs can be used to model vector valued functions, or functions that represent multiple tasks.
There are several different cases:

Multi-output (vector valued functions)
----------------------------------------

- **Correlated output dimensions**: this is the most common use case.
  See the `Multitask QEP Regression`_ example, which implements the inference strategy defined in `Bonilla et al., 2008`_.
- **Independent output dimensions**: here we will use an uncorrelated QEP for each output.

  - If the outputs share the same kernel and mean, you can train a `Batch Uncorrelated Multioutput QEP`_.
  - Otherwise, you can train a `ModelList Multioutput QEP`_.

.. toctree::
   :maxdepth: 1
   :hidden:

   Multitask_QEP_Regression.ipynb
   Batch_Uncorrelated_Multioutput_QEP.ipynb
   ModelList_QEP_Regression.ipynb

Scalar function with multiple tasks
----------------------------------------

See the `Hadamard Multitask QEP Regression`_ example.
This setting should be used only when each input corresponds to a single task.

.. toctree::
   :maxdepth: 1
   :hidden:

   Hadamard_Multitask_QEP_Regression.ipynb


.. _Multitask QEP Regression:
  ./Multitask_QEP_Regression.ipynb

.. _Bonilla et al., 2008:
  https://papers.nips.cc/paper/3189-multi-task-gaussian-process-prediction

.. _Batch Uncorrelated Multioutput QEP:
  ./Batch_Uncorrelated_Multioutput_QEP.ipynb

.. _ModelList Multioutput QEP:
  ./ModelList_QEP_Regression.ipynb

.. _Hadamard Multitask QEP Regression:
  ./Hadamard_Multitask_QEP_Regression.ipynb
