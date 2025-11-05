Exact QEPs with Scalable (GPU) Inference
========================================

In QPyTorch, Exact QEP inference is still our preferred approach to large regression datasets.
By coupling GPU acceleration with `LancZos Variance Estimates (LOVE)`_,
QPyTorch can perform inference on datasets with over 1,000,000 data points while making very few approximations.

LancZos Variance Estimates (LOVE)
----------------------------------

`LanczOs Variance Estimates (LOVE)`_ (introduced by Pleiss et al., 2019) is a technique to rapidly speed up predictive variances and posterior sampling.
Check out the `QEP Regression with Fast Variances and Sampling (LOVE)`_ notebook to see how to use LOVE in QPyTorch, and how it compares to standard variance computations.

.. toctree::
   :maxdepth: 1
   :hidden:

   Simple_QEP_Regression_With_LOVE_Fast_Variances_and_Sampling.ipynb

Exact QEPs with GPU Acceleration
-----------------------------------

Here are examples of Exact QEPs using GPU acceleration.

- For datasets with up to 10,000 data points, see our `single GPU regression example`_.
- For datasets with up to 1,000,000 data points, see our `multi GPU regression example`_.
- GPyTorch also integrates with KeOPs for extremely fast and memory-efficient kernel computations.
  See the `KeOPs integration notebook`_.

.. toctree::
   :maxdepth: 1
   :hidden:

   Simple_QEP_Regression_CUDA.ipynb
   Simple_MultiGPU_QEP_Regression.ipynb
   KeOps_QEP_Regression.ipynb

Scalable Posterior Sampling with CIQ
---------------------------------------

Here we provide a notebook of `QEP poseterior sampling with CIQ`_ demonstrating the use of Contour Integral Quadrature with msMINRES as described in the `CIQ paper`_.
For the most dramatic results, we recommend combining this technique with other techniques in this section like kernel checkpointing with KeOps,
which would allow for posterior sampling on up to hundreds of thousands of test examples.

.. toctree::
   :maxdepth: 1
   :hidden:

   Exact_QEP_Posterior_Sampling_with_CIQ.ipynb

Scalable Kernel Approximations
-----------------------------------

While exact computations are our preferred approach, QPyTorch offer approximate kernels to reduce the asymptotic complexity of inference.

- `Sparse Q-Exponential Process Regression (SQEPR)`_ (proposed by Titsias, 2009) which approximates kernels using a set of inducing points.
  This is a general purpose approximation.
- `Structured Kernel Interpolation (SKI/KISS-QEP)`_ (proposed by Wilson and Nickish, 2015) which interpolates inducing points on a regularly spaced grid.
  This is designed for low-dimensional data and stationary kernels.
- `Structured Kernel Interpolation for Products (SKIP)`_ (proposed by Gardner et al., 2018) which extends SKI to higher dimensions.

.. toctree::
   :maxdepth: 1
   :hidden:

   SQEPR_Regression_CUDA.ipynb
   KISSQEP_Regression.ipynb
   Scalable_Kernel_Interpolation_for_Products_CUDA.ipynb

Structure-Exploiting Kernels
-----------------------------------

If your data lies on a Euclidean grid, and your QEP uses a stationary kernel, the computations can be sped up dramatically.
See the `Grid Regression`_ example for more info.

.. toctree::
   :maxdepth: 1
   :hidden:

   Grid_QEP_Regression.ipynb



.. _QEP Regression with Fast Variances and Sampling (LOVE):
  ./Simple_QEP_Regression_With_LOVE_Fast_Variances_and_Sampling.ipynb

.. _LancZos Variance Estimates (LOVE):
  https://arxiv.org/pdf/1803.06058.pdf

.. _single GPU regression example:
  ./Simple_QEP_Regression_CUDA.ipynb

.. _multi GPU regression example:
  ./Simple_MultiGPU_QEP_Regression.ipynb

.. _KeOPs integration notebook:
  ./KeOps_QEP_Regression.ipynb

.. _QEP poseterior sampling with CIQ:
  ./Exact_QEP_Posterior_Sampling_with_CIQ.ipynb

.. _Sparse Q-Exponential Process Regression (SQEPR):
  ./SQEPR_Regression_CUDA.ipynb

.. _Structured Kernel Interpolation (SKI/KISS-QEP):
  ./KISSQEP_Regression.ipynb

.. _Structured Kernel Interpolation for Products (SKIP):
  ./Scalable_Kernel_Interpolation_for_Products_CUDA.ipynb

.. _Grid Regression:
  ./Grid_QEP_Regression.ipynb

.. _CIQ paper:
  https://arxiv.org/abs/2006.11267
