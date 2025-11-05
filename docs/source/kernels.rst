.. role:: hidden
    :class: hidden-section

qpytorch.kernels
===================================

.. automodule:: qpytorch.kernels
.. currentmodule:: qpytorch.kernels


QPyTorch kernels are ported from GPyTorch excepted those :hlmod:`highlighted`. A good starting point is
:code:`qpytorch.kernels.ScaleKernel(qpytorch.kernels.RBFKernel()) + qpytorch.kernels.ConstantKernel()`.


Kernel
----------------

.. autoclass:: Kernel
   :members:
   :special-members: __call__, __getitem__

Standard Kernels
-----------------------------

:hidden:`ConstantKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ConstantKernel
   :members:


:hidden:`CosineKernel`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CosineKernel
   :members:


:hidden:`CylindricalKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CylindricalKernel
   :members:


:hidden:`LinearKernel`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LinearKernel
   :members:

:hidden:`MaternKernel`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MaternKernel
   :members:

:hidden:`PeriodicKernel`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PeriodicKernel
   :members:

:hidden:`PiecewisePolynomialKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PiecewisePolynomialKernel
   :members:

:hidden:`PolynomialKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PolynomialKernel
   :members:

:hidden:`PolynomialKernelGrad`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PolynomialKernelGrad
   :members:

:hidden:`RBFKernel`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: RBFKernel
   :members:

:hidden:`RQKernel`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: RQKernel
   :members:

:hidden:`SpectralDeltaKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SpectralDeltaKernel
   :members:

:hidden:`SpectralMixtureKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SpectralMixtureKernel
   :members:


Composition/Decoration Kernels
-----------------------------------

:hidden:`AdditiveKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AdditiveKernel
   :members:

:hidden:`MultiDeviceKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MultiDeviceKernel
   :members:

:hidden:`AdditiveStructureKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AdditiveStructureKernel
   :members:

:hidden:`ProductKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ProductKernel
   :members:

:hidden:`ProductStructureKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ProductStructureKernel
   :members:

:hidden:`ScaleKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ScaleKernel
   :members:



Specialty Kernels
-----------------------------------

:hidden:`ArcKernel`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ArcKernel
   :members:

:hidden:`HammingIMQKernel`

..autoclass:: HammingIMQKernel
  :members:

:hidden:`IndexKernel`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: IndexKernel
   :members:

:hidden:`LCMKernel`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LCMKernel
   :members:

:hidden:`MultitaskKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MultitaskKernel
   :members:

:hidden:`RBFKernelGrad`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RBFKernelGrad
   :members:

:hidden:`RBFKernelGradGrad`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RBFKernelGradGrad
   :members:

:hlmod:`RQKernelGrad`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RQKernelGrad
   :members:

:hlmod:`RQKernelGradGrad`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RQKernelGradGrad
   :members:

:hlmod:`Matern32KernelGrad`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Matern32KernelGrad
   :members:

:hidden:`Matern52KernelGrad`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Matern52KernelGrad
   :members:

:hlmod:`Matern52KernelGradGrad`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Matern52KernelGradGrad
   :members:

Kernels for Scalable GP/QEP Regression Methods
-----------------------------------------------

:hidden:`GridKernel`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GridKernel
   :members:

:hidden:`GridInterpolationKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GridInterpolationKernel
   :members:

:hidden:`InducingPointKernel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: InducingPointKernel
   :members:

:hidden:`RFFKernel`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: RFFKernel
   :members:
