#!/usr/bin/env python3

import unittest

from qpytorch.kernels import PolynomialKernelGrad
from qpytorch.test import BaseKernelTestCase


class TestPolynomialKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return PolynomialKernelGrad(power=2, **kwargs)


if __name__ == "__main__":
    unittest.main()
