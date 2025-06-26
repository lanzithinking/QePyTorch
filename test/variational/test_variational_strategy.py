#!/usr/bin/env python3

import unittest

import torch

import qpytorch
from qpytorch.test.variational_test_case import VariationalTestCase

POWER = 1.0

class TestVariational(VariationalTestCase, unittest.TestCase):
    _power = POWER
    @property
    def batch_shape(self):
        return torch.Size([])

    @property
    def distribution_cls(self):
        return qpytorch.variational.CholeskyVariationalDistribution

    @property
    def mll_cls(self):
        return qpytorch.mlls.VariationalELBO

    @property
    def strategy_cls(self):
        return qpytorch.variational.VariationalStrategy

    def test_training_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_training_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertEqual(cholesky_mock.call_count, 2)  # One for each forward pass
        self.assertFalse(ciq_mock.called)

    def test_eval_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_eval_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertEqual(cholesky_mock.call_count, 1)  # One to compute cache, that's it!
        self.assertFalse(ciq_mock.called)

    def test_fantasy_call(self, *args, **kwargs):
        # we only want to check CholeskyVariationalDistribution
        if self.distribution_cls is qpytorch.variational.CholeskyVariationalDistribution:
            return super().test_fantasy_call(*args, **kwargs)

        with self.assertRaises(NotImplementedError):
            super().test_fantasy_call(*args, **kwargs)


class TestPredictive(TestVariational):
    @property
    def mll_cls(self):
        return qpytorch.mlls.PredictiveLogLikelihood


class TestRobust(TestVariational):
    @property
    def mll_cls(self):
        return qpytorch.mlls.GammaRobustVariationalELBO


class TestMeanFieldVariational(TestVariational):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldPredictive(TestPredictive):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldRobust(TestRobust):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestDeltaVariational(TestVariational):
    @property
    def distribution_cls(self):
        return qpytorch.variational.DeltaVariationalDistribution


class TestDeltaPredictive(TestPredictive):
    @property
    def distribution_cls(self):
        return qpytorch.variational.DeltaVariationalDistribution


class TestDeltaRobust(TestRobust):
    @property
    def distribution_cls(self):
        return qpytorch.variational.DeltaVariationalDistribution


class TestNGDVariational(TestVariational):
    @property
    def distribution_cls(self):
        return qpytorch.variational.NaturalVariationalDistribution

    def test_training_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = VariationalTestCase.test_training_iteration(self, *args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertEqual(cholesky_mock.call_count, 6)  # Three for each forward pass
        self.assertFalse(ciq_mock.called)

    def test_eval_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = VariationalTestCase.test_eval_iteration(self, *args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertEqual(cholesky_mock.call_count, 3)  # One to compute cache + 2 to compute variational distribution
        self.assertFalse(ciq_mock.called)


if __name__ == "__main__":
    unittest.main()
