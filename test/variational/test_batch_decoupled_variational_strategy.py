#!/usr/bin/env python3

import unittest

import torch

import qpytorch
from qpytorch.test.variational_test_case import VariationalTestCase

POWER = 1.0

def likelihood_cls():
    return qpytorch.likelihoods.GaussianLikelihood() if POWER==2 \
           else qpytorch.likelihoods.QExponentialLikelihood(power=torch.tensor(POWER))


def strategy_cls(model, inducing_points, variational_distribution, learn_inducing_locations):
    return qpytorch.variational.BatchDecoupledVariationalStrategy(
        model, inducing_points, variational_distribution, learn_inducing_locations
    )


def batch_dim_strategy_cls(model, inducing_points, variational_distribution, learn_inducing_locations):
    return qpytorch.variational.BatchDecoupledVariationalStrategy(
        model, inducing_points, variational_distribution, learn_inducing_locations, mean_var_batch_dim=-1
    )


class TestBatchDecoupledVariational(VariationalTestCase, unittest.TestCase):
    _power = POWER
    @property
    def batch_shape(self):
        return torch.Size([])

    @property
    def distribution_cls(self):
        return qpytorch.variational.CholeskyVariationalDistribution

    @property
    def likelihood_cls(self):
        return likelihood_cls

    @property
    def mll_cls(self):
        return qpytorch.mlls.VariationalELBO

    @property
    def strategy_cls(self):
        return strategy_cls

    def test_training_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_training_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertEqual(cholesky_mock.call_count, 2)  # One for each forward pass, and for computing prior dist
        self.assertFalse(ciq_mock.called)

    def test_eval_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_eval_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertEqual(cholesky_mock.call_count, 1)  # One to compute cache, that's it!
        self.assertFalse(ciq_mock.called)

    def test_fantasy_call(self, *args, **kwargs):
        # with self.assertRaises(AttributeError):
        #     super().test_fantasy_call(*args, **kwargs)
        pass


class TestBatchDecoupledPredictive(TestBatchDecoupledVariational):
    @property
    def mll_cls(self):
        return qpytorch.mlls.PredictiveLogLikelihood


class TestBatchDecoupledRobust(TestBatchDecoupledVariational):
    @property
    def mll_cls(self):
        return qpytorch.mlls.GammaRobustVariationalELBO


class TestMeanFieldBatchDecoupledVariational(TestBatchDecoupledVariational):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldBatchDecoupledPredictive(TestBatchDecoupledPredictive):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldBatchDecoupledRobust(TestBatchDecoupledRobust):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestBatchDecoupledVariationalBatchDim(TestBatchDecoupledVariational, unittest.TestCase):
    def _make_model_and_likelihood(
        self,
        num_inducing=16,
        batch_shape=torch.Size([]),
        inducing_batch_shape=torch.Size([]),
        strategy_cls=qpytorch.variational.VariationalStrategy,
        distribution_cls=qpytorch.variational.CholeskyVariationalDistribution,
        constant_mean=True,
    ):
        class _SV_PRegressionModel(qpytorch.models.ApproximateGP if POWER==2 else qpytorch.models.ApproximateQEP):
            def __init__(self, inducing_points):
                if POWER!=2: self.power = torch.tensor(POWER)
                variational_distribution = distribution_cls(num_inducing, batch_shape=batch_shape, power=self.power) if hasattr(self, 'power') \
                                           else distribution_cls(num_inducing, batch_shape=batch_shape)
                variational_strategy = strategy_cls(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                )
                super().__init__(variational_strategy)
                if constant_mean:
                    self.mean_module = qpytorch.means.ConstantMean(batch_shape=batch_shape + torch.Size([2]))
                    self.mean_module.initialize(constant=1.0)
                else:
                    self.mean_module = qpytorch.means.ZeroMean()
                self.covar_module = qpytorch.kernels.ScaleKernel(
                    qpytorch.kernels.RBFKernel(batch_shape=batch_shape + torch.Size([2])),
                    batch_shape=batch_shape + torch.Size([2]),
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                latent_pred = qpytorch.distributions.MultivariateQExponential(mean_x, covar_x, power=self.power) if hasattr(self, 'power') \
                              else qpytorch.distributions.MultivariateNormal(mean_x, covar_x)
                return latent_pred

        inducing_points = torch.randn(num_inducing, 2).repeat(*inducing_batch_shape, 1, 1)
        return _SV_PRegressionModel(inducing_points), self.likelihood_cls()

    @property
    def distribution_cls(self):
        return qpytorch.variational.CholeskyVariationalDistribution

    @property
    def mll_cls(self):
        return qpytorch.mlls.PredictiveLogLikelihood


class TestMeanFieldBatchDecoupledVariationalBatchDim(TestBatchDecoupledVariationalBatchDim, unittest.TestCase):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestBatchDecoupledVariationalOtherBatchDim(TestBatchDecoupledVariational, unittest.TestCase):
    def _make_model_and_likelihood(
        self,
        num_inducing=16,
        batch_shape=torch.Size([]),
        inducing_batch_shape=torch.Size([]),
        strategy_cls=qpytorch.variational.VariationalStrategy,
        distribution_cls=qpytorch.variational.CholeskyVariationalDistribution,
        constant_mean=True,
    ):
        class _SV_PRegressionModel(qpytorch.models.ApproximateGP if POWER==2 else qpytorch.models.ApproximateQEP):
            def __init__(self, inducing_points):
                if POWER!=2: self.power = torch.tensor(POWER)
                variational_distribution = distribution_cls(num_inducing, batch_shape=batch_shape, power=self.power) if hasattr(self, 'power') \
                                           else distribution_cls(num_inducing, batch_shape=batch_shape)
                variational_strategy = strategy_cls(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                )
                super().__init__(variational_strategy)
                if constant_mean:
                    self.mean_module = qpytorch.means.ConstantMean(
                        batch_shape=batch_shape[:-1] + torch.Size([2]) + batch_shape[-1:]
                    )
                    self.mean_module.initialize(constant=1.0)
                else:
                    self.mean_module = qpytorch.means.ZeroMean()
                self.covar_module = qpytorch.kernels.ScaleKernel(
                    qpytorch.kernels.RBFKernel(batch_shape=batch_shape[:-1] + torch.Size([2]) + batch_shape[-1:]),
                    batch_shape=batch_shape[:-1] + torch.Size([2]) + batch_shape[-1:],
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                latent_pred = qpytorch.distributions.MultivariateQExponential(mean_x, covar_x, power=self.power) if hasattr(self, 'power') \
                              else qpytorch.distributions.MultivariateNormal(mean_x, covar_x)
                return latent_pred

        inducing_points = torch.randn(num_inducing, 2).repeat(*inducing_batch_shape, 1, 1)
        return _SV_PRegressionModel(inducing_points), self.likelihood_cls()

    @property
    def strategy_cls(self):
        def _batch_dim_strategy_cls(model, inducing_points, variational_distribution, learn_inducing_locations):
            return qpytorch.variational.BatchDecoupledVariationalStrategy(
                model, inducing_points, variational_distribution, learn_inducing_locations, mean_var_batch_dim=-2
            )

        return _batch_dim_strategy_cls

    @property
    def batch_shape(self):
        return torch.Size([3])


if __name__ == "__main__":
    unittest.main()
