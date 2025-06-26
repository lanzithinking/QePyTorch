#!/usr/bin/env python3

import unittest

import torch
from linear_operator.operators import CholLinearOperator, TriangularLinearOperator

import gpytorch, qpytorch
from gpytorch.constraints import GreaterThan
from qpytorch.distributions import MultivariateNormal, MultivariateQExponential
from qpytorch.variational import NaturalVariationalDistribution, TrilNaturalVariationalDistribution

TEST_MDL = 'GP'; POWER = {'GP': 2.0, 'QEP': 1.0}[TEST_MDL]

class Float64Test(unittest.TestCase):
    def setUp(self):
        self.prev_type = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        torch.set_default_dtype(self.prev_type)


class TestNatVariational(Float64Test):
    def test_one_step_optimal_high_precision(self):
        X = torch.linspace(-3, 3, 10)
        Y = torch.sin(X)

        class Exact_({'GP':gpytorch.models.ExactGP, 'QEP':qpytorch.models.ExactQEP}[TEST_MDL]):
            def __init__(self, train_x, train_y, kern, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = qpytorch.means.ZeroMean()
                self.covar_module = kern

            def forward(self, x):
                mean = self.mean_module(x)
                covar = self.covar_module(x)
                return {'GP': gpytorch.distributions.MultivariateNormal(mean, covar),
                        'QEP': qpytorch.distributions.MultivariateQExponential(mean, covar, power=torch.tensor(POWER))}[TEST_MDL]

        likelihood = {'GP': gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(0, initial_value=0.1)),
                      'QEP': qpytorch.likelihoods.QExponentialLikelihood(noise_constraint=GreaterThan(0, initial_value=0.1), power=torch.tensor(POWER))}[TEST_MDL]

        kern = qpytorch.kernels.ScaleKernel(qpytorch.kernels.RBFKernel())
        kern.outputscale = 1

        model_exact = Exact_(X, Y, kern, likelihood)
        model_exact.eval()
        prediction_exact = model_exact(X)

        class NatGrads_({'GP':gpytorch.models.ApproximateGP, 'QEP':qpytorch.models.ApproximateQEP}[TEST_MDL]):
            def __init__(self, kern, inducing_points):
                variational_distribution = {'GP': gpytorch.variational.NaturalVariationalDistribution(inducing_points.shape[0]),
                                            'QEP': qpytorch.variational.NaturalVariationalDistribution(inducing_points.shape[0], power=torch.tensor(POWER))}[TEST_MDL]
                variational_strategy = qpytorch.variational.VariationalStrategy(
                    self,
                    inducing_points,
                    variational_distribution,
                    learn_inducing_locations=True,
                    jitter_val=1e-24,
                )
                super().__init__(variational_strategy)
                self.mean_module = qpytorch.means.ConstantMean()
                self.covar_module = kern

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return {'GP': gpytorch.distributions.MultivariateNormal(mean_x, covar_x),
                        'QEP': qpytorch.distributions.MultivariateQExponential(mean_x, covar_x, power=torch.tensor(POWER))}[TEST_MDL]

        model_ng = NatGrads_(kern, X)

        mll = qpytorch.mlls.VariationalELBO(likelihood, model_ng, num_data=X.shape[0])
        from torch.utils.data import DataLoader, TensorDataset

        data = DataLoader(TensorDataset(X, Y), batch_size=X.shape[0])

        variational_ngd_optimizer = qpytorch.optim.NGD(model_ng.variational_parameters(), num_data=X.size(0), lr=1)
        for _ in range(1):  # one step
            for x, y in data:
                variational_ngd_optimizer.zero_grad()

                loss = -mll(model_ng(x), y)
                # minibatch_iter.set_postfix(loss=loss.item())
                loss.backward()
                variational_ngd_optimizer.step()

        prediction_ng = model_ng(X)

        assert torch.allclose(prediction_exact.mean, prediction_ng.mean, rtol=1e-12, atol=1e-12)
        assert torch.allclose(prediction_exact.variance, prediction_ng.variance, rtol=1e-12, atol=1e-12)

    def test_invertible_init(self, D=5):
        mu = torch.randn(D)
        cov = torch.randn(D, D).tril_()
        dist = {'GP': MultivariateNormal(mu, CholLinearOperator(TriangularLinearOperator(cov))),
                'QEP': MultivariateQExponential(mu, CholLinearOperator(TriangularLinearOperator(cov)), power=torch.tensor(POWER))}[TEST_MDL]

        v_dist = {'GP': NaturalVariationalDistribution(D, mean_init_std=0.0),
                  'QEP': NaturalVariationalDistribution(D, mean_init_std=0.0, power=torch.tensor(POWER))}[TEST_MDL]
        v_dist.initialize_variational_distribution(dist)

        out_dist = v_dist()

        assert torch.allclose(out_dist.mean, dist.mean, rtol=1e-04, atol=1e-06)
        assert torch.allclose(out_dist.covariance_matrix, dist.covariance_matrix)

    def test_natgrad(self, D=5):
        mu = torch.randn(D)
        cov = torch.randn(D, D).tril_()
        dist = {'GP': MultivariateNormal(mu, CholLinearOperator(TriangularLinearOperator(cov))),
                'QEP': MultivariateQExponential(mu, CholLinearOperator(TriangularLinearOperator(cov)), power=torch.tensor(POWER))}[TEST_MDL]
        sample = dist.sample()

        v_dist = {'GP': NaturalVariationalDistribution(D),
                  'QEP': NaturalVariationalDistribution(D, power=torch.tensor(POWER))}[TEST_MDL]
        v_dist.initialize_variational_distribution(dist)
        mu = v_dist().mean.detach()

        v_dist().log_prob(sample).squeeze().backward()

        eta1 = mu.clone().requires_grad_(True)
        eta2 = (mu[:, None] * mu + cov @ cov.t()).requires_grad_(True)
        L = torch.linalg.cholesky(eta2 - eta1[:, None] * eta1)
        dist2 = {'GP': MultivariateNormal(eta1, CholLinearOperator(TriangularLinearOperator(L))),
                'QEP': MultivariateQExponential(eta1, CholLinearOperator(TriangularLinearOperator(L)), power=torch.tensor(POWER))}[TEST_MDL]
        dist2.log_prob(sample).squeeze().backward()

        assert torch.allclose(v_dist.natural_vec.grad, eta1.grad)
        assert torch.allclose(v_dist.natural_mat.grad, eta2.grad)

    def test_optimization_optimal_error(self, num_inducing=16, num_data=32, D=2):
        inducing_points = torch.randn(num_inducing, D)

        class SV_P({'GP':qpytorch.models.ApproximateGP, 'QEP':qpytorch.models.ApproximateQEP}[TEST_MDL]):
            def __init__(self):
                v_dist = {'GP': NaturalVariationalDistribution(num_inducing),
                          'QEP': NaturalVariationalDistribution(num_inducing, power=torch.tensor(POWER))}[TEST_MDL]
                v_strat = qpytorch.variational.UnwhitenedVariationalStrategy(self, inducing_points, v_dist)
                super().__init__(v_strat)
                self.mean_module = qpytorch.means.ZeroMean()
                self.covar_module = qpytorch.kernels.RBFKernel()

            def forward(self, x):
                return {'GP': MultivariateNormal(self.mean_module(x), self.covar_module(x)),
                        'QEP': MultivariateQExponential(self.mean_module(x), self.covar_module(x), power=torch.tensor(POWER))}[TEST_MDL]

        model = SV_P().train()
        likelihood = {'GP': gpytorch.likelihoods.GaussianLikelihood(),
                      'QEP': qpytorch.likelihoods.QExponentialLikelihood(power=torch.tensor(POWER))}[TEST_MDL].train()
        mll = qpytorch.mlls.VariationalELBO(likelihood, model, num_data)
        X = torch.randn((num_data, D))
        y = torch.randn(num_data)

        def loss():
            return -mll(model(X), y)

        optimizer = torch.optim.SGD(
            model.variational_strategy._variational_distribution.parameters(), lr=float(num_data)
        )

        optimizer.zero_grad()
        loss().backward()
        optimizer.step()  # Now we should be at the optimum

        optimizer.zero_grad()
        loss().backward()
        natgrad_natural_vec2, natgrad_natural_mat2 = (
            model.variational_strategy._variational_distribution.natural_vec.grad.clone(),
            model.variational_strategy._variational_distribution.natural_mat.grad.clone(),
        )
        # At the optimum, the (natural) gradients are zero:
        assert torch.allclose(natgrad_natural_vec2, torch.zeros(()))
        assert torch.allclose(natgrad_natural_mat2, torch.zeros(()))


class TestTrilNatVariational(Float64Test):
    def test_invertible_init(self, D=5):
        mu = torch.randn(D)
        cov = torch.randn(D, D).tril_()
        dist = {'GP': MultivariateNormal(mu, CholLinearOperator(TriangularLinearOperator(cov))),
                'QEP': MultivariateQExponential(mu, CholLinearOperator(TriangularLinearOperator(cov)), power=torch.tensor(POWER))}[TEST_MDL]

        v_dist = {'GP': TrilNaturalVariationalDistribution(D, mean_init_std=0.0),
                  'QEP': TrilNaturalVariationalDistribution(D, mean_init_std=0.0, power=torch.tensor(POWER))}[TEST_MDL]
        v_dist.initialize_variational_distribution(dist)

        out_dist = v_dist()

        assert torch.allclose(out_dist.mean, dist.mean)
        assert torch.allclose(out_dist.covariance_matrix, dist.covariance_matrix)

    def test_natgrad(self, D=5):
        mu = torch.randn(D)
        cov = torch.randn(D, D)
        cov = cov @ cov.t()
        dist = {'GP': MultivariateNormal(mu, CholLinearOperator(TriangularLinearOperator(torch.linalg.cholesky(cov)))),
                'QEP': MultivariateQExponential(mu, CholLinearOperator(TriangularLinearOperator(torch.linalg.cholesky(cov))), power=torch.tensor(POWER))}[TEST_MDL]
        sample = dist.sample()

        v_dist = {'GP': TrilNaturalVariationalDistribution(D, mean_init_std=0.0),
                  'QEP': TrilNaturalVariationalDistribution(D, mean_init_std=0.0, power=torch.tensor(POWER))}[TEST_MDL]
        v_dist.initialize_variational_distribution(dist)
        v_dist().log_prob(sample).squeeze().backward()
        dout_dnat1 = v_dist.natural_vec.grad
        dout_dnat2 = v_dist.natural_tril_mat.grad

        # mean_init_std=0. because we need to ensure both have the same distribution
        v_dist_ref = {'GP': NaturalVariationalDistribution(D, mean_init_std=0.0),
                      'QEP': NaturalVariationalDistribution(D, mean_init_std=0.0, power=torch.tensor(POWER))}[TEST_MDL]
        v_dist_ref.initialize_variational_distribution(dist)
        v_dist_ref().log_prob(sample).squeeze().backward()
        dout_dnat1_noforward_ref = v_dist_ref.natural_vec.grad
        dout_dnat2_noforward_ref = v_dist_ref.natural_mat.grad

        def f(natural_vec, natural_tril_mat):
            "Transform natural_tril_mat to L"
            Sigma = torch.inverse(-2 * natural_tril_mat)
            mu = natural_vec
            return mu, torch.linalg.cholesky(Sigma).inverse().tril()

        (mu_ref, natural_tril_mat_ref), (dout_dmu_ref, dout_dnat2_ref) = jvp(
            f,
            (v_dist_ref.natural_vec.detach(), v_dist_ref.natural_mat.detach()),
            (dout_dnat1_noforward_ref, dout_dnat2_noforward_ref),
        )

        assert torch.allclose(natural_tril_mat_ref, v_dist.natural_tril_mat), "Sigma transformation"
        assert torch.allclose(dout_dnat2_ref, dout_dnat2), "Sigma gradient"

        assert torch.allclose(mu_ref, v_dist.natural_vec), "mu transformation"
        assert torch.allclose(dout_dmu_ref, dout_dnat1), "mu gradient"


def jvp(f, x, v):
    "Simulate forward-mode AD using two reverse-mode AD"
    x = tuple(xx.requires_grad_(True) for xx in x)
    v = tuple(vv.requires_grad_(True) for vv in v)
    y = f(*x)
    grad_x = torch.autograd.grad(y, x, v, create_graph=True)
    jvp_val = torch.autograd.grad(grad_x, v, v)
    return y, jvp_val
