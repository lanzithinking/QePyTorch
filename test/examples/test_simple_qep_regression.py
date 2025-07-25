#!/usr/bin/env python3

import unittest
import warnings
from math import exp, pi

import torch
from torch import optim

import qpytorch
from qpytorch.constraints import Positive
from qpytorch.distributions import MultivariateQExponential
from qpytorch.kernels import RBFKernel, ScaleKernel
from qpytorch.likelihoods import QExponentialLikelihood
from qpytorch.means import ConstantMean
from qpytorch.priors import SmoothedBoxPrior, UniformPrior
from qpytorch.test import BaseTestCase
from gpytorch.test.utils import least_used_cuda_device

POWER = 1.0

class ExactQEPModel(qpytorch.models.ExactQEP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ExactQEPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateQExponential(mean_x, covar_x, power=self.likelihood.power)


class TestSimpleQEPRegression(BaseTestCase, unittest.TestCase):
    seed = 1

    def _get_data(self, cuda=False, num_data=11, add_noise=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        # Simple training data: let's try to learn a sine function
        train_x = torch.linspace(0, 1, num_data, device=device)
        train_y = torch.sin(train_x * (2 * pi))
        if add_noise:
            train_y.add_(torch.randn_like(train_x).mul_(0.1))
        test_x = torch.linspace(0, 1, 51, device=device)
        test_y = torch.sin(test_x * (2 * pi))
        return train_x, test_x, train_y, test_y

    def test_prior(self, cuda=False):
        train_x, test_x, train_y, test_y = self._get_data(cuda=cuda)
        # We're manually going to set the hyperparameters to be ridiculous
        likelihood = QExponentialLikelihood(
            noise_prior=SmoothedBoxPrior(exp(-3), exp(3), sigma=0.1),
            noise_constraint=Positive(),  # Prior for this test is looser than default bound
            power=torch.tensor(POWER),
        )
        qep_model = ExactQEPModel(None, None, likelihood)
        # Update lengthscale prior to accommodate extreme parameters
        qep_model.covar_module.base_kernel.register_prior(
            "lengthscale_prior", SmoothedBoxPrior(exp(-10), exp(10), sigma=0.5), "raw_lengthscale"
        )
        qep_model.mean_module.initialize(constant=1.5)
        qep_model.covar_module.base_kernel.initialize(lengthscale=1)
        likelihood.initialize(noise=0)

        if cuda:
            qep_model.cuda()
            likelihood.cuda()

        # Compute posterior distribution
        qep_model.eval()
        likelihood.eval()

        # The model should predict in prior mode
        function_predictions = likelihood(qep_model(train_x))
        correct_variance = qep_model.covar_module.outputscale + likelihood.noise

        self.assertAllClose(function_predictions.mean, torch.full_like(function_predictions.mean, fill_value=1.5))
        self.assertAllClose(
            function_predictions.variance, correct_variance.squeeze().expand_as(function_predictions.variance)
        )

    def test_prior_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_prior(cuda=True)

    def test_recursive_initialize(self, cuda=False):
        train_x, test_x, train_y, test_y = self._get_data(cuda=cuda)

        likelihood_1 = QExponentialLikelihood(power=torch.tensor(POWER))
        qep_model_1 = ExactQEPModel(train_x, train_y, likelihood_1)

        likelihood_2 = QExponentialLikelihood(power=torch.tensor(POWER))
        qep_model_2 = ExactQEPModel(train_x, train_y, likelihood_2)

        qep_model_1.initialize(**{"likelihood.noise": 1e-2, "covar_module.base_kernel.lengthscale": 1e-1})
        qep_model_2.likelihood.initialize(noise=1e-2)
        qep_model_2.covar_module.base_kernel.initialize(lengthscale=1e-1)
        self.assertTrue(torch.equal(qep_model_1.likelihood.noise, qep_model_2.likelihood.noise))
        self.assertTrue(
            torch.equal(
                qep_model_1.covar_module.base_kernel.lengthscale, qep_model_2.covar_module.base_kernel.lengthscale
            )
        )

    def test_posterior_latent_qep_and_likelihood_without_optimization(self, cuda=False):
        warnings.simplefilter("ignore", qpytorch.utils.warnings.NumericalWarning)
        train_x, test_x, train_y, test_y = self._get_data(cuda=cuda)
        # We're manually going to set the hyperparameters to be ridiculous
        likelihood = QExponentialLikelihood(noise_constraint=Positive(), power=torch.tensor(POWER))  # This test actually wants a noise < 1e-4
        qep_model = ExactQEPModel(train_x, train_y, likelihood)
        qep_model.covar_module.base_kernel.initialize(lengthscale=exp(-15))
        likelihood.initialize(noise=exp(-15))

        if cuda:
            qep_model.cuda()
            likelihood.cuda()

        # Compute posterior distribution
        qep_model.eval()
        likelihood.eval()

        # Let's see how our model does, conditioned with weird hyperparams
        # The posterior should fit all the data
        with qpytorch.settings.debug(False):
            function_predictions = likelihood(qep_model(train_x))

        self.assertAllClose(function_predictions.mean, train_y)
        self.assertAllClose(function_predictions.variance, torch.zeros_like(function_predictions.variance))

        # It shouldn't fit much else though
        test_function_predictions = qep_model(torch.tensor([1.1]).type_as(test_x))

        self.assertAllClose(test_function_predictions.mean, torch.zeros_like(test_function_predictions.mean))
        self.assertAllClose(
            test_function_predictions.variance,
            qep_model.covar_module.outputscale.expand_as(test_function_predictions.variance),
        )

    def test_posterior_latent_qep_and_likelihood_without_optimization_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_posterior_latent_qep_and_likelihood_without_optimization(cuda=True)

    def test_qep_posterior_mean_skip_variances_fast_cuda(self):
        if not torch.cuda.is_available():
            return
        with least_used_cuda_device():
            train_x, test_x, train_y, _ = self._get_data(cuda=True)
            likelihood = QExponentialLikelihood(power=torch.tensor(POWER))
            qep_model = ExactQEPModel(train_x, train_y, likelihood)

            qep_model.cuda()
            likelihood.cuda()

            # Compute posterior distribution
            qep_model.eval()
            likelihood.eval()

            with qpytorch.settings.skip_posterior_variances(True):
                mean_skip_var = qep_model(test_x).mean
            mean = qep_model(test_x).mean
            likelihood_mean = likelihood(qep_model(test_x)).mean

            self.assertTrue(torch.allclose(mean_skip_var, mean))
            self.assertTrue(torch.allclose(mean_skip_var, likelihood_mean))

    def test_qep_posterior_mean_skip_variances_slow_cuda(self):
        if not torch.cuda.is_available():
            return
        with least_used_cuda_device():
            train_x, test_x, train_y, _ = self._get_data(cuda=True)
            likelihood = QExponentialLikelihood(power=torch.tensor(POWER))
            qep_model = ExactQEPModel(train_x, train_y, likelihood)

            qep_model.cuda()
            likelihood.cuda()

            # Compute posterior distribution
            qep_model.eval()
            likelihood.eval()

            with qpytorch.settings.fast_pred_var(False):
                with qpytorch.settings.skip_posterior_variances(True):
                    mean_skip_var = qep_model(test_x).mean
                mean = qep_model(test_x).mean
                likelihood_mean = likelihood(qep_model(test_x)).mean
            self.assertTrue(torch.allclose(mean_skip_var, mean))
            self.assertTrue(torch.allclose(mean_skip_var, likelihood_mean))

    def test_qep_posterior_single_training_point_smoke_test(self):
        train_x, test_x, train_y, _ = self._get_data()
        train_x = train_x[0].unsqueeze(-1).unsqueeze(-1)
        train_y = train_y[0].unsqueeze(-1)
        likelihood = QExponentialLikelihood(power=torch.tensor(POWER))
        qep_model = ExactQEPModel(train_x, train_y, likelihood)

        qep_model.eval()
        likelihood.eval()

        with qpytorch.settings.fast_pred_var():
            preds = qep_model(test_x)
            single_mean = preds.mean
            single_variance = preds.variance

        self.assertFalse(torch.any(torch.isnan(single_variance)))
        self.assertFalse(torch.any(torch.isnan(single_mean)))

        qep_model.train()
        qep_model.eval()

        preds = qep_model(test_x)
        single_mean = preds.mean
        single_variance = preds.variance

        self.assertFalse(torch.any(torch.isnan(single_variance)))
        self.assertFalse(torch.any(torch.isnan(single_mean)))

    def test_posterior_latent_qep_and_likelihood_with_optimization(self, cuda=False, checkpoint=0):
        train_x, test_x, train_y, test_y = self._get_data(
            cuda=cuda, num_data=(11), add_noise=bool(checkpoint)
        )
        # We're manually going to set the hyperparameters to something they shouldn't be
        likelihood = QExponentialLikelihood(noise_prior=SmoothedBoxPrior(exp(-3), exp(3), sigma=0.1), power=torch.tensor(POWER))
        qep_model = ExactQEPModel(train_x, train_y, likelihood)
        mll = qpytorch.ExactMarginalLogLikelihood(likelihood, qep_model)
        qep_model.covar_module.base_kernel.initialize(lengthscale=exp(1))
        qep_model.mean_module.initialize(constant=0)
        likelihood.initialize(noise=exp(1))

        if cuda:
            qep_model.cuda()
            likelihood.cuda()

        # Find optimal model hyperparameters
        qep_model.train()
        likelihood.train()
        optimizer = optim.Adam(qep_model.parameters(), lr=0.15)
        with qpytorch.settings.fast_pred_var():
            for _ in range(50):
                optimizer.zero_grad()
                output = qep_model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

            for param in qep_model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)
            optimizer.step()

        # Test the model
        qep_model.eval()
        likelihood.eval()
        with qpytorch.settings.skip_posterior_variances(True):
            test_function_predictions = likelihood(qep_model(test_x))
        mean_abs_error = torch.mean(torch.abs(test_y - test_function_predictions.mean))

        self.assertLess(mean_abs_error.item(), 0.05)

    def test_fantasy_updates_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_fantasy_updates(cuda=True)

    def test_fantasy_updates(self, cuda=False):
        train_x, test_x, train_y, test_y = self._get_data(cuda=cuda)
        # We're manually going to set the hyperparameters to something they shouldn't be
        likelihood = QExponentialLikelihood(power=torch.tensor(POWER))
        qep_model = ExactQEPModel(train_x, train_y, likelihood)
        mll = qpytorch.ExactMarginalLogLikelihood(likelihood, qep_model)
        qep_model.covar_module.base_kernel.initialize(lengthscale=exp(1))
        qep_model.mean_module.initialize(constant=0)
        likelihood.initialize(noise=exp(1))

        if cuda:
            qep_model.cuda()
            likelihood.cuda()

        # Find optimal model hyperparameters
        qep_model.train()
        likelihood.train()
        optimizer = optim.Adam(qep_model.parameters(), lr=0.15)
        for _ in range(50):
            optimizer.zero_grad()
            with qpytorch.settings.debug(False):
                output = qep_model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        for param in qep_model.parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        optimizer.step()

        train_x.requires_grad = True
        qep_model.set_train_data(train_x, train_y)
        with qpytorch.settings.fast_pred_var(), qpytorch.settings.detach_test_caches(False):
            # Test the model
            qep_model.eval()
            likelihood.eval()
            test_function_predictions = likelihood(qep_model(test_x))
            test_function_predictions.mean.sum().backward()

            real_fant_x_grad = train_x.grad[5:].clone()
            train_x.grad = None
            train_x.requires_grad = False
            qep_model.set_train_data(train_x, train_y)

            # Cut data down, and then add back via the fantasy interface
            qep_model.set_train_data(train_x[:5], train_y[:5], strict=False)
            likelihood(qep_model(test_x))

            fantasy_x = train_x[5:].clone().detach().requires_grad_(True)
            fant_model = qep_model.get_fantasy_model(fantasy_x, train_y[5:])
            fant_function_predictions = likelihood(fant_model(test_x))

            self.assertAllClose(test_function_predictions.mean, fant_function_predictions.mean, atol=1e-4)

            fant_function_predictions.mean.sum().backward()
            self.assertTrue(fantasy_x.grad is not None)

            relative_error = torch.norm(real_fant_x_grad - fantasy_x.grad) / fantasy_x.grad.norm()
            self.assertLess(relative_error, 15e-1)  # This was only passing by a hair before

    def test_fantasy_updates_batch_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_fantasy_updates_batch(cuda=True)

    def test_fantasy_updates_batch(self, cuda=False):
        train_x, test_x, train_y, test_y = self._get_data(cuda=cuda)
        # We're manually going to set the hyperparameters to something they shouldn't be
        likelihood = QExponentialLikelihood(power=torch.tensor(POWER))
        qep_model = ExactQEPModel(train_x, train_y, likelihood)
        mll = qpytorch.ExactMarginalLogLikelihood(likelihood, qep_model)
        qep_model.covar_module.base_kernel.initialize(lengthscale=exp(1))
        qep_model.mean_module.initialize(constant=0)
        likelihood.initialize(noise=exp(1))

        if cuda:
            qep_model.cuda()
            likelihood.cuda()

        # Find optimal model hyperparameters
        qep_model.train()
        likelihood.train()
        optimizer = optim.Adam(qep_model.parameters(), lr=0.15)
        for _ in range(50):
            optimizer.zero_grad()
            with qpytorch.settings.debug(False):
                output = qep_model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        for param in qep_model.parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        optimizer.step()

        with qpytorch.settings.fast_pred_var():
            # Test the model
            qep_model.eval()
            likelihood.eval()
            test_function_predictions = likelihood(qep_model(test_x))

            # Cut data down, and then add back via the fantasy interface
            qep_model.set_train_data(train_x[:5], train_y[:5], strict=False)
            likelihood(qep_model(test_x))

            fantasy_x = train_x[5:].clone().unsqueeze(0).unsqueeze(-1).repeat(3, 1, 1).requires_grad_(True)
            fantasy_y = train_y[5:].unsqueeze(0).repeat(3, 1)
            fant_model = qep_model.get_fantasy_model(fantasy_x, fantasy_y)
            fant_function_predictions = likelihood(fant_model(test_x))

            self.assertAllClose(test_function_predictions.mean, fant_function_predictions.mean[0], atol=1e-4)

            fant_function_predictions.mean.sum().backward()
            self.assertTrue(fantasy_x.grad is not None)

    def test_posterior_latent_qep_and_likelihood_with_optimization_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_posterior_latent_qep_and_likelihood_with_optimization(cuda=True)

    def test_posterior_with_exact_computations(self):
        with qpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False):
            self.test_posterior_latent_qep_and_likelihood_with_optimization(cuda=False)

    def test_posterior_with_exact_computations_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                with qpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False):
                    self.test_posterior_latent_qep_and_likelihood_with_optimization(cuda=True)

    def test_posterior_latent_qep_and_likelihood_fast_pred_var(self, cuda=False):
        train_x, test_x, train_y, test_y = self._get_data(cuda=cuda)
        with qpytorch.settings.fast_pred_var(), qpytorch.settings.debug(False):
            # We're manually going to set the hyperparameters to
            # something they shouldn't be
            likelihood = QExponentialLikelihood(noise_prior=SmoothedBoxPrior(exp(-3), exp(3), sigma=0.1), power=torch.tensor(POWER))
            qep_model = ExactQEPModel(train_x, train_y, likelihood)
            mll = qpytorch.mlls.ExactMarginalLogLikelihood(likelihood, qep_model)
            qep_model.covar_module.base_kernel.initialize(lengthscale=exp(1))
            qep_model.mean_module.initialize(constant=0)
            likelihood.initialize(noise=exp(1))

            if cuda:
                qep_model.cuda()
                likelihood.cuda()

            # Find optimal model hyperparameters
            qep_model.train()
            likelihood.train()
            optimizer = optim.Adam(qep_model.parameters(), lr=0.1)
            for _ in range(50):
                optimizer.zero_grad()
                output = qep_model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

            for param in qep_model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)
            optimizer.step()

            # Test the model
            qep_model.eval()
            likelihood.eval()
            # Set the cache
            test_function_predictions = likelihood(qep_model(train_x))

            # Now bump up the likelihood to something huge
            # This will make it easy to calculate the variance
            likelihood.noise_covar.raw_noise.data.fill_(3)
            test_function_predictions = likelihood(qep_model(train_x))

            noise = likelihood.noise_covar.noise
            var_diff = (test_function_predictions.variance - noise).abs()

            self.assertLess(torch.max(var_diff / noise), 0.05)

    def test_pyro_sampling(self):
        try:
            import pyro  # noqa
            from pyro.infer.mcmc import MCMC, NUTS
        except ImportError:
            return
        train_x, test_x, train_y, test_y = self._get_data(cuda=False)
        likelihood = QExponentialLikelihood(noise_constraint=qpytorch.constraints.Positive(), power=torch.tensor(POWER))
        qep_model = ExactQEPModel(train_x, train_y, likelihood)

        # Register normal QEPyTorch priors
        qep_model.mean_module.register_prior("mean_prior", UniformPrior(-1, 1), "constant")
        qep_model.covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.01, 0.5), "lengthscale")
        qep_model.covar_module.register_prior("outputscale_prior", UniformPrior(1, 2), "outputscale")
        likelihood.register_prior("noise_prior", UniformPrior(0.05, 0.3), "noise")

        def pyro_model(x, y):
            with qpytorch.settings.fast_computations(False, False, False):
                sampled_model = qep_model.pyro_sample_from_prior()
                output = sampled_model.likelihood(sampled_model(x))
                pyro.sample("obs", output, obs=y)
            return y

        nuts_kernel = NUTS(pyro_model, adapt_step_size=True)
        mcmc_run = MCMC(nuts_kernel, num_samples=3, warmup_steps=20, disable_progbar=True)
        mcmc_run.run(train_x, train_y)

        qep_model.pyro_load_from_samples(mcmc_run.get_samples())

        qep_model.eval()
        expanded_test_x = test_x.unsqueeze(-1).repeat(3, 1, 1)
        output = qep_model(expanded_test_x)

        self.assertEqual(output.mean.size(0), 3)

        # All 3 samples should do reasonably well on a noiseless dataset.
        self.assertLess(torch.norm(output.mean[0] - test_y) / test_y.norm(), 0.2)
        self.assertLess(torch.norm(output.mean[1] - test_y) / test_y.norm(), 0.2)
        self.assertLess(torch.norm(output.mean[2] - test_y) / test_y.norm(), 0.2)

    def test_posterior_latent_qep_and_likelihood_fast_pred_var_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_posterior_latent_qep_and_likelihood_fast_pred_var(cuda=True)


if __name__ == "__main__":
    unittest.main()
