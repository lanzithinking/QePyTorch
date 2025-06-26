from typing import Union
import unittest

import torch

from qpytorch import settings
from qpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal, MultivariateQExponential, MultitaskMultivariateQExponential
from qpytorch.kernels import ScaleKernel, RBFKernel, MultitaskKernel
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from qpytorch.likelihoods import Likelihood, QExponentialLikelihood, MultitaskQExponentialLikelihood
from qpytorch.means import ConstantMean, MultitaskMean
from gpytorch.models import GP, ExactGP, VariationalGP
from qpytorch.models import QEP, ExactQEP, VariationalQEP
from qpytorch.test import BaseTestCase
from gpytorch.utils.memoize import clear_cache_hook
from qpytorch.variational import CholeskyVariationalDistribution, LMCVariationalStrategy, VariationalStrategy

POWER = 2.0
mod = 'gpytorch' if POWER==2 else 'qpytorch'
ExactMarginalLogLikelihood = __import__(mod).ExactMarginalLogLikelihood
exec(f"from {mod+'.mlls'} import {'PredictiveLogLikelihood, MarginalLogLikelihood, VariationalELBO'}")

class SingleModel(ExactGP if POWER==2 else ExactQEP):
    def __init__(self, train_inputs, train_targets, likelihood, batch_shape):
        super(SingleModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(RBFKernel(batch_shape=batch_shape))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x) if not hasattr(self.likelihood, 'power') \
               else MultivariateQExponential(mean_x, covar_x, power=self.likelihood.power)


class MultitaskModel(ExactGP if POWER==2 else ExactQEP):
    def __init__(self, train_inputs, train_targets, likelihood, num_tasks):
        super(MultitaskModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=num_tasks)
        self.covar_module = MultitaskKernel(ScaleKernel(RBFKernel()), num_tasks=num_tasks)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x) if not hasattr(self.likelihood, 'power') \
               else MultitaskMultivariateQExponential(mean_x, covar_x, power=self.likelihood.power)


class MultitaskVariationalModel(VariationalGP if POWER==2 else VariationalQEP):
    def __init__(self, num_latents, num_tasks):
        self.power = torch.tensor(POWER)
        inducing_points = torch.rand(num_latents, 21, 1)
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents]), power=self.power
        )
        variational_strategy = LMCVariationalStrategy(
            VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )
        super().__init__(variational_strategy)
        self.mean_module = ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x) if self.power==2 \
               else MultivariateQExponential(mean_x, covar_x, power=self.power)


class TestMissingData(BaseTestCase, unittest.TestCase):
    seed = 20
    warning = "Observation NaN policy 'fill' makes the kernel matrix dense during exact prediction."

    def _check(
        self,
        model: Union[GP, QEP],
        likelihood: Likelihood,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        test_x: torch.Tensor,
        test_y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        mll: MarginalLogLikelihood,
        epochs: int = 30,
        atol: float = 0.2
    ) -> None:
        model.train()
        likelihood.train()

        for _ in range(epochs):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y).sum()
            self.assertFalse(torch.any(torch.isnan(output.mean)).item())
            self.assertFalse(torch.any(torch.isnan(output.covariance_matrix)).item())
            self.assertFalse(torch.isnan(loss).item())
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()

        with torch.no_grad():
            if isinstance(model, (ExactGP, ExactQEP)):
                self._check_predictions_exact(model, test_x, test_y, atol)
            else:
                prediction = model(test_x)
                self._check_prediction(prediction, test_y, atol)

    def _check_predictions_exact(self, model: Union[ExactGP, ExactQEP], test_x: torch.Tensor, test_y: torch.Tensor, atol: float):
        with settings.observation_nan_policy("mask"):
            prediction = model(test_x)
            self._check_prediction(prediction, test_y, atol)

        clear_cache_hook(model.prediction_strategy)

        with settings.observation_nan_policy("fill"), self.assertWarns(RuntimeWarning, msg=self.warning):
            prediction = model(test_x)
            self._check_prediction(prediction, test_y, atol)

        clear_cache_hook(model.prediction_strategy)

        with settings.observation_nan_policy("mask"):
            model(test_x)
        with settings.observation_nan_policy("fill"), self.assertWarns(RuntimeWarning, msg=self.warning):
            prediction = model(test_x)
            self._check_prediction(prediction, test_y, atol)

        clear_cache_hook(model.prediction_strategy)

        with settings.observation_nan_policy("fill"), self.assertWarns(RuntimeWarning, msg=self.warning):
            model(test_x)
        with settings.observation_nan_policy("mask"):
            prediction = model(test_x)
            self._check_prediction(prediction, test_y, atol)

    def _check_prediction(self, prediction: Union[MultivariateNormal, MultivariateQExponential], test_y: torch.Tensor, atol: float):
        self.assertFalse(torch.any(torch.isnan(prediction.mean)).item())
        self.assertFalse(torch.any(torch.isnan(prediction.covariance_matrix)).item())
        self.assertAllClose(prediction.mean, test_y, atol=atol)

    def test_single(self):
        train_x = torch.linspace(0, 1, 41)
        test_x = torch.linspace(0, 1, 51)
        train_y = torch.sin(2 * torch.pi * train_x).squeeze()
        train_y += torch.normal(0, 0.01, train_y.shape)
        test_y = torch.sin(2 * torch.pi * test_x).squeeze()
        train_y[::4] = torch.nan

        batch_shape = torch.Size(())
        likelihood = GaussianLikelihood(batch_shape=batch_shape) if POWER==2 \
                     else QExponentialLikelihood(batch_shape=batch_shape, power=torch.tensor(POWER))
        model = SingleModel(train_x, train_y, likelihood, batch_shape=batch_shape)

        mll = ExactMarginalLogLikelihood(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.15)

        with settings.observation_nan_policy("mask"):
            self._check(model, likelihood, train_x, train_y, test_x, test_y, optimizer, mll)

    def test_single_batch(self):
        train_x = torch.stack([torch.linspace(0, 1, 41), torch.linspace(1, 2, 41)]).reshape(2, 41, 1)
        test_x = torch.stack([torch.linspace(0, 1, 51), torch.linspace(1, 2, 51)]).reshape(2, 51, 1)
        train_y = torch.sin(2 * torch.pi * train_x).squeeze()
        train_y += torch.normal(0, 0.01, train_y.shape)
        test_y = torch.sin(2 * torch.pi * test_x).squeeze()
        train_y[0, ::4] = torch.nan

        batch_shape = torch.Size((2,))
        likelihood = GaussianLikelihood(batch_shape=batch_shape) if POWER==2 \
                     else QExponentialLikelihood(batch_shape=batch_shape, power=torch.tensor(POWER))
        model = SingleModel(train_x, train_y, likelihood, batch_shape=batch_shape)

        mll = ExactMarginalLogLikelihood(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.15)

        with settings.observation_nan_policy("mask"):
            self._check(model, likelihood, train_x, train_y, test_x, test_y, optimizer, mll)

    def test_multitask(self):
        num_tasks = 10
        train_x = torch.linspace(0, 1, 41)
        test_x = torch.linspace(0, 1, 51)
        coefficients = torch.rand(1, num_tasks)
        train_y = torch.sin(2 * torch.pi * train_x)[:, None] * coefficients
        train_y += torch.normal(0, 0.01, train_y.shape)
        test_y = torch.sin(2 * torch.pi * test_x)[:, None] * coefficients
        train_y[::3, : num_tasks // 2] = torch.nan
        train_y[::4, num_tasks // 2 :] = torch.nan

        likelihood = MultitaskGaussianLikelihood(num_tasks) if POWER==2 \
                     else MultitaskQExponentialLikelihood(num_tasks, power=torch.tensor(POWER))
        model = MultitaskModel(train_x, train_y, likelihood, num_tasks)

        mll = ExactMarginalLogLikelihood(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.15)

        with settings.observation_nan_policy("mask"):
            self._check(model, likelihood, train_x, train_y, test_x, test_y, optimizer, mll)

    def test_variational_multitask(self):
        num_latents = 1
        train_x = torch.linspace(0, 1, 41)
        test_x = torch.linspace(0, 1, 51)
        train_y = torch.stack([
            torch.sin(train_x * (2 * torch.pi)) + torch.randn(train_x.size()) * 0.2,
            -torch.sin(train_x * (2 * torch.pi)) + torch.randn(train_x.size()) * 0.2,
        ], -1)
        test_y = torch.stack([
            torch.sin(test_x * (2 * torch.pi)),
            -torch.sin(test_x * (2 * torch.pi)),
        ], -1)
        num_tasks = train_y.shape[-1]

        # nan out a few train_y
        train_y[-3:, 1] = torch.nan

        likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks) if POWER==2 \
                     else MultitaskQExponentialLikelihood(num_tasks, power=torch.tensor(POWER))
        model = MultitaskVariationalModel(num_latents, num_tasks)
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.05)

        mll = VariationalELBO(likelihood, model, num_data=train_y.size(0))
        with settings.observation_nan_policy("mask"):
            self._check(model, likelihood, train_x, train_y, test_x, test_y, optimizer, mll, epochs=50, atol=0.7)
        with settings.observation_nan_policy("fill"):
            self._check(model, likelihood, train_x, train_y, test_x, test_y, optimizer, mll, epochs=50, atol=0.3)

        mll = PredictiveLogLikelihood(likelihood, model, num_data=train_y.size(0))
        with settings.observation_nan_policy("mask"):
            self._check(model, likelihood, train_x, train_y, test_x, test_y, optimizer, mll, epochs=50, atol=0.7)
        with settings.observation_nan_policy("fill"):
            self._check(model, likelihood, train_x, train_y, test_x, test_y, optimizer, mll, epochs=50, atol=0.3)
