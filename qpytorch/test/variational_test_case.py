#!/usr/bin/env python3

from abc import abstractproperty
from unittest.mock import MagicMock, patch

import linear_operator
import torch

import qpytorch

from gpytorch.test.base_test_case import BaseTestCase


class VariationalTestCase(BaseTestCase):
    def _make_model_and_likelihood(
        self,
        num_inducing=16,
        batch_shape=torch.Size([]),
        inducing_batch_shape=torch.Size([]),
        strategy_cls=qpytorch.variational.VariationalStrategy,
        distribution_cls=qpytorch.variational.CholeskyVariationalDistribution,
        constant_mean=True,
    ):
        _power = getattr(self, '_power', 2.0)
        class _SV_PRegressionModel(qpytorch.models.ApproximateGP if _power==2 else qpytorch.models.ApproximateQEP):
            def __init__(self, inducing_points):
                if _power!=2: self.power = torch.tensor(_power)
                variational_distribution = distribution_cls(num_inducing, batch_shape=batch_shape, power=self.power) if hasattr(self, 'power') \
                                           else distribution_cls(num_inducing, batch_shape=batch_shape)
                variational_strategy = strategy_cls(
                    self,
                    inducing_points,
                    variational_distribution,
                    learn_inducing_locations=True,
                )
                super().__init__(variational_strategy)
                if constant_mean:
                    self.mean_module = qpytorch.means.ConstantMean()
                    self.mean_module.initialize(constant=1.0)
                else:
                    self.mean_module = qpytorch.means.ZeroMean()
                self.covar_module = qpytorch.kernels.ScaleKernel(qpytorch.kernels.RBFKernel())

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                latent_pred = qpytorch.distributions.MultivariateQExponential(mean_x, covar_x, power=self.power) if hasattr(self, 'power') \
                              else qpytorch.distributions.MultivariateNormal(mean_x, covar_x)
                return latent_pred

        inducing_points = torch.randn(num_inducing, 2).repeat(*inducing_batch_shape, 1, 1)
        return _SV_PRegressionModel(inducing_points), self.likelihood_cls()

    def _training_iter(
        self,
        model,
        likelihood,
        batch_shape=torch.Size([]),
        mll_cls=qpytorch.mlls.VariationalELBO,
        cuda=False,
    ):
        train_x = torch.randn(*batch_shape, 32, 2).clamp(-2.5, 2.5)
        train_y = torch.linspace(-1, 1, self.event_shape[0])
        train_y = train_y.view(self.event_shape[0], *([1] * (len(self.event_shape) - 1)))
        train_y = train_y.expand(*self.event_shape)
        mll = mll_cls(likelihood, model, num_data=train_x.size(-2))
        if cuda:
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            model = model.cuda()
            likelihood = likelihood.cuda()

        # Single optimization iteration
        model.train()
        likelihood.train()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.sum().backward()

        # Make sure we have gradients for all parameters
        for _, param in model.named_parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        for _, param in likelihood.named_parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)

        return output, loss

    def _eval_iter(self, model, batch_shape=torch.Size([]), cuda=False):
        test_x = torch.randn(*batch_shape, 32, 2).clamp(-2.5, 2.5)
        if cuda:
            test_x = test_x.cuda()
            model = model.cuda()

        # Single optimization iteration
        model.eval()
        with torch.no_grad():
            output = model(test_x)

        return output

    def _fantasy_iter(
        self,
        model,
        likelihood,
        batch_shape=torch.Size([]),
        cuda=False,
        num_fant=10,
        covar_module=None,
        mean_module=None,
    ):
        model.likelihood = likelihood
        val_x = torch.randn(*batch_shape, num_fant, 2).clamp(-2.5, 2.5)
        val_y = torch.linspace(-1, 1, num_fant)
        val_y = val_y.view(num_fant, *([1] * (len(self.event_shape) - 1)))
        val_y = val_y.expand(*batch_shape, num_fant, *self.event_shape[1:])
        if cuda:
            model = model.cuda()
            val_x = val_x.cuda()
            val_y = val_y.cuda()
        updated_model = model.get_fantasy_model(val_x, val_y, covar_module=covar_module, mean_module=mean_module)
        return updated_model

    @abstractproperty
    def batch_shape(self):
        raise NotImplementedError

    @abstractproperty
    def distribution_cls(self):
        raise NotImplementedError

    @property
    def event_shape(self):
        return torch.Size([32])

    @property
    def likelihood_cls(self):
        return qpytorch.likelihoods.GaussianLikelihood if self._power==2 else qpytorch.likelihoods.QExponentialLikelihood

    @abstractproperty
    def mll_cls(self):
        raise NotImplementedError

    @abstractproperty
    def strategy_cls(self):
        raise NotImplementedError

    @property
    def cuda(self):
        return False

    def test_eval_iteration(
        self,
        data_batch_shape=None,
        inducing_batch_shape=None,
        model_batch_shape=None,
        eval_data_batch_shape=None,
        expected_batch_shape=None,
    ):
        # Batch shapes
        model_batch_shape = model_batch_shape if model_batch_shape is not None else self.batch_shape
        data_batch_shape = data_batch_shape if data_batch_shape is not None else self.batch_shape
        inducing_batch_shape = inducing_batch_shape if inducing_batch_shape is not None else self.batch_shape
        expected_batch_shape = expected_batch_shape if expected_batch_shape is not None else self.batch_shape
        eval_data_batch_shape = eval_data_batch_shape if eval_data_batch_shape is not None else self.batch_shape

        # Mocks
        _wrapped_cholesky = MagicMock(wraps=torch.linalg.cholesky_ex)
        _wrapped_cg = MagicMock(wraps=linear_operator.utils.linear_cg)
        _wrapped_ciq = MagicMock(wraps=linear_operator.utils.contour_integral_quad)
        _cholesky_mock = patch("torch.linalg.cholesky_ex", new=_wrapped_cholesky)
        _cg_mock = patch("linear_operator.utils.linear_cg", new=_wrapped_cg)
        _ciq_mock = patch("linear_operator.utils.contour_integral_quad", new=_wrapped_ciq)

        # Make model and likelihood
        model, likelihood = self._make_model_and_likelihood(
            batch_shape=model_batch_shape,
            inducing_batch_shape=inducing_batch_shape,
            distribution_cls=self.distribution_cls,
            strategy_cls=self.strategy_cls,
        )

        # Do one forward pass
        self._training_iter(model, likelihood, data_batch_shape, mll_cls=self.mll_cls, cuda=self.cuda)

        # Now do evaluation
        with _cholesky_mock as cholesky_mock, _cg_mock as cg_mock, _ciq_mock as ciq_mock:
            # Iter 1
            _ = self._eval_iter(model, eval_data_batch_shape, cuda=self.cuda)
            output = self._eval_iter(model, eval_data_batch_shape, cuda=self.cuda)
            self.assertEqual(output.batch_shape, expected_batch_shape)
            self.assertEqual(output.event_shape, self.event_shape)
            return cg_mock, cholesky_mock, ciq_mock

    def test_eval_smaller_pred_batch(self):
        return self.test_eval_iteration(
            model_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
            inducing_batch_shape=(torch.Size([3, 1]) + self.batch_shape),
            data_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
            eval_data_batch_shape=(torch.Size([4]) + self.batch_shape),
            expected_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
        )

    def test_eval_larger_pred_batch(self):
        return self.test_eval_iteration(
            model_batch_shape=(torch.Size([4]) + self.batch_shape),
            inducing_batch_shape=(self.batch_shape),
            data_batch_shape=(torch.Size([4]) + self.batch_shape),
            eval_data_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
            expected_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
        )

    def test_training_iteration(
        self,
        data_batch_shape=None,
        inducing_batch_shape=None,
        model_batch_shape=None,
        expected_batch_shape=None,
        constant_mean=True,
    ):
        # Batch shapes
        model_batch_shape = model_batch_shape if model_batch_shape is not None else self.batch_shape
        data_batch_shape = data_batch_shape if data_batch_shape is not None else self.batch_shape
        inducing_batch_shape = inducing_batch_shape if inducing_batch_shape is not None else self.batch_shape
        expected_batch_shape = expected_batch_shape if expected_batch_shape is not None else self.batch_shape

        # Mocks
        _wrapped_cholesky = MagicMock(wraps=torch.linalg.cholesky_ex)
        _wrapped_cg = MagicMock(wraps=linear_operator.utils.linear_cg)
        _wrapped_ciq = MagicMock(wraps=linear_operator.utils.contour_integral_quad)
        _cholesky_mock = patch("torch.linalg.cholesky_ex", new=_wrapped_cholesky)
        _cg_mock = patch("linear_operator.utils.linear_cg", new=_wrapped_cg)
        _ciq_mock = patch("linear_operator.utils.contour_integral_quad", new=_wrapped_ciq)

        # Make model and likelihood
        model, likelihood = self._make_model_and_likelihood(
            batch_shape=model_batch_shape,
            inducing_batch_shape=inducing_batch_shape,
            distribution_cls=self.distribution_cls,
            strategy_cls=self.strategy_cls,
            constant_mean=constant_mean,
        )

        # Do forward pass
        with _cholesky_mock as cholesky_mock, _cg_mock as cg_mock, _ciq_mock as ciq_mock:
            # Iter 1
            self.assertEqual(model.variational_strategy.variational_params_initialized.item(), 0)
            self._training_iter(
                model,
                likelihood,
                data_batch_shape,
                mll_cls=self.mll_cls,
                cuda=self.cuda,
            )
            self.assertEqual(model.variational_strategy.variational_params_initialized.item(), 1)
            # Iter 2
            output, loss = self._training_iter(
                model,
                likelihood,
                data_batch_shape,
                mll_cls=self.mll_cls,
                cuda=self.cuda,
            )
            self.assertEqual(output.batch_shape, expected_batch_shape)
            self.assertEqual(output.event_shape, self.event_shape)
            self.assertEqual(loss.shape, expected_batch_shape)
            return cg_mock, cholesky_mock, ciq_mock

    def test_training_iteration_batch_inducing(self):
        return self.test_training_iteration(
            model_batch_shape=(torch.Size([3]) + self.batch_shape),
            data_batch_shape=self.batch_shape,
            inducing_batch_shape=(torch.Size([3]) + self.batch_shape),
            expected_batch_shape=(torch.Size([3]) + self.batch_shape),
        )

    def test_training_iteration_batch_data(self):
        return self.test_training_iteration(
            model_batch_shape=self.batch_shape,
            inducing_batch_shape=self.batch_shape,
            data_batch_shape=(torch.Size([3]) + self.batch_shape),
            expected_batch_shape=(torch.Size([3]) + self.batch_shape),
        )

    def test_training_iteration_batch_model(self):
        return self.test_training_iteration(
            model_batch_shape=(torch.Size([3]) + self.batch_shape),
            inducing_batch_shape=self.batch_shape,
            data_batch_shape=self.batch_shape,
            expected_batch_shape=(torch.Size([3]) + self.batch_shape),
        )

    def test_training_all_batch_zero_mean(self):
        return self.test_training_iteration(
            model_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
            inducing_batch_shape=(torch.Size([3, 1]) + self.batch_shape),
            data_batch_shape=(torch.Size([4]) + self.batch_shape),
            expected_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
            constant_mean=False,
        )

    def test_fantasy_call(
        self,
        data_batch_shape=None,
        inducing_batch_shape=None,
        model_batch_shape=None,
        expected_batch_shape=None,
        constant_mean=True,
    ):
        # Batch shapes
        model_batch_shape = model_batch_shape if model_batch_shape is not None else self.batch_shape
        data_batch_shape = data_batch_shape if data_batch_shape is not None else self.batch_shape
        inducing_batch_shape = inducing_batch_shape if inducing_batch_shape is not None else self.batch_shape
        expected_batch_shape = expected_batch_shape if expected_batch_shape is not None else self.batch_shape

        num_inducing = 16
        num_fant = 10
        # Make model and likelihood
        model, likelihood = self._make_model_and_likelihood(
            batch_shape=model_batch_shape,
            inducing_batch_shape=inducing_batch_shape,
            distribution_cls=self.distribution_cls,
            strategy_cls=self.strategy_cls,
            constant_mean=constant_mean,
            num_inducing=num_inducing,
        )

        # we iterate through the covar and mean module possible settings
        covar_mean_options = [
            {"covar_module": None, "mean_module": None},
            {"covar_module": qpytorch.kernels.MaternKernel(), "mean_module": qpytorch.means.ZeroMean()},
        ]
        for cm_dict in covar_mean_options:
            fant_model = self._fantasy_iter(
                model, likelihood, data_batch_shape, self.cuda, num_fant=num_fant, **cm_dict
            )
            self.assertTrue(isinstance(fant_model, {2.0: qpytorch.models.ExactGP, 1.0: qpytorch.models.ExactQEP}[self._power]))

            # we check to ensure setting the covar_module and mean_modules are okay
            if cm_dict["covar_module"] is None:
                self.assertEqual(type(fant_model.covar_module), type(model.covar_module))
            else:
                self.assertNotEqual(type(fant_model.covar_module), type(model.covar_module))
            if cm_dict["mean_module"] is None:
                self.assertEqual(type(fant_model.mean_module), type(model.mean_module))
            else:
                self.assertNotEqual(type(fant_model.mean_module), type(model.mean_module))

            # now we check to ensure the shapes of the fantasy strategy are correct
            self.assertTrue(fant_model.prediction_strategy is not None)
            for key in fant_model.prediction_strategy._memoize_cache.keys():
                if key[0] == "mean_cache":
                    break
            mean_cache = fant_model.prediction_strategy._memoize_cache[key]
            self.assertEqual(mean_cache.shape, torch.Size([*expected_batch_shape, num_inducing + num_fant]))

        # we remove the mean_module and covar_module and check for errors
        del model.mean_module
        with self.assertRaises(ModuleNotFoundError):
            self._fantasy_iter(model, likelihood, data_batch_shape, self.cuda, num_fant=num_fant)

        model.mean_module = qpytorch.means.ZeroMean()
        del model.covar_module
        with self.assertRaises(ModuleNotFoundError):
            self._fantasy_iter(model, likelihood, data_batch_shape, self.cuda, num_fant=num_fant)

        # finally we check to ensure failure for a non-gaussian likelihood
        with self.assertRaises(NotImplementedError):
            self._fantasy_iter(
                model,
                qpytorch.likelihoods.BernoulliLikelihood(),
                data_batch_shape,
                self.cuda,
                num_fant=num_fant,
            )

    def test_fantasy_call_batch_inducing(self):
        return self.test_fantasy_call(
            model_batch_shape=(torch.Size([3]) + self.batch_shape),
            data_batch_shape=self.batch_shape,
            inducing_batch_shape=(torch.Size([3]) + self.batch_shape),
            expected_batch_shape=(torch.Size([3]) + self.batch_shape),
        )

    def test_fantasy_call_batch_data(self):
        return self.test_fantasy_call(
            model_batch_shape=self.batch_shape,
            inducing_batch_shape=self.batch_shape,
            data_batch_shape=(torch.Size([3]) + self.batch_shape),
            expected_batch_shape=(torch.Size([3]) + self.batch_shape),
        )

    def test_fantasy_call_batch_model(self):
        return self.test_fantasy_call(
            model_batch_shape=(torch.Size([3]) + self.batch_shape),
            inducing_batch_shape=self.batch_shape,
            data_batch_shape=self.batch_shape,
            expected_batch_shape=(torch.Size([3]) + self.batch_shape),
        )
