#!/usr/bin/env python3

import torch
from torch.autograd.functional import jacobian
from torch.func import vmap, jacrev
from linear_operator.operators import IdentityLinearOperator, DiagLinearOperator

from .approximate_qep import ApproximateQEP
from ..distributions import MultivariateQExponential

class PDESolver(ApproximateQEP):
    """
    The Bayesian solver to partial differential equation (PDE) based on q-exponential process (QEP).
    The class takes the left-hand side function of a PDE and propagates the approximate QEP distribution.
    
    :param lhs_f: a callable that returns the left-hand side of a PDE.
    :param ~qpytorch.variational._VariationalStrategy variational_strategy: The strategy that determines
        how the model marginalizes over the variational distribution (over inducing points)
        to produce the approximate posterior distribution (over data)
    """
    def __init__(self, lhs_f, variational_strategy):
        super().__init__(variational_strategy)
        self.lhs_f = lhs_f
        assert callable(self.lhs_f), "lhs_f must be a function that returns the left-hand side of a PDE."
    
    def forward(self):
        raise NotImplementedError
    
    def qloss(self, lhs, rhs, power=2.0):
        """
        Loss defined by Q-Exponential density.
        """
        qep = MultivariateQExponential(torch.zeros_like(lhs), IdentityLinearOperator(lhs.shape[-1], lhs.shape[:-1]), power=torch.tensor(power, device=lhs.device))
        return -qep.log_prob(rhs)
    
    def linearize_PDE(self, u0, batch_jacobian=False):
        """
        Linearize PDE left-hand side by Taylor expansion:
        
        .. math::
        
            P(\\tilde{u}) \\approx P(\\tilde{u}_0) + \\nabla P(\\tilde{u}_0)(\\tilde{u}-\\tilde{u}_0)
        
        where :math:`\tilde u_0` is the reference.
        
        :param u0: reference of linearization.
        :param batch_jacobian: If True, compute batch Jacobian. (Default: False.)
        :return: A tuple of value :math:`P(\tilde{u}_0)` and Jacobian :math:`\nabla P(\tilde{u}_0)`.
        """
        fun = self.lhs_f(u0)
        u0.requires_grad = True
        jac = vmap(jacrev(self.lhs_f))(u0) if batch_jacobian else jacobian(self.lhs_f, u0)
        return fun, jac
    
    def propagate(self, dist, u0=None, eps=1e-6, diag=True, **kwargs) -> MultivariateQExponential:
        """
        Propagate approximating (variational) distribution :math:`\tilde{u} \sim \textrm{q-ED}(\mu, \Sigma)` through linearization:
        
        .. math::
        
            P(\\tilde{u}) \\sim \\textrm{q-ED}(\\mu_*, \\Sigma_*), \\
            
            \\mu_* = P(\\tilde{u}_0) + \\nabla P(\\tilde{u}_0)(\\mu-\\tilde{u}_0), \\quad \\Sigma_* = P(\\tilde{u}_0) \\Sigma P(\\tilde{u}_0)^{\\mathsf T} + \\delta I.
        
        :param dist: approximating (variational) distribution, usually :class:`~qpytorch.distributions.MultitaskMultivariateQExponential`.
        :param u0: reference of linearization.
        :param eps: small nugget added to the diagonal of propagated covariance. (Default: 1e-6.)
        :param diag: if True, make the propagated covariance diagonal. (Default: True.)
        """
        mu, covar = dist.mean, dist._covar
        if u0 is None: u0 = mu
        batch_dim = covar.batch_dim
        linrz = self.linearize_PDE(u0, batch_jacobian=batch_dim>0)
        fun, jac = linrz
        mu_ = fun + (jac*(mu-u0).reshape(mu.shape[:batch_dim]+(1,)*(fun.ndim-batch_dim)+mu.shape[batch_dim:])).sum(dim=tuple(range(-(mu.ndim-batch_dim), 0)))
        if not kwargs.pop('interleaved', getattr(dist, '_interleaved', True)):
            jac = jac.permute(tuple(range(fun.ndim))+tuple(range(-1,fun.ndim-jac.ndim-1, -1)))
        jac = jac.reshape(*jac.shape[:fun.ndim],-1)
        if not diag:
            covar_ = jac.matmul(covar).matmul(jac.transpose(-1,-2)) + eps*torch.eye(jac.shape[-2], device=covar.device)
        else:
            var_ = (jac.matmul(covar)*jac).sum(-1) + eps
            covar_ = DiagLinearOperator(var_)
        return MultivariateQExponential(mu_, covar_, power=dist.power)