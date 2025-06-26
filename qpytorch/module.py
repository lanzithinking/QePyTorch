#!/usr/bin/env python3

from gpytorch.module import Module as GModule


class Module(GModule):
    def named_hyperparameters(self):
        from .variational._variational_distribution import _VariationalDistribution

        for module_prefix, module in self.named_modules():
            if not isinstance(module, _VariationalDistribution):
                for elem in module.named_parameters(prefix=module_prefix, recurse=False):
                    yield elem

    def named_variational_parameters(self):
        from .variational._variational_distribution import _VariationalDistribution

        for module_prefix, module in self.named_modules():
            if isinstance(module, _VariationalDistribution):
                for elem in module.named_parameters(prefix=module_prefix, recurse=False):
                    yield elem

    def update_added_loss_term(self, name, added_loss_term):
        from .mlls import AddedLossTerm

        if not isinstance(added_loss_term, AddedLossTerm):
            raise RuntimeError("added_loss_term must be a AddedLossTerm")
        if name not in self._added_loss_terms.keys():
            raise RuntimeError("added_loss_term {} not registered".format(name))
        self._added_loss_terms[name] = added_loss_term
