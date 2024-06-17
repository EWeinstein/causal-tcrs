"""
Modified distributions used with CAIRE.

 - MissingDataOneHotCategorical: modified one-hot categorical distribution, which handles missing data.
 - Normal: modified Normal distribution, for which gpu sampling does not involve an unnecessary cpu sync.

"""


import torch
from pyro.distributions import OneHotCategorical
from pyro.distributions.torch_distribution import TorchDistributionMixin


class MissingDataOneHotCategorical(OneHotCategorical):
    """One hot categorical distribution which allows for missing data (represented as all zeros)."""

    def log_prob(self, value):
        return (self._categorical.logits * value).sum(dim=-1)


def _standard_normal(shape, dtype, device):
    """Sample from a standard normal distribution.
    Based on torch.distributions.utils._standard_normal, but with std=torch.ones implicit to avoid cpu syncs
    (see https://pytorch.org/docs/stable/generated/torch.normal.html)
    """
    if torch._C._get_tracing_state():
        # [JIT WORKAROUND] lack of support for .normal_()
        return torch.normal(mean=torch.zeros(shape, dtype=dtype, device=device))
    return torch.empty(shape, dtype=dtype, device=device).normal_()


class Normal(torch.distributions.Normal, TorchDistributionMixin):

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        samp = self.loc + eps * self.scale
        return samp