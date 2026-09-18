"""A differentiable 12L:12D logistic reference; time is in physical hours.

This is a proposed mathematical component, not a trained forecasting model.
Light starts at 0 h. One signed contrast is shared by plants of each genotype.
"""
import torch
from torch import nn


def light_state(hours):
    """Right-continuous schedule: [0,12) light, [12,24) dark, repeated."""
    return (torch.remainder(hours,24.)<12.).to(hours.dtype)


def cumulative_light(hours):
    """Total illuminated hours between 0 and a nonnegative elapsed time."""
    return 12.*torch.floor(hours/24.)+torch.remainder(hours,24.).clamp(max=12.)


class LightDarkLogistic(nn.Module):
    def __init__(self,n_genotypes=4):
        super().__init__()
        self.beta=nn.Parameter(torch.zeros(n_genotypes))

    def contrast(self,g_idx):
        return torch.tanh(self.beta[g_idx])

    def phase_rates(self,mean_rate,g_idx):
        a=self.contrast(g_idx)
        return mean_rate*(1+a),mean_rate*(1-a)

    def rate(self,mean_rate,g_idx,hours):
        q=2.*light_state(hours)-1.
        return mean_rate[:,None]*(1.+self.contrast(g_idx)[:,None]*q[None,:])

    def rhs(self,height,mean_rate,capacity,g_idx,hours):
        """Height [plant,time]; rate/capacity [plant]; hours [time].

        Capacity is assumed strictly positive. Height and capacity use the same
        physical or normalized length units; mean_rate is always per hour.
        """
        return self.rate(mean_rate,g_idx,hours)*height*(1.-height/capacity[:,None])

    def effective_time(self,g_idx,hours):
        contrast_integral=2.*cumulative_light(hours)-hours
        return hours[None,:]+self.contrast(g_idx)[:,None]*contrast_integral[None,:]

    def solution(self,initial_height,mean_rate,capacity,g_idx,hours):
        """Exact positive solution for 0 < initial_height < capacity.

        This solution verifies the reference law. PhytoODE's neural prediction
        would only be softly constrained by rhs, so it would not inherit all
        exact-solution guarantees automatically.
        """
        if not bool(torch.all((initial_height>0)&(initial_height<capacity)&(mean_rate>0))):
            raise ValueError('Require 0 < initial_height < capacity and mean_rate > 0')
        eta=torch.log(initial_height/(capacity-initial_height))[:,None]
        eta=eta+mean_rate[:,None]*self.effective_time(g_idx,hours)
        return capacity[:,None]*torch.sigmoid(eta)

    @staticmethod
    def collocation_mask(hours):
        """Classical derivative residuals must exclude light-switching times."""
        return ~torch.isclose(torch.remainder(hours,12.),torch.zeros_like(hours),atol=1e-9,rtol=0.)
