"""Domain-Specific Batch Normalization (DSBN, Chang et al., CVPR 2019).

In a mixed real+synthetic batch a shared BatchNorm2d normalizes the real
samples with statistics pooled over both domains. DSBN gives each domain its
own BN branch (running statistics and affine parameters), routes real samples
through branch 0 and synthetic samples through branch 1 during training, and
uses the real branch at test time. Convolution weights stay shared.

    n = convert_to_dsbn(model)          # swap every BatchNorm2d in place
    set_dsbn_domain(model, domain)      # (B,) tensor of 0/1, before each forward
    set_dsbn_domain(model, None)        # evaluation: real branch for the whole batch

Requires a BatchNorm model (resnet_unet; SegFormer uses LayerNorm) and a
single GPU, since the per-module domain attribute does not survive
nn.DataParallel's replica scatter.
"""

import torch
from torch import nn


class DomainSpecificBatchNorm2d(nn.Module):
    """BatchNorm2d with one independent branch per domain."""

    def __init__(self, num_features, num_domains=2, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(num_features, eps=eps, momentum=momentum,
                           affine=affine, track_running_stats=track_running_stats)
            for _ in range(num_domains)
        ])
        self.domain = None

    def forward(self, x):
        if self.domain is None:
            return self.bns[0](x)

        domain = torch.as_tensor(self.domain, device=x.device)
        uniq = torch.unique(domain)
        if uniq.numel() == 1:
            return self.bns[int(uniq.item())](x)

        out = torch.empty_like(x)
        for d in range(self.num_domains):
            idx = (domain == d).nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                out[idx] = self.bns[d](x[idx])
        return out


def convert_to_dsbn(module, num_domains=2):
    """Recursively replace every nn.BatchNorm2d with a DomainSpecificBatchNorm2d
    of the same configuration. Returns the number of layers converted. The new
    branches initialize without drawing random numbers, so calling this after
    weight init leaves the rest of the model identical to a non-DSBN run."""
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, DomainSpecificBatchNorm2d(
                child.num_features, num_domains=num_domains,
                eps=child.eps, momentum=child.momentum,
                affine=child.affine, track_running_stats=child.track_running_stats))
            count += 1
        else:
            count += convert_to_dsbn(child, num_domains=num_domains)
    return count


def set_dsbn_domain(model, domain):
    """Set the domain selector on every DSBN layer: a (B,) tensor of 0/1, or
    None to route the whole batch through the real branch."""
    for m in model.modules():
        if isinstance(m, DomainSpecificBatchNorm2d):
            m.domain = domain
