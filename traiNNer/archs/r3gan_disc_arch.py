"""R3GAN discriminator: norm-free residual conv blocks, R1-regularized.

Faithful to the R3GAN recipe: no normalization layers (neither batch-norm nor
spectral norm), stability comes from R1 regularization (``net_d_reg_every`` /
``net_d_reg_weight`` in the train options). Do NOT combine with spectral norm —
SN + R1 double-regularizes and collapses a discriminator this size (measured,
see ABLATIONS.md round 3/4).

Returns a logit map (B, 1, H/2, W/2), matching the ``DUnet`` interface.
Set ``use_spectral_norm: true`` for SN mode without R1 (round-4 S9 recipe).
"""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.utils.parametrizations import spectral_norm

from traiNNer.utils.registry import ARCH_REGISTRY


class _ResBlock(nn.Module):
    def __init__(self, ch: int, use_sn: bool) -> None:
        super().__init__()
        sn = spectral_norm if use_sn else (lambda c: c)
        self.c1 = sn(nn.Conv2d(ch, ch, 3, 1, 1))
        self.c2 = sn(nn.Conv2d(ch, ch, 3, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        h = F.leaky_relu(self.c1(F.leaky_relu(x, 0.2)), 0.2)
        return x + self.c2(h)


@ARCH_REGISTRY.register()
class R3GANDisc(nn.Module):
    def __init__(
        self,
        num_in_ch: int = 3,
        num_feat: int = 64,
        num_blocks: int = 6,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        sn = spectral_norm if use_spectral_norm else (lambda c: c)
        self.inp = sn(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.down = sn(nn.Conv2d(num_feat, num_feat, 4, 2, 1))
        self.body = nn.Sequential(
            *[_ResBlock(num_feat, use_spectral_norm) for _ in range(num_blocks)]
        )
        self.out = nn.Sequential(
            sn(nn.Conv2d(num_feat, num_feat, 3, 1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat, 1, 3, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.inp(x), 0.2)
        x = F.leaky_relu(self.down(x), 0.2)
        x = self.body(x)
        return self.out(x)
