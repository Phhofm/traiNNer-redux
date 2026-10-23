"""No-reference TOPIQ (KonIQ CFANet) as a generator loss.

Experimental: maximizes the predicted MOS of the generated image.

Caution: no-reference IQA losses can be reward-hacked (the generator drifts
toward "what scores high" instead of what looks good). Always pair with
structural losses and judge visually.

Input: RGB in [0, 1], shape (B, C, H, W). The metric network is frozen and
run in eval mode; gradients flow into the generator through the input.
"""

import torch
from torch import Tensor, nn

from traiNNer.archs.topiq_arch import CFANet
from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class TopiqNRLoss(nn.Module):
    def __init__(self, loss_weight: float = 0.2) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.net = CFANet(model_name="cfanet_nr_koniq_res50", use_ref=False)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def forward(self, output: Tensor, _target: Tensor) -> Tensor:
        mos = self.net(output)  # ~[0, 100]
        return self.loss_weight * (1.0 - (mos / 100.0).clamp(0.0, 1.0)).mean()
