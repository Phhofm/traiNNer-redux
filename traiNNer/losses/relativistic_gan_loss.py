"""Correct the GAN objective to the R3GAN recipe: relativistic loss + R2 penalty.

Reference: /home/phips/Documents/R3GAN/R3GAN-main (brownvc/r3gan).

Trainer.py AccumulateDiscriminatorGradients:
    D loss    = softplus(-(RealLogits - FakeLogits)).mean()          [relativistic]
              + (Gamma/2) * (R1Penalty + R2Penalty)
    G loss    = softplus(-(FakeLogits - RealLogits)).mean()          [relativistic]
    R1Penalty = ZeroCenteredGradientPenalty(RealSamples, RealLogits)   (grad of D(real).sum() wrt real)
    R2Penalty = ZeroCenteredGradientPenalty(FakeSamples, FakeLogits)   (same on fake)
    Gamma is scheduled (e.g. FFHQ-64: 0.05 -> 0.005 over 20M images).

sr_model.py wiring notes:
    - fake_d_pred is computed from self.output.detach() — fine for R2 (no
      generator grads needed), the penalty flows into D only.
    - real_d_pred already keeps grad (used for l_d_real) — R1 can reuse it.
    - The D step backward is split: hinge/vanilla losses use scaler_d on the
      summed loss; the reg terms must be backwarded separately (their grads
      must NOT hit the G), then added to the D loss for bookkeeping.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class RelativisticGanLoss(nn.Module):
    """Relativistic GAN loss (R3GAN objective).

    D step: softplus(-(RealLogits - FakeLogits))
    G step: softplus(-(FakeLogits - RealLogits))
    """

    def __init__(self, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def _relativistic(self, real: Tensor, fake: Tensor) -> Tensor:
        return fake - real

    def forward(self, input: Tensor, target: bool | Tensor) -> Tensor:
        raise NotImplementedError(
            "RelativisticGanLoss needs the other side's logits. Use gan_type="
            "'relativistic' in the config (wired in sr_model) instead of this "
            "standalone loss."
        )


def r2_penalty(fake_pred: Tensor, fake_img: Tensor) -> Tensor:
    """Zero-centered gradient penalty on fake samples (R2). R3GAN applies it
    symmetrically with R1 (Trainer.py AccumulateDiscriminatorGradients)."""
    grad_fake = torch.autograd.grad(
        outputs=fake_pred.sum(), inputs=fake_img, create_graph=True
    )[0]
    return grad_fake.pow(2).reshape(grad_fake.shape[0], -1).sum(1).mean()
