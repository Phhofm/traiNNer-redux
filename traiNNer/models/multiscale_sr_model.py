"""
MultiScaleSRModel: Train Once, Export Multiple Scales.

This model extends SRModel to support multi-scale training where:
- A random scale (1x, 2x, 3x, 4x) is sampled per batch
- LR is generated on-the-fly by downsampling GT
- The network receives the scale parameter to select the right head
- All scales share the same body, providing multi-task regularization

Usage in config:
    model_type: MultiScaleSRModel
    multiscale:
      scales: [1, 2, 3, 4]
      degradation_mode: bicubic  # or 'lanczos', 'area'

Author: Philip Hofmann
License: MIT
"""

import random
from collections import OrderedDict

import torch
from torch import Tensor
from torch.nn import functional as F

from traiNNer.models.sr_model import SRModel
from traiNNer.utils import get_root_logger
from traiNNer.utils.color_util import pixelformat2rgb_pt, rgb2pixelformat_pt
from traiNNer.utils.redux_options import ReduxOptions


class MultiScaleSRModel(SRModel):
    """
    Multi-Scale SR Model for training networks with multiple upscaling heads.

    Key features:
    - Samples random scale per batch from configured scales
    - Generates LR on-the-fly by downsampling pristine GT
    - Passes scale parameter to network forward method
    - Logs sampled scale for monitoring
    """

    def __init__(self, opt: ReduxOptions) -> None:
        super().__init__(opt)

        logger = get_root_logger()

        # Get multiscale config
        multiscale_opt = getattr(opt, "multiscale", None) or {}
        self.supported_scales = multiscale_opt.get("scales", [1, 2, 3, 4])
        self.degradation_mode = multiscale_opt.get("degradation_mode", "bicubic")

        # For 1x (denoising), we can optionally add noise
        self.denoise_sigma = multiscale_opt.get("denoise_sigma", 0.0)

        logger.info(
            "MultiScaleSRModel initialized with scales=%s, degradation=%s",
            self.supported_scales,
            self.degradation_mode,
        )

        # Track scale distribution for logging
        self.scale_counts = dict.fromkeys(self.supported_scales, 0)

    def optimize_parameters(
        self, current_iter: int, current_accum_iter: int, apply_gradient: bool
    ) -> None:
        """
        Override optimize_parameters to implement multi-scale training.

        The key changes from SRModel:
        1. Sample a random scale each batch
        2. Generate LR on-the-fly by downsampling GT
        3. Pass scale to network forward
        """
        assert self.gt is not None
        assert self.scaler_g is not None

        # Sample random scale for this batch
        scale = random.choice(self.supported_scales)
        self.scale_counts[scale] += 1

        # Generate LR on-the-fly by downsampling pristine GT
        # GT stays pristine, LR is derived from it
        with torch.no_grad():
            if scale == 1:
                # For 1x (denoising), LR = GT + optional noise
                lq = self.gt.clone()
                if self.denoise_sigma > 0:
                    noise = torch.randn_like(lq) * self.denoise_sigma
                    lq = torch.clamp(lq + noise, 0, 1)
            else:
                # For upscaling, downsample GT to create LR
                lq = F.interpolate(
                    self.gt,
                    scale_factor=1.0 / scale,
                    mode=self.degradation_mode,
                    antialias=True if self.degradation_mode != "area" else False,
                )

        # Rest follows SRModel.optimize_parameters but with scale passed to network
        skip_d_update = False

        if self.net_d is not None:
            for p in self.net_d.parameters():
                p.requires_grad = False

        n_samples = self.gt.shape[0]
        self.loss_samples += n_samples
        loss_dict: dict[str, Tensor | float] = OrderedDict()

        # Process input
        lq = rgb2pixelformat_pt(lq, self.opt.input_pixel_format)

        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            if self.optimizer_g is not None:
                # KEY CHANGE: Pass scale to network
                output = self.net_g(lq, scale=scale)

                self.output = pixelformat2rgb_pt(
                    output, self.gt, self.opt.output_pixel_format
                )

                assert isinstance(self.output, Tensor)
                l_g_total = torch.tensor(0.0, device=lq.device)

                # Compute losses (same as SRModel)
                for label, loss in self.losses.items():
                    if label == "l_g_gan":
                        # Skip GAN for now in multi-scale (complex to handle)
                        continue

                    if hasattr(loss, "loss_module"):
                        l_g_loss = loss(self.output, self.gt, current_iter=current_iter)
                    else:
                        l_g_loss = loss(self.output, self.gt)

                    if isinstance(l_g_loss, dict):
                        for sublabel, loss_val in l_g_loss.items():
                            if loss_val > 0:
                                weighted = loss_val * abs(loss.loss_weight)
                                l_g_total += weighted / self.accum_iters
                                loss_dict[f"{label}_{sublabel}"] = weighted
                    else:
                        weighted = l_g_loss * abs(loss.loss_weight)
                        l_g_total += weighted / self.accum_iters
                        loss_dict[label] = weighted

                if not l_g_total.isfinite():
                    raise RuntimeError("Training failed: NaN/Inf found in loss.")

                loss_dict["l_g_total"] = l_g_total
                loss_dict["scale"] = float(scale)  # Log current scale

                self.scaler_g.scale(l_g_total).backward()

                if apply_gradient:
                    self.scaler_g.unscale_(self.optimizer_g)

                    if self.grad_clip:
                        clip_threshold = self.get_automation_clipping_threshold()
                        torch.nn.utils.clip_grad_norm_(
                            self.net_g.parameters(), clip_threshold
                        )

                    self.scaler_g.step(self.optimizer_g)
                    self.scaler_g.update()
                    self.optimizer_g.zero_grad()
            else:
                with torch.inference_mode():
                    self.output = self.net_g(lq, scale=scale)

        # Log losses
        for key, value in loss_dict.items():
            val = (
                value
                if isinstance(value, float)
                else value.to(dtype=torch.float32).detach()
            )
            self.log_dict[key] = self.log_dict.get(key, 0) + val * n_samples

        self.log_dict = self.reduce_loss_dict(self.log_dict)

        # Update EMA
        if self.net_g_ema is not None and apply_gradient:
            if not (self.use_amp and self.optimizers_skipped[0]):
                self.net_g_ema.update()

    def test(self) -> None:
        """Override test to use max scale for validation."""
        max_scale = max(self.supported_scales)

        if self.net_g_ema is not None:
            self.net_g_ema.eval()
            with torch.inference_mode():
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    self.output = self.net_g_ema(self.lq, scale=max_scale)
        else:
            self.net_g.eval()
            with torch.inference_mode():
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    self.output = self.net_g(self.lq, scale=max_scale)
            self.net_g.train()
