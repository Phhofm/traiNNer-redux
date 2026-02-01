#!/usr/bin/env python3
"""
ParagonSR3 Multi-Scale ONNX Converter
=====================================

Exports a single trained ParagonSR3 checkpoint to multiple scale-specific ONNX models.

Features:
- Exports 1x, 2x, 3x, 4x from ONE checkpoint
- Auto-patches AdaptiveAvgPool2d -> ReduceMean (TensorRT Friendly)
- Exports Dynamic FP32 ONNX (Best for trtexec --fp16)
- Validates PSNR match between PyTorch and ONNX

Usage:
    python convert_onnx_release.py \\
        --checkpoint "models/paragonsr3_photo.safetensors" \\
        --arch paragonsr3_photo_multiscale \\
        --scales 1,2,3,4 \\
        --output "release_output" \\
        --device cuda

    # Then build TRT engines for each scale:
    trtexec --onnx=release_output/paragonsr3_photo_1x_fp32.onnx \\
            --saveEngine=paragonsr3_photo_1x_fp16.trt --fp16

Author: Philip Hofmann
License: MIT
"""

import argparse
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import shape_inference
from PIL import Image
from torch import nn

# ---------------------------------------------------------------------
# SETUP: Import Architecture
# ---------------------------------------------------------------------
try:
    repo_root = Path(__file__).parents[2]
    sys.path.insert(0, str(repo_root))
    from traiNNer.archs import paragonsr3_arch
    from traiNNer.archs.paragonsr3_arch import ParagonSR3, ParagonSR3ExportWrapper
    from traiNNer.utils.registry import ARCH_REGISTRY

    ARCH_MAP = ARCH_REGISTRY
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import architecture: {e}")
    print("Ensure traiNNer is installed or run from the repo root.")
    sys.exit(1)

warnings.filterwarnings("ignore")


# =============================================================================
# HELPER: TensorRT Compatibility Patcher
# =============================================================================


class TensorRTGlobalAvgPool(nn.Module):
    """Replaces nn.AdaptiveAvgPool2d(1) with torch.mean() for TRT."""

    def forward(self, x):
        return x.mean(dim=[-1, -2], keepdim=True)


def patch_model_for_tensorrt(model: nn.Module) -> nn.Module:
    """Recursively replace AdaptiveAvgPool2d(1) with TensorRTGlobalAvgPool."""
    replaced_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.AdaptiveAvgPool2d):
            out = module.output_size
            is_one = (out == 1) if isinstance(out, int) else (out == (1, 1))

            if is_one:
                if "." in name:
                    parent_name, child_name = name.rsplit(".", 1)
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                    child_name = name
                setattr(parent, child_name, TensorRTGlobalAvgPool())
                replaced_count += 1

    print(f"      Patched {replaced_count} AdaptiveAvgPool layers for TRT.")
    return model


# =============================================================================
# UTILITIES
# =============================================================================


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, border: int = 0) -> float:
    if border > 0:
        img1 = img1[border:-border, border:-border, :]
        img2 = img2[border:-border, border:-border, :]
    mse = np.mean((img1 - img2) ** 2)
    if mse <= 1e-10:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def preprocess_image(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(img_array, axis=0)  # -> BCHW


def postprocess_output(output: np.ndarray) -> np.ndarray:
    output = output.squeeze(0)
    output = np.transpose(output, (1, 2, 0))  # CHW -> HWC
    output = np.clip(output, 0, 1)
    return (output * 255.0).round().astype(np.uint8)


# =============================================================================
# CONVERTER CLASS
# =============================================================================


class ParagonSR3Converter:
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
        )
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Parse scales
        if args.scales:
            self.scales = [int(s.strip()) for s in args.scales.split(",")]
        else:
            self.scales = [1, 2, 3, 4]

        print(f"Will export scales: {self.scales}")

    def load_model(self) -> ParagonSR3:
        print(f"\n[1/3] Loading architecture: {self.args.arch}")
        arch_fn = ARCH_MAP.get(self.args.arch)

        if arch_fn is None:
            raise ValueError(f"Architecture '{self.args.arch}' not found in registry.")

        model = arch_fn()

        if self.args.checkpoint and not self.args.no_weights:
            print(f"      Loading weights: {self.args.checkpoint}")
            if str(self.args.checkpoint).endswith(".safetensors"):
                from safetensors.torch import load_file

                state_dict = load_file(self.args.checkpoint)
            else:
                state_dict = torch.load(self.args.checkpoint, map_location="cpu")
                if "params_ema" in state_dict:
                    state_dict = state_dict["params_ema"]
                elif "params" in state_dict:
                    state_dict = state_dict["params"]

            # Strip 'module.' prefix if present (DDP)
            new_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_dict[k] = v

            missing, unexpected = model.load_state_dict(new_dict, strict=False)
            if missing:
                print(f"      [WARNING] Missing keys: {len(missing)}")
            if unexpected:
                print(f"      [WARNING] Unexpected keys: {len(unexpected)}")
            if not missing and not unexpected:
                print("      Weights loaded perfectly.")
        else:
            print("      Skipping weight loading (--no-weights or no checkpoint).")

        model.to(self.device).eval()

        # Fuse RepConv branches
        if hasattr(model, "fuse_model"):
            print("      Fusing RepConv branches for deployment...")
            model.fuse_model()

        # Patch for TRT
        model = patch_model_for_tensorrt(model)
        return model

    def export_all_scales(self, model: ParagonSR3) -> list[Path]:
        print(f"\n[2/3] Exporting ONNX for scales: {self.scales}")
        output_paths = []

        for scale in self.scales:
            output_path = self.output_dir / f"{self.args.arch}_{scale}x_fp32.onnx"
            print(f"\n      Exporting {scale}x to {output_path.name}...")

            # Wrap model with locked scale
            wrapped = ParagonSR3ExportWrapper(model, scale=scale)

            # Create dummy input
            dummy_input = torch.randn(1, 3, 64, 64, device=self.device)

            # Export
            with torch.no_grad():
                torch.onnx.export(
                    wrapped,
                    dummy_input,
                    output_path,
                    export_params=True,
                    opset_version=self.args.opset,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch", 2: "height", 3: "width"},
                        "output": {0: "batch", 2: "height", 3: "width"},
                    },
                )

            # Optimize
            try:
                onnx_model = onnx.load(str(output_path))
                onnx_model = shape_inference.infer_shapes(onnx_model)
                onnx.save(onnx_model, str(output_path))
            except Exception as e:
                print(f"      Warning: ONNX optimization failed: {e}")

            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"      Success: {output_path.name} ({size_mb:.2f} MB)")
            output_paths.append(output_path)

        return output_paths

    def validate(self, model: ParagonSR3, output_paths: list[Path]) -> None:
        if not self.args.val_dir:
            return

        print("\n[3/3] Validating ONNX accuracy...")
        val_dir = Path(self.args.val_dir)
        val_images = sorted(
            [
                p
                for p in val_dir.glob("*")
                if p.suffix.lower() in [".png", ".jpg", ".webp"]
            ]
        )[: self.args.val_count]

        if not val_images:
            print("      No images found for validation.")
            return

        print(f"      Testing on {len(val_images)} images...")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.args.device == "cuda"
            else ["CPUExecutionProvider"]
        )

        for onnx_path in output_paths:
            # Extract scale from filename
            scale = int(onnx_path.stem.split("_")[-2].replace("x", ""))

            try:
                sess = ort.InferenceSession(str(onnx_path), providers=providers)
            except Exception as e:
                print(f"      Error loading {onnx_path.name}: {e}")
                continue

            psnrs = []
            for img_path in val_images[:5]:  # Quick validation
                inp = preprocess_image(img_path)
                _, _, h, w = inp.shape
                if h > 256 or w > 256:
                    inp = inp[:, :, :256, :256]

                # PyTorch
                with torch.no_grad():
                    pt_in = torch.from_numpy(inp).to(self.device)
                    pt_out = model(pt_in, scale=scale).cpu().numpy()
                    pt_img = postprocess_output(pt_out)

                # ONNX
                onnx_out = sess.run(None, {"input": inp})[0]
                onnx_img = postprocess_output(onnx_out)

                psnrs.append(calculate_psnr(pt_img, onnx_img, border=4))

            avg = sum(psnrs) / len(psnrs)
            status = "✅ PASS" if avg > 50 else "⚠️  FAIL"
            print(f"      {onnx_path.name}: {avg:.1f} dB {status}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ParagonSR3 Multi-Scale ONNX Converter"
    )
    parser.add_argument("--checkpoint", help="Path to .pth or .safetensors")
    parser.add_argument(
        "--arch",
        default="paragonsr3_photo_multiscale",
        help="Model variant (e.g., paragonsr3_photo_multiscale)",
    )
    parser.add_argument(
        "--scales",
        default="1,2,3,4",
        help="Comma-separated list of scales to export (default: 1,2,3,4)",
    )
    parser.add_argument("--output", default="release_onnx", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument("--val_dir", help="Folder of images to validate ONNX accuracy")
    parser.add_argument("--val_count", type=int, default=10, help="Images to test")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Skip loading weights (for testing export only)",
    )

    args = parser.parse_args()

    converter = ParagonSR3Converter(args)
    model = converter.load_model()
    output_paths = converter.export_all_scales(model)

    if args.val_dir:
        converter.validate(model, output_paths)

    print("\n" + "=" * 60)
    print("DONE. Models exported successfully!")
    print(f"Output directory: {converter.output_dir}")
    for p in output_paths:
        print(f"  - {p.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
