# ParagonSR3 Architecture

ParagonSR3 is a state-of-the-art Single Image Super-Resolution (SISR) and Video Super-Resolution architecture built on the philosophy of **"Surgical Efficiency via Re-parameterization."**

The primary goal of ParagonSR3 is to surpass the validation metrics of heavy transformer models like HAT-L, while remaining drastically faster, highly resource-efficient, and easily deployable via TensorRT (TRT) in FP16 precision.

## Core Design Philosophy

Older architectures often rely on applying expensive Self-Attention everywhere (e.g., standard HAT) or scaling up pure Convolutional networks simply by adding more parameters. ParagonSR3 takes a surgical approach:
1. **Compute Heavily During Training, Run Lightly During Inference**: Leveraging structural re-parameterization.
2. **Attention is a Beacon, not a Floodlight**: Using attention only where it matters (guided by feature variance) rather than on every single pixel.
3. **Multi-Task Regularization**: Training all upscaling scales simultaneously to force the network to learn generalized, high-quality feature representations.

## Key Architectural Components

### 1. GatedRepConv Engine & Re-parameterization
The core engine of ParagonSR3 is the **GatedRepConv** block. During training, this block consists of a 3x3 convolution, a 1x1 convolution, and an identity connection. This multi-branch design allows the network to learn rich, complex feature representations.
Before deployment, a `fuse_model()` function is called, which mathematically collapses the 1x1 and identity branches into the single 3x3 convolution. At inference time (e.g., in TensorRT), it runs as an incredibly fast, pure 3x3 Convolution.

Within this block, the architecture also utilizes **SCA (Simple Channel Attention)**. This extremely lightweight mechanism is the "secret sauce" that allows ConvNets to compete with or beat the PSNR metrics of heavy Transformers (similar to NAFNet).

### 2. Multi-Head Pipeline & Hierarchical Feature Fusion
ParagonSR3 utilizes a shared body with separate upsampling heads for `1x`, `2x`, `3x`, and `4x` scales. By training all scales simultaneously, the main network body learns highly robust features. The `1x` (denoising) head acts as a powerful regularizer that significantly improves the feature quality for the larger upscaling heads, leveraging **SiLU activations** for an efficient, zero-cost metric bump.

Before the final upsampling head, ParagonSR3 employs **Cross-Scale Feature Fusion (Hierarchical Tapping)**. It taps the intermediate output features from every residual group, concatenating the high-frequency shallow features with the global-context-aware deep features. This hierarchical fusion is the architectural key to maximizing PSNR/SSIM on geometric datasets like Urban100, providing the upsampler with a perfectly balanced feature hierarchy. At export time, you simply slice off the heads you don't need, resulting in a perfectly optimized model for your target scale.

### 3. IET Normalization (iLN) & Variance-Based Routing
ParagonSR3 integrates Image Restoration Layer Normalization (iLN). Standard LayerNorm can often cause NaN (numerical instability) issues when converted to FP16 TensorRT. iLN safely computes statistics in FP32 while keeping the tensor operations in the native dtype.

Crucially, iLN calculates the **true statistical variance (`torch.var`)** of the local features. This variance acts as a gating signal: flat, uniform areas bypass expensive attention mechanisms, while high-frequency, complex textures are routed into the Attention Beacons.

### 4. Sparse Attention Beacons
Instead of deploying heavy Self-Attention at every layer, ParagonSR3 uses attention "Beacons":
- **WindowAttention with Depthwise Halo Exchange**: Standard SDPA localized window attention, augmented with a 3x3 depthwise convolution before the projection. This allows features to "see" one pixel outside their 16x16 window boundary, eliminating grid artifacts and providing the exact same PSNR benefits as HAT's overlapping attention, but without the massive TensorRT memory overhead of complex window unfolding.
- **Dynamic TokenDictionaryCA**: A highly compressed global context beacon. Instead of comparing every pixel to every pixel, features query a small set (e.g., 64) of global tokens. Inspired by IET (Individualized Exploratory Transformer), ParagonSR3 uses a tiny MLP and global average pooling to generate these tokens dynamically on the fly, tailoring the global context specifically to the unique content of the image being upscaled.

By placing these beacons sparsely (e.g., every 4th or 12th block), the network gains full global context and structural coherence with a fraction of the FLOPs required by HAT.

## Model Variants

To keep deployment simple and foolproof, ParagonSR3 provides exactly two variants, both defaulting to the multi-scale `[1, 2, 3, 4]` training setup.

### Virtuoso (Photo Variant)
* **Goal**: Maximum PSNR/SSIM, aiming to cleanly beat HAT-L.
* **Structure**: A wider (180 feature channels) and deeper (10 groups x 8 blocks) network.
* **Attention Layout**: Aggressively interleaves Window and Token attention beacons to ensure global consistency across high-resolution images. Features deep refinement in the final layers.

### Aegis (Video Variant)
* **Goal**: Blazing fast video processing with temporal stability and robustness against heavy compression artifacts.
* **Structure**: A lighter (64 feature channels), shallower (4 groups x 6 blocks) network. Uses Recurrent state concatenation (passing previous frame features).
* **Attention Layout**: Almost entirely pure Convolutional for maximum TensorRT frame-rates. However, it specifically injects a single sparse **TokenDictionaryCA beacon** into the middle of each group. This token beacon gives the video model the absolute global context required to detect and flatten large macroblocking artifacts from heavy video compression—something pure CNNs struggle with—at virtually zero latency cost.
