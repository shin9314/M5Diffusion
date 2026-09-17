# Local AI upscaling

The uploaded image is processed by **Real-ESRGAN realesr-general-x4v3**, a pretrained neural super-resolution model. This is separate from diffusion image generation and requires no prompt or diffusion checkpoint. All image processing stays on this Mac.

- **4×:** native neural output.
- **2×:** the same 4× neural output is resized once with Lanczos to exactly twice the input dimensions. It is still neural enhancement, not plain interpolation of the input.
- Transparent images retain their alpha channel, resized independently with Lanczos. RGB pixels are enhanced by the neural model. Output is suitable for PNG.
- Maximum input: **4,000,000 pixels**. Maximum final output: **16,000,000 pixels**, with each side no greater than **4096 pixels**. For example, square 4× input may be at most 1000×1000; square 2× input at most 2000×2000.
- EXIF rotation is honored. Model operations use RGB; grayscale/palette images are converted to RGB/RGBA. Working image precision is 8 bits per channel.

## Model provenance and license

Official upstream repository: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).

Official release download:
[realesr-general-x4v3.pth, v0.2.5.0](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth).

Local path: `models/upscale/realesr-general-x4v3.pth`. Downloaded size: approximately **4.7 MiB**, not 16 MiB. SHA-256, calculated from the downloaded official artifact and pinned in the loader:

```text
8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292
```

The loader verifies that hash before using `torch.load(..., map_location='cpu', weights_only=True)` and loads every expected parameter strictly. It does not accept arbitrary user checkpoint files. No new global Python dependency is installed.

Architecture was verified against the upstream [SRVGGNetCompact source](https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/srvgg_arch.py) and [official inference model selection](https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py): 3 RGB input/output channels, 64 features, 32 internal convolution stages, PReLU activations, PixelShuffle 4× and nearest-neighbor input residual. Including first/last convolutions, there are **34 convolution layers**. The implementation retains upstream parameter names and loads the published weights without remapping or dropping keys.

Upstream license: **BSD 3-Clause, Copyright (c) 2021 Xintao Wang**. The full required license and disclaimer are retained in [third-party/Real-ESRGAN-LICENSE.txt](third-party/Real-ESRGAN-LICENSE.txt). The single general model is used directly, corresponding to denoise strength 1 in upstream selection; the separate weak-denoise model and interpolation between model weights are not implemented.

## Memory and tile boundaries

Inference uses 192×192 input tiles, with a **34-pixel input halo** on all available sides. Thirty-four successive 3×3 convolutions have a 69×69 receptive field (radius 34). The complete halo is inferred, but only each tile's disjoint central output region is kept. This prevents artificial tile boundaries from affecting retained pixels. At true image boundaries the model uses its normal zero padding.

Only one tile's GPU activations are live at a time. The output is assembled on CPU. For 2× output, a full intermediate 4× RGB image is assembled before one final resize, avoiding independent tile-resize seams. This intermediate can contain up to 64,000,000 pixels for the largest allowed 2× input; it is an 8-bit CPU RGB buffer, not a full-resolution GPU activation tensor.

The model is loaded per job, using FP16 on Apple Metal via Torch MPS and FP32 if MPS is unavailable. References are discarded, GPU work synchronized, and the MPS allocation cache released after the job. No model stays resident in a global cache. UI work serializes GPU jobs and releases the diffusion engine before upscaling.

## API and tests

```python
validate_image(image, scale)  # Lightweight; does not import Torch or touch GPU.
result, metadata = upscale_image(image, scale, progress=callback)
# callback(fraction_between_0_and_1, human_readable_message)
```

`tests/test_upscale.py`: 10 CPU tests passed; the exclusive actual-model MPS test also passed (**11 total**). Checks cover exact official checkpoint structure, input/output limits, no Torch import during lightweight validation, CPU tiled/full equivalence with a reduced-width architecture, 2×/4× dimensions, alpha preservation, progress monotonicity, and actual GPU neural output differing from ordinary Lanczos resizing. A numerical tile test permits at most one 8-bit level of rounding difference.

Real-ESRGAN can reconstruct plausible detail; it cannot guarantee recovery of the original lost detail or perfectly reproduce text and faces.
