# Public launch facts — v0.1.0-alpha

Use this page as the factual source for external announcements. No external posts have been made as part of preparing these materials.

## One-line description

Fast local image generation for Apple Silicon, powered by MLX and Metal.

## Short description

M5Diffusion is an independent, experimental alpha project that runs Stable Diffusion 1.5 locally on Apple Silicon using an MLX-native inference engine and a browser UI. It supports local checkpoint import, limited standard SD1.x LoRAs, and separate Real-ESRGAN image upscaling. The source is available under MIT, with separate upstream and model license terms.

## Links and tested hardware

- Repository: https://github.com/shin9314/M5Diffusion
- Download/release: https://github.com/shin9314/M5Diffusion/releases/tag/v0.1.0-alpha
- Built and tested on: MacBook Air M5, 24 GB unified memory.
- Recorded benchmark environment: macOS 26.6, MLX 0.32.2, Python 3.10.
- Requirements: Apple Silicon, macOS 26 or later, at least 16 GB unified memory. Only the M5 Air with 24 GB has been validated.

## Benchmark summary

SD1.5, 512×512, 20 DPM++ 2M Karras steps, CFG 7, seed 12345, identical checkpoint and prompt, no LoRA or quantization:

| Controlled MLX A/B | Native convolution | Selective optimized convolution |
| --- | ---: | ---: |
| Pooled warm median diffusion | 7.486 s | 6.025 s |
| Pooled warm median generation total | 8.309 s | 6.822 s |

The median of three paired session diffusion ratios is **1.23×**. Three sessions used A/B, B/A, A/B order, with 11 images per variant per session and the first excluded: 30 warm samples per variant. This ratio is not the quotient of the pooled medians.

Model loading and PNG encoding are outside the reported generation total. These are local optimization measurements, not a controlled comparison with PyTorch MPS or Draw Things, and not a universal speed guarantee. Thermal state and background load affect results. Do not mix historical development timings with this controlled result. See [BENCHMARKS.md](../BENCHMARKS.md) for boundaries, quality checks, raw-record links and limitations.

## Technical highlights and Show HN facts

- Independent project; built and tested on a MacBook Air M5 with 24 GB unified memory.
- MLX-native SD1.5 engine with local inference on Apple Silicon through Metal.
- Selective eligible 3×3 convolutions use im2col plus MLX matmul; remaining convolutions keep native MLX execution.
- Selectively padded fused self-attention and compiled diffusion steps; FP16 UNet/CLIP, FP32 latents, scheduler and VAE.
- Separate local Real-ESRGAN upscaling of uploaded images, with 2×/4× output and before/after preview. This path uses Torch MPS.
- Source available; original project code is MIT licensed. Upstream notices are retained, and checkpoints/LoRAs/model weights keep their own terms.
- Experimental alpha; no claim to be the fastest engine or to beat unmeasured external applications.

## Limitations and distribution facts

- SD1.5 only. SDXL, FLUX and ControlNet are unsupported; standard SD1.x LoRA support is [limited](lora.md).
- Other Apple Silicon generations and memory configurations are unverified; Intel Macs are unsupported.
- Local single-user service; not a hosted or multi-user deployment.
- The DMG includes a private runtime and dependencies, but not SD1.5 weights. Users must obtain compatible weights under their applicable terms. Upscaler weight preparation is explicit and separate.
- The app is not Developer ID signed or notarized. See [installation guidance](../INSTALL.md).
- The public release is a pre-release. This launch preparation does not replace its tag or assets or change the inference implementation.

## Short demo recording recipe

Target a 10–20 second **warm-generation** clip. Timing is a recording target, not a promised first-launch time.

1. Complete installation, model import/conversion and one warm-up generation before recording. Set 512×512, 20 steps, CFG 7 and seed 12345. Keep other GPU jobs idle.
2. Capture only the Web UI content area. Hide browser chrome, bookmarks, usernames, local paths and unrelated windows. Use a prompt and output safe to publish.
3. Start on the prepared UI, enter the prompt, click Generate, and show the resulting image in one continuous real-time take. Add a visible label: “Warm generation; model already loaded. MacBook Air M5 / 24 GB. SD1.5 / 512×512 / 20 steps / CFG 7.”
4. Do not accelerate or omit the generation wait. If it takes longer than the target, retain the actual duration. App opening may be a separate labeled introduction; do not imply that cold startup, conversion or first model loading fits the warm-generation timing.
5. Export a small GIF/WebP for README use or host a video outside Git. Check the exported frames for private information and confirm that elapsed time matches the recording. Keep the large original recording out of the repository.

Use the actual captured run for any on-screen elapsed time; benchmark medians are not the duration of an individual demo. A first-launch demonstration should explicitly retain or disclose its setup/loading time and has no 10–20 second guarantee.
