# M5Diffusion

Fast local image generation for Apple Silicon, powered by MLX and Metal.

Experimental alpha software optimized and currently benchmarked primarily on a MacBook Air M5 with 24GB Unified Memory.

**v0.1.0-alpha** runs SD1.5 locally on a Mac, with a browser UI and a separate uploaded-image upscaler. The release freezes the measured inference implementation; it is not a claim of universal performance across Apple Silicon.

## Features

- Local SD1.5 text-to-image with prompt, negative prompt, seed, steps, CFG and image dimensions.
- MLX FP16 UNet/CLIP, selective im2col + native MLX matmul, selectively padded fused self-attention, and compiled diffusion steps.
- DPM++ 2M Karras sampling; FP32 latents, scheduler and VAE.
- Local SD1.5 safetensors import and **limited standard LoRA support**. See [supported variants](docs/lora.md).
- Real-ESRGAN uploaded-image enhancement, 2×/4× output, transparency preservation, before/after preview and PNG download.
- Local processing; weights and generated images are not included in the source repository.

## Requirements and installation

Apple Silicon, macOS 26 or later, and at least 16 GB unified memory. **Only an M5 MacBook Air with 24 GB has been validated.** Other supported-architecture Macs remain unverified. Allow storage for the app, imported weights, conversion and outputs.

Download the latest DMG from this repository’s Releases page. Open `M5Diffusion-v0.1.0-alpha.dmg`, drag **M5Diffusion.app** to Applications, then open it. The app includes a private Python runtime and dependencies: no Homebrew, system-Python modification or automatic sudo. The first launch prepares a writable application-data directory and opens the local Web UI. Follow its model setup to import a compatible SD1.5 `.safetensors` file. Upscaling has a separate explicit official-model download.

This alpha is not Developer ID signed or notarized. See [INSTALL.md](INSTALL.md) for trusted-download/Gatekeeper guidance, startup checks, model setup, quitting and troubleshooting.

## Web UI

The browser opens at [127.0.0.1:7861](http://127.0.0.1:7861). Generate images using the labeled controls, then save the PNG. The upscaling mode accepts an existing image independently of text-to-image generation. Processing is serialized to avoid competing GPU jobs. Quit the app to stop its server. See [UI details](docs/UI.md) and [upscaler provenance](docs/upscale.md).

## Measured performance and quality

Controlled optimization comparison on the tested M5 Air: same SD1.5 checkpoint, 512×512, 20 steps, CFG 7, seed 12345, prompt and sampler. Three counterbalanced A/B sessions, 11 images per arm/session, first image excluded; at least 60 seconds between runs.

| Pooled warm median, 30 images per arm | Native MLX convolution baseline | Accepted selective convolution implementation |
|---|---:|---:|
| Generation total | 8.309 s | **6.822 s** |
| Diffusion | 7.486 s | **6.025 s** |
| Seconds/step | 0.374 | **0.301** |
| VAE | 0.795 s | **0.745 s** |

Median of the three paired diffusion-median ratios: **1.230×**. Accepted 10-image session medians: **6.295 / 5.072 / 6.204 seconds**. This is an optimization comparison within MLX, **not** a controlled claim against PyTorch MPS or Draw Things. Historical MPS/MLX measurements are separate archival observations.

Fixed-request quality checks against the preceding MLX implementation recorded minimum SSIM **0.99827862** and PSNR **48.08 dB**. Recorded swap growth was **0 MiB**. Floating-point rounding still differs; these checks are not a guarantee for every prompt or checkpoint. Performance depends on thermal state, background system load, model configuration, and hardware. Current benchmark results are experimental and should not be interpreted as universal performance guarantees. Full definitions, provenance and limitations: [BENCHMARKS.md](BENCHMARKS.md).

## Example output

![SD1.5 generated mountain lake landscape](docs/images/sd15-example.png)

Example local SD1.5 output; not a quality guarantee for other prompts or models.

## Architecture

The app launcher manages a local FastAPI service and private writable data. The service loads the MLX engine for diffusion or Torch MPS for Real-ESRGAN. GPU jobs run serially. Adapted Apple MLX reference modules provide the SD1.5 model structure; the accepted attention/convolution paths keep all pixels, tokens and sampling steps. Allocator limits and schedule metadata reuse are not a manually managed activation workspace.

## Known limitations

SD1.5 only; **SDXL, FLUX and ControlNet are unsupported**. LoRA support is limited to the documented standard SD1.x formats. Apple Silicon only; M1–M4 and other memory configurations are not validated. Fanless thermal behavior and background applications can change performance. This remains research/alpha software, with unsigned distribution and incomplete broad compatibility testing.

## Development, roadmap and license

Read [CONTRIBUTING.md](CONTRIBUTING.md), [ROADMAP.md](ROADMAP.md), and [CHANGELOG.md](CHANGELOG.md). No competitor results are published; the proposed fair-comparison protocol is [documented separately](docs/competitive-benchmark.md).

Original project code is [MIT licensed](LICENSE). Adapted Apple code retains MIT notices; Real-ESRGAN code retains BSD-3-Clause notices. Dependencies and model weights have their own terms; see [third-party inventory](docs/third-party-licenses.md). The application license does not relicense checkpoints or LoRAs.

Provided as-is, without warranty. Use only images and model weights you have permission to use. Reconstructed details may be plausible rather than faithful to an original scene. This project is not affiliated with or endorsed by Apple, Stability AI, or the upstream model authors.
