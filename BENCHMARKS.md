# Benchmarks

These are local measurements on one fanless MacBook Air M5 with 24 GB unified memory, macOS 26.6, MLX 0.32.2 and Python 3.10. They are not a comparison against Draw Things or a promise for other Macs.

## Controlled release result

The [counterbalanced record](benchmark/release-v0.1.0.json) compares the same MLX implementation with native convolution (A) against selective im2col plus MLX matrix multiplication (B). Both use compiled UNet, padded fused attention and identical model, precision, sampler and steps. The accepted path applies only to eligible 3×3 convolutions with spatial height at most 32 and at least 1,280 input channels; other convolutions retain MLX native execution.

| Warm median, seconds | Native A | Accepted B |
| --- | ---: | ---: |
| Generation total | 8.309 | 6.822 |
| Diffusion | 7.486 | 6.025 |
| Diffusion per step | 0.374 | 0.301 |
| VAE decode | 0.795 | 0.745 |

These pooled medians include 30 warm images per variant. Three sessions used order A/B, B/A, A/B, with 11 images per run and the first image excluded. Each run had a minimum 60-second cooldown. No warm timing outliers were removed. The median paired session diffusion ratio was **1.230×**; this differs from dividing the two pooled medians.

| Session | Order | A diffusion median, s | B diffusion median, s | A/B |
| --- | --- | ---: | ---: | ---: |
| 1 | A/B | 7.532 | 6.295 | 1.197 |
| 2 | B/A | 7.042 | 5.072 | 1.388 |
| 3 | A/B | 7.631 | 6.204 | 1.230 |

Peak MLX allocator usage was about 6,434 MiB in both variants. This is not total system memory or peak process RSS. Recorded system swap growth was zero; global swap occupation and growth are distinct fields.

## Workload and timing boundaries

- Stable Diffusion 1.5, 512×512, batch one, seed 12345, 20 DPM++ 2M Karras steps, rho 7, CFG 7, empty negative prompt.
- Prompt: “a small cozy cabin beside a lake, mountains in the background, golden morning light, landscape photography”.
- Identical NumPy PCG64 float32 NCHW initial latent bytes. FP16 UNet/CLIP; FP32 VAE and latent state. No quantization or LoRA.
- `generation-wall-v2` total is synchronized generation wall time less separately measured diagnostic overhead. It includes preparation and postprocessing. Model loading and PNG encoding are outside that total and must be reported separately for end-to-end comparisons.
- Source recorded at measurement: `bae60a0ec8eea17a3710b1432ccfea758f7be306`, with a working-tree digest in the record. This is measurement provenance, not a claim that the release tag was measured.

Model component SHA-256 values are recorded in the controlled JSON. In particular, the UNet hash is `d27cd69d4a0aa32105087a619f32a51bc087e133be93fe23da92f3c0bcc07d79`. Reproduce with the same weights rather than merely a similarly named checkpoint.

The fixed-request [quality screen](benchmark/release-v0.1.0.json) reported minimum SSIM **0.99827862** and PSNR **48.08 dB** against the preceding MLX path. This is a numerical regression screen, not broad perceptual validation across prompts and checkpoints. Optimizations can change floating-point rounding; bit-identical output is not promised.

Power source and macOS thermal state were recorded. The Mac was on AC while charging at low battery percentage; thermal state varied between nominal and fair. Temperature, GPU clock and power sensors were unavailable and remain null. A fixed cooldown does not establish equal silicon temperature, and the session variation is material. These measurements support this local optimization; they do not isolate every power or thermal effect.

## Archived measurements

Earlier MPS measurements (13.49 s total / 12.012 s diffusion) and earlier MLX measurements (10.879 s total / 10.118 s diffusion) are historical development observations under different conditions. **Do not calculate a release speedup by comparing those numbers with the controlled result above.** Archived JSON remains for provenance, not as a controlled competitor baseline.

The [optimization ledger](docs/optimization-ledger.md), [instrumented profile](docs/diffusion-profile.md) and [Metal experiments](docs/metal.md) distinguish microbenchmarks, instrumented timings and full generation. Instrumentation changes execution and adds synchronization; profile category sums are not uninstrumented GPU duration. Experimental TensorOps support does not establish an end-to-end gain and is not the selected production convolution path.

## Reproduction and future comparisons

Use a dedicated GPU window, close competing GPU applications, keep the same power mode and record battery, charging and thermal state. Use the controlled runner and its recorded commands, retaining cold runs and raw samples. Report uncertainty and unavailable sensors rather than substituting estimates. See the [competitive benchmark protocol](docs/competitive-benchmark.md); no external application ranking has been measured for this release.
