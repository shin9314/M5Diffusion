# Progress

## Phase 0 — infrastructure
- Dedicated ~/M5Diffusion and venv; existing environment untouched.
- M5 base, 10 CPU and 10 GPU cores, 24 GiB; macOS 26.6; Xcode 27.0.
- Metal4 and Apple10 runtime queries true; FP32/FP16/BF16 MLX matmul passed; Metal tensor allocation FP32/FP16/BF16/INT8/INT4 passed. Allocation does not prove accelerated arithmetic. Public FP8 tensor enum requires macOS 27 and is not enabled here.
- Shared deterministic scheduler/noise unit tests: 3 passed.
- Hardware facts: benchmark/results/hardware-*.json.

## Phase 1 — initial MLX inference
- Strict loading of all SD1.5 weights succeeded; generated 512², 20-step images.
- Model load 1.35 s; cold generation 16.29 s; warm 10.63 s.
- Profiling by synchronized stages: warm diffusion 9.06 s; VAE 1.48 s; CLIP .046 s; preparation .038 s; readback .0005 s.
- MAJOR regression: default MLX allocator cache grew to 15 GiB, system swap increased to 9.8 GiB. Do not use this default in production.
- Action: limit the allocator cache and memory budget before further comparison. This initial unbounded result is diagnostic, not an accepted speedup result.
- Unit test: 3 passed. Initial images saved; cross-backend quality validation pending.

## Sources
- https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion
- https://developer.apple.com/videos/play/tech-talks/111432/
- Local Xcode SDK Metal/MTLTensor.h and live Metal runtime probe.

## Metrics limitations
No privileged collectors or sudo used. Temperature, GPU frequency and watts unavailable: null, not zero. GPU utilization now sampled read-only from the driver (whole-device, not process-specific). Thermal state available from NSProcessInfo. Process RSS, MLX allocator peak, driver allocations and whole-system memory are distinct and must not be added. Global swap can remain occupied after a benchmark exits.

## Phase 2 — memory and residency
- Models stay resident within each benchmark/UI process. Immutable schedule metadata cached by shape/steps. This is NOT a user-controlled activation-buffer pool; MLX owns/recycles those buffers.
- Allocator cache bounded at 2 GiB after a 256 MiB trial caused allocation churn. MLX allocation budget 10 GiB; do not count this as preallocated memory.
- Initial 15 GiB cache regression removed. Global swap remains occupied from earlier runs; report before/after, never claim zero swap.

## Phase 3 — compile
- Whole-step mx.compile tested on standard attention, three images; warm stage-total 21.01/21.74 s. Not selected on this evidence.
- Cold compile time included in first generation; no fabricated isolated compilation metric.

## Phase 4 — attention
- Synchronized diagnostic profile saved in mlx-op-profile.json. IMPORTANT: MLX is lazy; a Linear timing can include its preceding SDPA. MultiHeadAttention includes children; do not sum inclusive timings.
- Runtime confirmed no fused attention for SD1.5 head dimension 40. Standard full SDPA 30.82 ms; zero-padded D=64 fused SDPA 6.91 ms (4.46x for this operator). Original 1/sqrt(40) scaling retained; padding changes no mathematical attention term.
- End-to-end padded MLX: 10.18/10.68 s warm at 512²/20 steps, vs matched MPS warm14.18 s in prior serial run (provisional ~1.36x; thermal/order caveats).
- Peak MLX allocation 6434 MiB including VAE workspace; final active allocation2194 MiB, allocator cache~2163 MiB.
- Exact shape choices and errors saved in attention-probe.json; full quality comparison in quality-*.json.

## Phase 5 — custom Metal
- Implemented fused FP32 CFG + DPM++ 2M update with shape-tail and final-step correctness tests.
- For the 512px image latent: eager 0.0570 ms, compiled 0.0351 ms, custom Metal 0.0288 ms; max absolute error 1.53e-5.
- This operation contributes very little to total image time; custom version not silently promoted as a major engine speedup.

## Phase 6 — MPP TensorOps
- Dynamic Metal compilation and actual FP16/BF16 MPP matmul succeeded on this machine. 16 opt-in GPU tests passed.
- Shape-dependent performance: 320-cubed slower, 1024x320x320 faster, 1024x768x320 slower than MLX matmul. Mixed results and output precision caveat preclude default replacement.
- INT4 arithmetic still unverified. Public FP8 requires a newer OS. See docs/metal.md and benchmark/metal.json.

## Phase 12 — early usable UI milestone
- Separate loopback-only Japanese UI and serialized API. Dedicated worker preserves MLX stream ownership and model residency.
- Desktop M5Diffusion.app launches/reuses the service. Existing A1111 application untouched.
- Default padded + compiled graph selected after equivalent-quality trial; SD1.5 only; unsupported LoRA visibly disabled.
- UI tests use a fake engine; actual browser Generate→PNG display→download/metadata links verified, 512x512/20steps. Model-resident second generation also checked before delivery.

## Measurement correction
- Early total_time was a sum of stage timers. New generation-wall-v2 includes preparation, initialization, readback and postprocessing, excluding measured diagnostic export; model load/PNG remain separate.
- Whole generation speedups must not mix timing schema versions. Earlier 1.36x was provisional and cannot describe sustained performance.
- GPU workloads are serialized across root/subagents to avoid benchmark interference. Other desktop activity, thermal state and pre-existing system swap remain potential confounders.

## Phase 9 — sustained measurement (initial milestone)
- Same SD1.5 weights, seed12345, 512x512, 20 steps, CFG7, DPM++2M Karras. New complete-generation timers, model load/PNG excluded.
- 10 warm MLX images: median10.8794 s, range8.5281–12.0457; first-five median9.3182 versus last-five11.7008 (+25.6%).
- Subsequent 10 warm MPS images: median13.4902 s, range12.3376–18.8578. Descriptive median ratio1.2400x only; sequence/thermal state confounded.
- Reverse-order MLX recheck: 3 warm images median14.8183 s, range14.1528–14.8561. Earlier short MPS median9.1631 s. Variation rules out a stable1.3x claim at this stage.
- MLX peak allocation6434MiB (~6.28GiB), active2194MiB. No occupied-swap growth in these runs; ~8.6GiB system swap was already occupied from earlier experiments. No claim of zero system swap or measured zero swap I/O.
- GPU driver utilization recorded. Temperature/frequency/power unavailable; cannot uniquely attribute slowdown to thermal throttling. Machine drawing AC power during check; power settings unchanged.
- 30-minute run not performed. Goal of reproducible baseline-vs-MLX image comparison is achieved; sustained1.3x objective remains unproven.

## Regression and current bottleneck
- CPU suite38 passed,16 opt-in tests skipped; Metal agent separately ran all16 GPU tests successfully.
- Final image similarity MPS vs optimized MLX: PSNR50.3511dB, SSIM0.998390, latent relativeL2 .006166. This is one prompt/seed, not dataset-wide quality assurance.
- Post-optimization diagnostic top10 saved in mlx-padded-op-profile.json. Largest measured module is an upsample Conv2d (~31.8ms); other Conv2d and attention dominate. Inclusive/lazy module timing is diagnostic, not hardware kernel-count measurement.
- Phase7 quantization flags exist but quality/performance adoption is unvalidated. Phase8 VAE independently timed (~.7–1.0s in MLX), no tiled/custom-convolution optimization adopted. Phase10/11/13 and ControlNet remain future work.

## Model library and Phase 11 — standard SD1.x LoRA
- UI supports file selection, drag/drop and local-path import; models use validated SD1.5 .safetensors checkpoints, LoRA uses validated standard SD1.x variants. Unsupported architectures and adapter tensors fail explicitly.
- Model checkpoint conversion runs in a separate CPU subprocess, offline, cached atomically; strict full reference tensor shapes prevent partial models being randomly initialized. Actual conversion verified (UNet686/VAE248/CLIP196 tensor shapes plus sampled values).
- Stable content-hash IDs, streaming uploads with limits/disk checks, duplicate prevention and catalog errors for invalid manually placed files.
- One resident base model per generation worker. Switching LoRAs merges scaled deltas from retained original target weights without reloading the base; compiled graph rebuilt on adapter changes; up to4 adapters.
- 74 CPU tests passed,17 GPU tests skipped in default suite. LoRA-specific GPU test separately passed; real compiled SD1.5 regression restored weights and outputs exactly after removal (benchmark/lora-regression.json).
- Actual UI path-import of existing checkpoint and synthetic nonzero LoRA passed; real upload endpoint returned201 and deduplicated the adapter. UI generated512x512/20steps with LoRA0.7, then removed adapter and generated again. Synthetic adapter checks correctness, not artistic quality of third-party trained adapters.
- SDXL/SD2/FLUX, unsupported LyCORIS variants and quantized-model LoRA remain outside supported scope.
