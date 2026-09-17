# Diffusion profiling and microbenchmarks

These tools leave the production engine, vendor model, UI, and LoRA unchanged. Run them exclusively; concurrent generation invalidates the comparisons. Every standalone microbenchmark caps the MLX cache at 2 GiB and allocator memory limit at 10 GiB before input allocation.

`profile_diffusion.py` profiles three repeated initial UNet timesteps for the actual SD1.5 checkpoint, 512×512/CFG 7 inputs and padded attention. It temporarily instruments the vendor forward methods and leaf modules. Inputs are explicitly evaluated before a timed operation so work belonging to an upstream lazy producer is not accidentally attributed to the consumer. Attention is an atomic scope including projections and internal layout; nested leaf records are suppressed. Other normalization, convolution and linear layers are also atomic. No nested times are added twice.

Each result separately reports Python graph-construction time and synchronized host elapsed time. The latter includes dispatch, execution and synchronization; it is not hardware GPU time. Subtracting graph-construction time would not make it exact GPU time. Hardware kernel counts, physical allocation counts and GPU-only timestamps are null because the available MLX Python API does not expose them. [MLX Metal capture](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html) can produce an Xcode-inspectable trace with `MTL_CAPTURE_ENABLED=1` and `--capture path.gputrace`, but capture itself is not a timing baseline.

The measured instrumentation slowed the uncompiled step by 1.172×. The instrumented result matched the unmodified step exactly (maximum absolute output difference 0.0). Unmodified repeated step times were 998.6, 1019.5 and 1000.5 ms, with Python graph construction 2.9–9.6 ms. These are diagnostic uncompiled one-step timings under the then-current machine state, not the stable compiled 20-step generation result.

Instrumented category shares:

| Scope | Share of summed diagnostic operation wall |
|---|---:|
| Convolution | 42.95% |
| Linear outside attention | 12.57% |
| Self-attention including projections | 11.79% |
| Normalization | 11.64% |
| Residual addition/subtraction | 6.94% |
| Activation | 5.16% |
| Cross-attention including projections | 4.38% |
| Other elementwise | 1.81% |
| Concatenate/copy | 1.49% |
| Reshape | 0.64% |

Full top-20 records, shapes, counts and timing samples are saved in `benchmark/optimization/profile-diffusion.json`. The three leading named operations were `up_blocks.1.upsample` (43.12 ms, input 2×32×32×1280), `up_blocks.2.resnets.0.conv1` (34.56 ms, input 2×32×32×1920), and `up_blocks.1.resnets.1.conv1` (26.38 ms, input 2×16×16×2560). Categories are diagnostic shares, not an exact decomposition of fused compiled production runtime.

Transpose records outside atomic modules were zero. This does **not** mean the model performs no transposes: attention-internal transposes are included in its atomic time. Reshape/transpose/broadcast can be views, and their operation counts are not physical copies. Copy category covers explicit concatenation or padding only.

## Attention experiment

`bench_attention.py` warms each case three times and measures ten times, recording all samples. It compares native SDPA, explicit matmul/softmax, compiled explicit math, zero-padding to 48/64, query chunking, and online key-tiled stable softmax. The tiled method computes every query-key interaction, but uses FP32 accumulation and has different rounding. No tokens are dropped.

For 2×8 heads, 4096 queries/keys, D=40, median native SDPA was 62.80 ms versus 24.64 ms with padding to 64 **including allocation/padding/unpadding**. Query chunking took 48.30 ms; the Python-composed online tiled algorithm took 131.74 ms. Padding to 48 used the slower nonfused fallback (74.76 ms); forcing fusion reported unsupported dimension rather than fabricating a timing. Padding-only cost was 1.225 ms. The padded result differed from native by maximum absolute 0.0004883, RMSE 0.00001947 on random test inputs.

At 4096 queries/77 keys, native D=40 SDPA took 0.946 ms, explicit attention 0.754 ms, and padded64 1.841 ms. These results support preserving the existing selective self-attention padding and not padding cross-attention. They do not justify a new custom Metal attention kernel. See `benchmark/optimization/attention.json`.

## Convolution experiment

`bench_conv.py` also uses three warmups and ten measured repetitions. Native MLX NHWC/OHWI convolution, compiled convolution, conv+SiLU, conv+residual, NCHW-to-NHWC roundtrip and physical layout copy are separated. The NCHW test calls the same native NHWC operator with transposes, not an independent NCHW kernel.

The dominant 32×32, 1280→1280 convolution took 48.59 ms native versus 50.03 ms compiled. Conv+SiLU took 53.58 ms eager versus 48.55 ms compiled. At 32×32, 1920→640, native and compiled took 40.90 and 40.67 ms. Layout roundtrip was slower on all tested shapes. This identifies convolution as the next high-value target but is not evidence to change the production layout. No direct MPS or TensorOps convolution comparison has yet been measured by this script. See `benchmark/optimization/conv.json`.

## Measured top 20 operations

Rows are ranked by cumulative recorded wall time across three instrumented first timesteps. The time column is that sum divided by three; a grouped residual row includes multiple same-shaped additions per step. These are measured synchronized host times, not hardware-kernel timings.

| Rank | Operation | Category | Mean cumulative ms / step | Calls / step |
|---|---|---|---:|---:|
| 1 | `up_blocks.1.upsample` | convolution | 43.117 | 1 |
| 2 | `up_blocks.2.resnets.0.conv1` | convolution | 34.563 | 1 |
| 3 | `up_blocks.1.resnets.1.conv1` | convolution | 26.375 | 1 |
| 4 | `up_blocks.2.resnets.1.conv1` | convolution | 23.383 | 1 |
| 5 | `up_blocks.1.resnets.0.conv1` | convolution | 21.325 | 1 |
| 6 | `up_blocks.3.attentions.2.transformer_blocks.0.attn1` | self_attention | 18.956 | 1 |
| 7 | `up_blocks.1.resnets.2.conv1` | convolution | 18.932 | 1 |
| 8 | `down_blocks.0.attentions.0.transformer_blocks.0.attn1` | self_attention | 18.510 | 1 |
| 9 | `up_blocks.2.resnets.2.conv1` | convolution | 17.167 | 1 |
| 10 | `up_blocks.3.attentions.1.transformer_blocks.0.attn1` | self_attention | 16.885 | 1 |
| 11 | `down_blocks.0.attentions.1.transformer_blocks.0.attn1` | self_attention | 16.880 | 1 |
| 12 | `up_blocks.1.resnets.1.conv2` | convolution | 15.634 | 1 |
| 13 | `up_blocks.2.resnets.0.conv2` | convolution | 15.141 | 1 |
| 14 | `up_blocks.3.attentions.0.transformer_blocks.0.attn1` | self_attention | 13.792 | 1 |
| 15 | `up_blocks.2.resnets.1.conv2` | convolution | 13.772 | 1 |
| 16 | `add` | residual | 13.460 | 15 |
| 17 | `up_blocks.1.resnets.2.conv2` | convolution | 13.292 | 1 |
| 18 | `up_blocks.2.resnets.2.conv2` | convolution | 13.124 | 1 |
| 19 | `down_blocks.2.resnets.0.conv2` | convolution | 12.497 | 1 |
| 20 | `up_blocks.1.resnets.0.conv2` | convolution | 12.433 | 1 |

## Profile-directed follow-up: batch split, TensorOps and MPS

The follow-up ran under a faster machine state than the earlier diagnostic session. Compare alternatives against their own same-run baseline; absolute timings from the two sessions are not interchangeable. All rows below use three warmups and ten timed repetitions. Shapes are input spatial H=W and channels Ci→Co, with batch two and 3×3 kernels.

| Shape | MLX native | Two batch-one calls | Existing MPP matmul + im2col | MPP / native speedup |
|---|---:|---:|---:|---:|
| 32, 1280→1280 | 19.256 ms | 19.955 ms | 8.940 ms | 2.154× |
| 32, 1920→640 | 14.978 ms | 15.912 ms | 7.796 ms | 1.921× |
| 16, 2560→1280 | 10.375 ms | 11.330 ms | 5.890 ms | 1.762× |

Batch splitting preserved outputs exactly but was slower: reject that candidate. The existing MPP prototype includes input padding, patch formation/contiguity, weight transpose/contiguity and final FP32→FP16 cast. It improved these three native-convolution microbenchmarks, with maximum absolute error 0.001953/0.003906/0.003906 respectively. This confirms actual TensorOps execution, not neural-accelerator hardware utilization. Native MLX GEMM-based im2col is being compared separately; these results alone do not establish MPP superiority to that alternative or whole-image quality/speed. The prototype is isolated in `bench_conv_candidates.py`, not installed in the production engine.

`bench_conv_mps.py` uses the same CPU-generated deterministic FP16 values in both runtimes and excludes one-time upload and result-readback. For those shapes, MPS native NCHW medians were 6.768/4.879/5.389 ms, while its NHWC boundary with layout views took 7.038/5.517/5.292 ms. Explicitly materializing both boundary conversions took 6.603/5.020/5.369 ms. Small view-versus-copy differences fall within runtime/kernel-selection variability; do not infer copies are free. Maximum absolute discrepancy from MLX native was 0.003906.

MLX convolution-plus-residual compilation measured 20.022/15.632/10.695 ms, versus eager 20.144/15.576/11.196 ms; no consistent major gain. MPS is a runtime comparison, not a demonstrated MLX↔MPS zero-copy integration. Actual mixed-runtime generation would need additional integration and synchronization. Raw samples, errors and notes are in `benchmark/optimization/conv-candidates.json` and `conv-mps.json`.

## Workspace reuse: supported behavior and limits

The installed MLX 0.32.2 `conv2d`, `matmul`, and `add` signatures return arrays without an `out=` buffer argument (verified in the local `mlx/core/__init__.pyi`). The [custom Metal kernel API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html) likewise specifies output shapes and dtypes, not externally allocated output buffers. `mx.compile(inputs=..., outputs=...)` captures and updates array state; that API is not a promise that a caller-selected Metal allocation will be overwritten safely.

This does **not** mean MLX arrays cannot be updated in Python. It means preallocating an array and assigning a later expression to it is not evidence that convolution scratch, attention workspace or residual allocations have been reused. Reporting such assignment as a full workspace pool would be misleading. MLX already owns a reusable allocator cache; its [cache-limit API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html) documents reclamation and allows cache-disable experiments. Cache experiments and active/peak-byte telemetry can measure effects, but cannot yield an unavailable physical-allocation count.

A true manually controlled intermediate pool would require a custom extension/backend contract that explicitly handles output ownership, aliasing and asynchronous lifetime. Overwriting buffers still referenced by a lazy graph can change results. No speculative mutation scheme was added. The existing immutable schedule/timestep cache remains accurately described as metadata reuse; full activation-workspace reuse is not implemented. [Lazy evaluation documentation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html) also explains why scalar access, NumPy conversion and excessive evaluation boundaries can impose synchronization. Production diffusion avoids these host accesses inside its step loop.
