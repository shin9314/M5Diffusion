# Custom Metal and M5 TensorOps experiments

`m5diffusion/kernels/fused_scheduler.py` combines classifier-free guidance, the FP32 denoised estimate and DPM++ 2M update in one elementwise Metal kernel. It uses the same coefficient order and arithmetic as the engine. `native_scheduler` provides the eager reference; `mx.compile(native_scheduler)` is the fair fused framework comparison. Outputs are newly allocated and owned by MLX; no unsafe manual in-place aliasing is used. The exact element count is dispatched, including nonmultiples of the threadgroup size.

`m5diffusion/metal/tensor_backend.py` implements an experimental FP16/BF16 matrix multiply with actual `mpp::tensor_ops::matmul2d` calls, not a renamed ordinary MLX matmul. It uses one 32-thread simdgroup per 32×32 output tile and float32 output. The deliberately small implementation requires all dimensions divisible by 32 and is not wired into the image engine. It is a capability experiment, not an optimized replacement for MLX's GEMM. Installed MLX 0.32.2 includes MPP/NAX matmul source; framework-native GEMM can already use newer hardware paths.

Run while no image generation or other GPU benchmark is active:

```sh
.venv/bin/python microbenchmark_metal.py
M5DIFFUSION_GPU_TESTS=1 .venv/bin/python -m pytest tests/test_metal.py
```

The report separates compilation/execution correctness from timing. Timing includes Python submission, allocation, and final synchronization, with three warmups and twenty repeated batches of twenty calls. It is not isolated GPU timestamp timing. The custom FP32 scheduler uses compiler-permitted floating-point contraction, so tests tolerate small rounding differences. Its microbenchmark gain must not be described as full-image speedup. Native and MPP matmul use different output dtype (half versus float32), which limits direct interpretation of their timing comparison.

A Metal 4/Apple10 device feature flag, availability of `MTLTensor`, and success allocating an INT4 tensor do not prove INT4 arithmetic, NAX dispatch or a performance improvement. This experiment does not implement packed INT4 MPP arithmetic. The standalone Metal compiler toolchain is absent on this machine; runtime compilation through MLX uses the operating-system Metal compiler and does not require a global installation.

References:
- [Apple MPP Programming Guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf), March 16, 2026: tensor views, simdgroup matmul, tiling.
- [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html): runtime compilation, exact dispatch shape, input handling.
- Installed MLX header: `.venv/lib/python3.10/site-packages/mlx/include/mlx/backend/metal/kernels/steel/gemm/nax.h`.

## Verified result on this Mac

All 16 opt-in GPU tests passed. MPP FP16 and BF16 kernels compiled and executed with maximum absolute errors of 8.58e-6 and 5.72e-6 against FP32 CPU matrix multiplication. MPP's installed headers do not remove `const` when matching tensor element types; input views therefore explicitly remove pointer constness, but are used only as read operands.

For a 512×512 image (1×64×64×4 latent shape), warm scheduler median was 0.0570 ms eager, 0.0351 ms compiled, and 0.0288 ms custom Metal. Maximum absolute deviation was 1.53e-5. This is about six microseconds saved versus the compiled scheduler and does not justify expecting a visible full-image speedup.

MPP matmul performance was mixed: 320×320×320 took 0.0290 ms versus native 0.0240; 1024×320×320 took 0.0351 versus 0.0469; 1024×768×320 took 0.1354 versus 0.0709. Neither a universal advantage nor an end-to-end image-generation benefit is established. See `benchmark/metal.json` for repetitions, minima, and all sizes. TensorOps is available; automatically replacing the framework's optimized GEMMs is not justified by these results.

| Image size | Eager scheduler, ms | Compiled scheduler, ms | Custom Metal scheduler, ms |
|---|---:|---:|---:|
| 512×512 | 0.05704 | 0.03515 | 0.02880 |
| 768×768 | 0.05512 | 0.03519 | 0.02635 |
| 1024×1024 | 0.05596 | 0.03432 | 0.02571 |

Values above are medians from the saved report; no additional GPU benchmark was run for this documentation update. The CPU-only full test suite passed 38 tests and skipped the 16 opt-in GPU tests. GPU tests previously passed all 16 cases.

GPU utilization telemetry queries the driver's whole-device `PerformanceStatistics` counter through read-only `ioreg` approximately once per second; unavailable values remain null. It is not attributable exclusively to the image-generation process. `initial_swap_mb` and `peak_swap_mb` measure system-wide occupied swap, while `swap_growth_mb` is the nonnegative peak increase above the entry snapshot. Existing swap occupation does not establish active swapping during generation.
