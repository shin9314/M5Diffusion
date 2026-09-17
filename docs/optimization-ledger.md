# SD1.5 diffusion optimization ledger

Scope: fixed512x512,20steps,CFG7,seed12345, unchanged model/prompt/DPM++2M Karras. UI, LoRA and new models are outside this phase.

## Initial reproduction — observation, not an optimization
Optimization: none; current compiled padded MLX at commit4ea6126.
Baseline: historical10.118s diffusion median.
After: 16.619520s diffusion, 18.285409s total,10warm images.
Speedup: not comparable across sessions/power conditions; no causal speedup claimed.
Quality: current implementation unchanged.
Memory: peak6434MiB MLX; recorded global swap0.
Accepted/Rejected: reference observation only.
Reason: previous10.118s not reliably reproduced in current conditions. AC reported but19% battery discharging at initial read; thermal_state0 independently probed.
Evidence: benchmark/results/20260916-234827-perf-reproduction.json

## Measurement policy
- At least3counterbalanced pairs,11images perbackend with first excluded,60s minimum cooldown.
- Record total,diffusion,step,VAE statistics and per-image thermal; unavailable sensors null.
- Instrumented shares are diagnostic, not percentages of production compiled GPU kernel time.
- One candidate change per adoption; retain rejected evidence.

## Selective explicit im2col + native MLX GEMM — pending controlled confirmation
Optimization: replace only stride-1/pad-1 3x3 UNet convolutions with input spatial size <=32 and input channels >=1280; keep other convolutions unchanged. Preserve batch=2, FP16, weights, bias, all convolution terms, scheduler and 20 steps.
Baseline: same-window screening current compiled MLX diffusion 7.053598s (one warm image; not final).
After: 5.431446s (one warm image); independent compile harness 10-image median 5.474105s.
Speedup: preliminary single warm ratio 1.299x; controlled pairs required for adoption.
Quality: current MLX vs candidate SSIM 0.9982786 / PSNR 48.0836dB. Archived matched MPS vs candidate SSIM 0.9981988 / PSNR 49.6502dB. Conditioning unchanged against MLX, latent relative L2 0.00917. These are one fixed prompt/seed, not a general quality guarantee.
Memory: preliminary full-generation peak 6434.46MiB vs native6434.43MiB (VAE dominates whole-generation peak).
Accepted/Rejected: PENDING.
Reason: actual profile identified convolutions as largest scope; microbenchmarks show large gain only at low-resolution/high-channel shapes. Differences are FP16 rounding; no terms or steps removed.
Evidence: benchmark/optimization/conv-im2col.json; quality-im2col-screen.json; quality-im2col-vs-mps.json; benchmark/results/*resume-screen-*.json.

## Existing padded fused self-attention — retained
Optimization: head dimension40 ->64 only for large self-attention, original scale retained.
Baseline: native62.80ms on4096x4096/D40.
After:24.64ms including padding/copy/unpadding.
Speedup:2.55x microbenchmark.
Quality: random-input maximum error0.0004883, RMSE0.00001947; already part of current production baseline.
Memory: no new production change.
Accepted/Rejected: RETAIN existing implementation.
Reason: chunked48.30ms, tiled131.74ms, pad48fallback74.76ms were slower; forced48fusion unsupported. Cross-attention pad64 also slower(1.841 vs0.946ms), rejected. No new custom attention kernel justified by profile.

## Conv layout, batch split and pointwise fusion — rejected
Optimization: NCHW roundtrip, split CFG convolution batch2 into batch1 calls, compile Conv+residual separately.
Baseline: same-window top3 native19.26/14.98/10.38ms.
After: split19.96/15.91/11.33ms; compiledConv+residual20.02/15.63/10.70ms (extra residual workload).
Speedup: no material improvement.
Quality: batch-split maxerror0.
Memory: no production change.
Accepted/Rejected: REJECTED.
Reason: native UNet already NHWC; added layout conversions and separate dispatch do not help. Compile fusion is workload-dependent; no further fusion adopted from these samples.

## MPP TensorOps and MPS convolution — measured, not integrated
Optimization: existing runtime MPP matmul plus explicit patches; independent Torch MPS convolution.
Baseline: same-window MLX native19.26/14.98/10.38ms.
After: MPP8.94/7.80/5.89ms; MPS nativeNCHW6.77/4.88/5.39ms.
Speedup: MPP1.76–2.15x operator-only. No claim it beats the separate native-GEMM experiment under different power/thermal conditions.
Quality: MPP maxerror0.00195/0.00391/0.00391.
Memory: bounded isolated microbenchmarks; independent backend setup and readback excluded from MPS timers.
Accepted/Rejected: NOT INTEGRATED.
Reason: native-MLX GEMM can realize the same profile-directed strategy without backend interoperability. MPS timing does not measure MLX-to-MPS transfer/synchronization. No hardware Neural Accelerator utilization is proven.

## Explicit activation workspace — not adopted
Optimization: considered writable latent/residual/attention/conv/CFG buffers.
Baseline: existing bounded MLX allocator cache and immutable schedule metadata.
After: no manual physical-buffer pool implemented.
Speedup: unavailable.
Quality: unchanged.
Memory: real active/cache/peak allocator gauges measured; allocation event count and cumulative bytes unavailable(null).
Accepted/Rejected: NOT IMPLEMENTED.
Reason: public convolution/matmul/elementwise APIs have no out= destination-buffer contract. Python assignment or compiled state outputs are not proof of physical buffer reuse. Cache0/2048MiB measurements test the real allocator cache, not a fabricated explicit workspace optimization.

## Compile scope screening — retain current full-step scope
Optimization: none / UNet-only / attention-only / UNet+CFG / full-step / two-step graph, all with the same selective-im2col candidate.
Baseline: full-step warm10 median5.474105s; first call5.442239s.
After: none5.871967s; UNet-only5.804767s; attention-only6.652285s; UNet+CFG7.088500s; two-step6.762498s.
Speedup: no reliable additional scope improvement established. Scopes ran sequentially and later runs slowed; these absolute differences are thermally/order confounded and not controlled causal scope ratios.
Quality: all scope PNGs pixel-identical to the full-step candidate(SSIM1,PSNR infinite).
Memory: all scopes completed under10GiB memory cap; raw per-sample active/peak/cache gauges saved.
Accepted/Rejected: RETAIN current full-step compilation; do not adopt another scope on this evidence.
Reason: no robust win demonstrated. First-call time includes tracing/compilation/execution and is not isolated compiler duration. Exact hardware GPU duration unavailable; dispatch calling-thread CPU median0.351s per20steps vs5.474s elapsed, process CPU0.562s. Dispatch elapsed5.425s includes runtime backpressure; do not mislabel it as Python CPU work.
Evidence: benchmark/optimization/compile-im2col-*.json and per-sample JSONL/CSV.


## Controlled acceptance: selective im2col
Optimization: same selective im2col candidate, current full-step compilation retained.
Baseline: session warm10 medians7.532/7.042/7.631s (see raw summary for exact values).
After: session warm10 medians6.295/5.072/6.204s.
Speedup: all three paired sessions improve; aggregate and dispersion in final report.
Quality: fixed-request SSIM/PSNR recorded in quality JSON and final report. No quality/step/resolution settings changed.
Memory: per-image allocator/RSS/swap retained.
Accepted/Rejected: ACCEPTED. Stop condition A: all three10-image medians below7.5s.
Reason: counterbalanced AB/BA/AB,11images each,first excluded,60s cooldown each; model/source hashes verified. Some measured native images were extreme outliers and remain in raw/mean/std; median is not a claim that every image took the median time. Temperature causal degradation unavailable; late/early elapsed change reported separately.
Production: identical function/class/installer source extracted from the measured candidate; only imports relocated. Existing LoRA weight names/types remain inherited Conv2d.
Allocator: cache0 median10.059760s versus2048MiB10.436525s in sequential diagnostic native runs; order drift precludes adoption. Existing bound retained.
