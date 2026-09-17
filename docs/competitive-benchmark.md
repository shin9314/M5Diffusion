# Competitive benchmark protocol

This is a proposed protocol, not a results table. No controlled Draw Things versus PyTorch MPS versus M5Diffusion ranking is available for v0.1.0-alpha.

Use the same physical Mac, macOS release, power source and power mode. Record hardware, memory, battery/charging state, application and framework versions, model/component SHA-256, precision and any quantization. Stop other GPU jobs. Measure each application independently; do not overlap runs.

## Equivalent workload

Use the same SD 1.5 checkpoint and VAE, 512×512, batch one, 20 steps, CFG 7, prompt and negative prompt. Select equivalent DPM++ 2M Karras behavior, including sigma schedule, prediction type and timestep conventions. Confirm that a similarly named sampler actually matches. Reuse identical initial latent bytes if all applications permit it; a shared integer seed alone does not guarantee equal noise. Disable LoRA, refiners, upscaling, previews and hidden acceleration that changes model quality. If a setting cannot be matched, explicitly label the comparison non-equivalent and report it separately.

## Collection

1. Report fresh-launch model load, first image and warm generation separately. Include file decoding or PNG saving only in a separately defined user-facing end-to-end metric.
2. Run at least three counterbalanced sessions, rotating application order. Collect one cold plus at least ten warm images per application/session; retain every sample and explain failures.
3. Allow at least 60 seconds between runs, record thermal state before and after, and extend the wait when needed. Cooldown alone does not prove equal temperature. Leave unavailable clock, power and temperature sensors null.
4. Capture synchronized generation wall time, diffusion time where exposed, VAE time where exposed, memory metric/source, global swap occupation and swap growth. Do not compare an allocator peak directly to process RSS or invent stage times an application does not expose.
5. Publish medians, dispersion, session-level ratios and raw records. Keep cold results distinct; never compare one application's fastest sample to another's median.

## Quality and interpretation

Retain representative outputs with explicit publication permission. Compare matched initial-noise output using SSIM and PSNR plus visual inspection; test varied prompts, seeds and at least one additional compatible checkpoint. Numerical similarity alone is not an aesthetic rating. Disclose precision, model or scheduler differences that prevent a like-for-like comparison.

A speed claim must name hardware, workload, measured boundary, versions and uncertainty. Report external application results only after measurements exist. Keep archived development results separate from a new controlled comparison.
