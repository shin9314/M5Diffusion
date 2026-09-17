# Contributing

This is experimental alpha software. Keep changes small and identify whether they affect installation, UI, inference numerics or performance. During v0.1.0-alpha release preparation, the accepted UNet convolution/attention paths, compilation settings, scheduler and VAE numerics are frozen.

Use a private virtual environment and the locked dependencies. Run CPU tests with `.venv/bin/python -m pytest -q`. Hardware tests are opt-in using `M5DIFFUSION_GPU_TESTS=1`; run them on a supported Mac with exclusive GPU access, and report the hardware and OS. Never compare timings while another GPU job is active.

A performance change needs a measured bottleneck, a single isolated change, counterbalanced warm repetitions, quality checks and memory evidence. Keep rejected experiments documented. Do not reduce resolution/steps or silently replace a checkpoint to improve headline numbers. Hardware counters unavailable to the test must remain unavailable, not estimated and presented as measured.

Do not commit model weights, user images, output PNGs, credentials, local logs or personal filesystem paths. Keep upstream copyright/license notices. Describe imported code and dependencies, and separate model-license obligations from code licensing.

Before a release: run CPU and GPU tests, generation and consecutive-image checks, upscaling, app start/quit/restart, clean model setup and intentional error cases. Verify the built app/DMG, collect dependency notices, scan the public tree and Git history, and record the source SHA. Publishing, tagging and remote pushes should be deliberate release actions; do not push a work-in-progress build automatically.
