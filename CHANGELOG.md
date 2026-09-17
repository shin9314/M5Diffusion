# Changelog

## v0.1.0-alpha

Initial experimental distribution for Apple Silicon.

- Local SD1.5 inference using MLX and Metal; accepted selective im2col/native matmul convolution and padded self-attention, with unchanged 20-step quality-comparison conditions.
- Counterbalanced optimization evidence, fixed-request quality checks, and documented measurement limitations.
- Local browser UI, SD1.5 safetensors import, limited standard LoRA loading, and saved PNG/metadata.
- Uploaded-image Real-ESRGAN 2×/4× enhancement with transparency preservation.
- Mac app/DMG distribution preparation with a private runtime, startup checks, model setup and friendly failures.
- Public-facing installation, benchmark, roadmap, contribution and license documentation.

Developer ID signing/notarization and validation across other Mac generations are not completed. Model weights are not included in the repository or release disk image. No competitor ranking is claimed.
