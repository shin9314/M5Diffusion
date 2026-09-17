# Supported LoRA adapters

The MLX SD1.x FP16 engine applies real additive weight updates. Importing a LoRA does not merely store or label it: activation forms each delta as `scale × alpha / rank × (up @ down)`, merges into the selected model, and rebuilds the compiled diffusion callable. If alpha is absent, it defaults to rank. Several selected adapters sum their updates. Removing all adapters restores the exact original weights; switching adapters always starts from those originals, so updates do not accumulate accidentally.

Only targeted original parameters are retained while adapters are active. All adapter files, every tensor name, rank, finite value, base target, and output shape are validated before model weights change. Unknown keys are errors rather than silently discarded tensors. Adapter import validates syntax and contents; checkpoint-specific compatibility is validated when applying to that checkpoint.

Supported safetensors variants:

- Standard Kohya/A1111 `lora_unet_…`, `lora_te_…`, or `lora_te1_…` with paired `.lora_down.weight` / `.lora_up.weight` and optional scalar `.alpha`.
- Kohya UNet names using Diffusers `down_blocks` / `mid_block` / `up_blocks`, or original SD1.x LDM `input_blocks` / `middle_block` / `output_blocks`. Original LDM resnet submodule names and standard down/up samplers are mapped as well.
- Diffusers/PEFT names prefixed with `unet.` or `text_encoder.`, with `.lora_A.weight` / `.lora_B.weight` (including the conventional `.default` adapter slot).
- Legacy Diffusers `.lora.down.weight` / `.lora.up.weight`, attention `to_q_lora` / `to_k_lora` / `to_v_lora` / `to_out_lora`, and CLIP `.lora_linear_layer.down.weight` / `.up.weight`.
- UNet and CLIP linear adapters, and ordinary convolution adapters with down k×k + up 1×1, or down 1×1 + up k×k. Compatible 2D versus 1×1 forms are accepted.
- GEGLU feed-forward targets: the combined delta is split through the same vendor weight mapper used for loading the base model. Convolution layout conversion also uses that mapper.

Not supported: SDXL/SD2/Flux adapters, LoHa, LoKr, DoRA, LyCORIS variants, bias tensors, both convolution factors having spatial kernels larger than 1×1, arbitrary named PEFT slots, unsafe pickle/`.pt`/`.ckpt` adapter files, or merging into a quantized engine. Renaming a file does not make another model family compatible. A file without identifying metadata still must match every target and shape in the actual SD1.x checkpoint.

CPU BF16 safetensors decoding uses the installed Torch library, on CPU only. Applying validated updates and image generation use MLX. Source naming conventions were checked against the installed Diffusers `loaders/lora_conversion_utils.py` and `utils/state_dict_utils.py`; the application does not call converters that discard unrecognized tensors.

## Verification

`tests/test_lora.py`: 15 CPU cases plus one opt-in GPU case passed. Coverage includes both Kohya name families, modern/legacy Diffusers naming, alpha and strength arithmetic, both supported convolution orientations, malformed pairs, NaN, unknown targets, metadata rejection, multiple adapters, exact restoration, failed-switch atomicity, and GEGLU splitting.

A real SD1.5 engine with compiled padded attention was also tested with a synthetic nonzero convolution LoRA at an 8×8 latent size. Applied output changed by maximum absolute 0.0039551; removing the adapter restored both the original weights exactly and compiled output with maximum absolute difference 0.0. No original-weight backup remained after removal. See `benchmark/lora-regression.json`. This regression verifies weight and graph replacement, not the artistic effect of any downloaded adapter.

```python
engine.set_loras([(path_to_adapter, 0.8), (second_adapter, 0.3)])
engine.set_loras([])  # Exact restoration of the original model weights.
```
