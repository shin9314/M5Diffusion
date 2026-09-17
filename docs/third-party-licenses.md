# Third-party licenses and model provenance

M5Diffusion's original code uses the [MIT license](../LICENSE). This does not relicense dependencies, vendored code or model weights.

## Included source and runtimes

| Component | Source / notice | License |
| --- | --- | --- |
| Vendored MLX Stable Diffusion example | [Apple source](https://github.com/ml-explore/mlx-examples/tree/796f5b53cab69a3d48a44233ce21aae889e94a08/stable_diffusion), [retained notice](../vendor/MLX-EXAMPLES-LICENSE) | MIT, copyright Apple Inc. |
| Real-ESRGAN SRVGG architecture adaptation | [upstream](https://github.com/xinntao/Real-ESRGAN), [retained notice](third-party/Real-ESRGAN-LICENSE.txt) | BSD-3-Clause, copyright Xintao Wang |
| Private CPython runtime | [python-build-standalone](https://github.com/astral-sh/python-build-standalone), CPython 3.10.21 | Python Software Foundation license and bundled component notices; retain the runtime's notices |
| Python and native dependencies | [locked versions](../requirements-lock.txt), [machine-readable inventory](third-party/dependency-inventory.json) | Upstream licenses, listed below |

The application bundle must retain distribution license/NOTICE files and Python runtime notices. Some wheels contain native libraries under additional licenses: a top-level package label does not replace those notices. The inventory records notice paths relative to installed site-packages; it contains no developer home paths. Development/test packages in the lock are included for completeness, even when not required for normal inference.

## Bundled configuration and tokenizer assets

`assets/sd15-config/` contains configuration JSON and tokenizer vocabulary/merges copied from the SD 1.5 Diffusers layout; it contains no model tensors. Retain the [CreativeML Open RAIL-M full text](third-party/CreativeML-Open-RAIL-M.txt) from [CompVis](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE) and the [OpenAI CLIP MIT notice](third-party/OpenAI-CLIP-LICENSE.txt) for the CLIP tokenizer origin. These notices are separate from the application MIT license. The configuration describes the original pipeline; supported runtime behavior is documented in this project's README.

## Models are separate

No diffusion or upscaler weights are included in source control or the distributable application. Imported checkpoints and adapters retain their own licenses and usage conditions. Review the model publisher's terms; M5Diffusion's MIT license does not grant rights to them.

Stable Diffusion 1.5 uses CreativeML Open RAIL-M terms; see the [official Apple SD 1.5 model card](https://huggingface.co/apple/coreml-stable-diffusion-v1-5) and [CreativeML Open RAIL-M text](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE). Derived checkpoints may add or change applicable terms.

The optional Real-ESRGAN download is the official `realesr-general-x4v3.pth` release asset, 4,885,111 bytes, SHA-256 `8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292`. See [upscaler provenance](upscale.md) for the exact upstream architecture, release and license links. It is obtained only through explicit setup and kept in local model storage.

## Locked dependency inventory

License labels below are descriptive metadata from the installed distributions, not a substitute for their complete license text. `See distribution notices` entries include multi-license or long-form notices in the machine-readable inventory.

| Package | Version | Declared license metadata |
| --- | --- | --- |
| accelerate | 1.15.0 | Apache |
| annotated-doc | 0.0.5 | MIT |
| annotated-types | 0.8.0 | MIT |
| anyio | 4.15.1 | MIT |
| certifi | 2026.7.22 | MPL-2.0 |
| charset-normalizer | 3.5.1 | MIT |
| click | 8.5.0 | BSD-3-Clause |
| diffusers | 0.35.1 | Apache 2.0 License |
| exceptiongroup | 1.3.1 | License :: OSI Approved :: MIT License |
| fastapi | 0.141.1 | MIT |
| filelock | 3.32.7 | MIT |
| fsspec | 2026.7.0 | BSD-3-Clause |
| h11 | 0.16.0 | MIT |
| hf-xet | 1.6.0 | Apache-2.0 |
| huggingface_hub | 0.36.2 | Apache |
| idna | 3.19 | BSD-3-Clause |
| ImageIO | 2.37.4 | BSD-2-Clause |
| importlib_metadata | 9.0.1 | Apache-2.0 |
| iniconfig | 2.3.0 | MIT |
| Jinja2 | 3.1.6 | License :: OSI Approved :: BSD License |
| lazy-loader | 0.5 | BSD-3-Clause |
| MarkupSafe | 3.0.3 | BSD-3-Clause |
| mlx | 0.32.2 | MIT |
| mlx-metal | 0.32.2 | MIT |
| mpmath | 1.3.0 | BSD |
| networkx | 3.4.2 | License :: OSI Approved :: BSD License |
| numpy | 2.2.6 | See distribution notices (license text in package metadata) |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pillow | 12.3.0 | MIT-CMU |
| pluggy | 1.6.0 | MIT |
| psutil | 7.2.2 | BSD-3-Clause |
| pydantic | 2.13.5 | MIT |
| pydantic_core | 2.46.5 | MIT |
| Pygments | 2.21.0 | BSD-2-Clause |
| pytest | 9.1.1 | MIT |
| PyYAML | 6.0.3 | MIT |
| regex | 2026.9.10 | Apache-2.0 AND CNRI-Python |
| requests | 2.34.2 | Apache-2.0 |
| safetensors | 0.8.0 | License :: OSI Approved :: Apache Software License |
| scikit-image | 0.25.2 | See distribution notices (license text in package metadata) |
| scipy | 1.15.3 | See distribution notices (license text in package metadata) |
| starlette | 1.6.0 | BSD-3-Clause |
| sympy | 1.14.0 | BSD |
| tifffile | 2025.5.10 | BSD-3-Clause |
| tokenizers | 0.19.1 | License :: OSI Approved :: Apache Software License |
| tomli | 2.4.1 | MIT |
| torch | 2.8.0 | BSD-3-Clause |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| transformers | 4.44.2 | Apache 2.0 License |
| typing-inspection | 0.4.4 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| urllib3 | 2.8.0 | MIT |
| uvicorn | 0.53.0 | BSD-3-Clause |
| zipp | 4.1.0 | MIT |
