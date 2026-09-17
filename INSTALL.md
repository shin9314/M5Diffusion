# Install M5Diffusion v0.1.0-alpha

## Mac app

1. Use an Apple Silicon Mac running macOS 26 or later, with at least 16 GB unified memory. The validated machine is an M5 MacBook Air, 24 GB; other generations remain unverified.
2. Download the latest DMG from this repository’s Releases page. Open `M5Diffusion-v0.1.0-alpha.dmg`, then drag **M5Diffusion.app** into **Applications**.
3. Open the app. Its private runtime requires no Homebrew, sudo or changes to system Python. First launch prepares application data and opens a browser.
4. In model setup, import a compatible SD1.5 `.safetensors` checkpoint you are licensed to use. Checkpoints are not bundled. For uploaded-image enhancement, use the separate explicit download of the official Real-ESRGAN model.
5. Generate or upscale, then use **Save** to download the resulting PNG. Quit M5Diffusion from its app menu when finished; closing only the browser is not the same as quitting the app.

The app is locally ad-hoc signed, **not Developer ID signed or notarized**. Only if you trust the download and its provenance, follow macOS **System Settings → Privacy & Security → Open Anyway** when offered. Do not disable Gatekeeper or remove quarantine globally. A fresh-machine trusted-download experience has not been certified by Apple.

## Data and runtime

The bundled private runtime is CPython 3.10.21 from the official Astral python-build-standalone release, with the locked application dependencies. The app prepares writable files under:

```text
~/Library/Application Support/M5Diffusion/current/
```

Model and output data are preserved across this launch preparation. Source-checkout runs use their own `models/` and `outputs/` folders. Model conversions and outputs need additional free storage beyond the approximately 1 GB runtime/app; leave several GB available. Startup checks cover architecture, OS, memory, storage, MLX and model readiness.

The local Web UI uses `http://127.0.0.1:7861`. It is a single-user local service, not a publicly exposed or multi-user server. Imported checkpoints/LoRAs and downloaded upscaler weights are separate from the application code license. See [model terms](docs/third-party-licenses.md).

## Troubleshooting

- **Missing model:** use model setup to import an SD1.5 safetensors checkpoint; installing the app alone does not provide diffusion weights.
- **Unsupported or damaged model:** choose a valid SD1.5 checkpoint. SDXL/FLUX and arbitrary pickle checkpoints are not supported by this importer.
- **Port already used:** quit the other process using 7861, or close another running copy of M5Diffusion. Do not terminate an unknown process automatically.
- **Memory/storage shortage:** close other memory-heavy applications, free storage, and retry. Minimum hardware requirements do not guarantee every large workload fits.
- **Cannot write data:** ensure your user can write its Application Support directory. Do not launch with sudo.
- **Web page fails to open:** check the app's startup message, then open the local address above. If the service failed its checks, resolve that error before retrying.

## Source checkout

For developers on Apple Silicon/macOS 26 or later with **Python 3.10** already installed. The dependency lock and packaged runtime were tested with CPython 3.10.21; other Python versions are not validated. Use the repository’s **Code → HTTPS** address to clone it, then enter the new checkout. Alternatively, extract the source archive from Releases.

Run from that checkout:

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pytest -q
.venv/bin/python serve.py
```

Open `http://127.0.0.1:7861` after starting the server. In the Web UI, import a licensed compatible SD1.5 `.safetensors` file through model setup; no weights are bundled. Stop the server with Control-C. GPU-specific tests are opt-in; see [CONTRIBUTING.md](CONTRIBUTING.md).

 The GUI app is the intended path for users without terminal experience. Do not install dependencies into system Python. First model import or official upscaler acquisition may need network access; inference thereafter is local.
