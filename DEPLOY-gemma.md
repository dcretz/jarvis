# Deploying `gemma.py` on another PC

`gemma.py` transcribes/translates an audio file (`input.wav`) to text using
Google's **gemma-4-E4B-it** model — a small (4.5B effective / 8B total) multimodal
model that handles audio directly (no separate encoder), via the Hugging Face
`transformers` `any-to-any` pipeline, quantized to 4-bit so it fits on a 6 GB GPU.

---

## 1. Hardware requirements

- **NVIDIA GPU with >= 6 GB VRAM** and a recent driver (>= ~550; tested on 591.86).
  - In **4-bit** the model uses **~5–6 GB VRAM** and runs fully on the GPU (no
    CPU/disk offload, RAM stays free). Tested on a **GTX 1660 SUPER (6 GB)**.
- ~16 GB free disk: the model downloads in full bf16 (~16 GB) the first time, then
  is quantized to 4-bit in memory at load. (Download is one-time; cached after.)
- **30 seconds max audio** per clip — this is a model limit for gemma-4 E2B/E4B.

## 2. Software prerequisites

- **Python 3.11** (not 3.12+ — on 3.14 the install fails; that was the first error here).
- **NVIDIA driver supporting CUDA 12.4** (the torch build below is cu124).
- `git` (transformers is installed from source — see below).

## 3. Setup steps

```powershell
# from the project folder, create the Python 3.11 venv
py -3.11 -m venv venv311
.\venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip

# STEP 1 — PyTorch >= 2.6 (REQUIRED: gemma-4's audio tower needs torch>=2.6), CUDA 12.4 build
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# STEP 2 — the rest (transformers comes from a pinned git commit)
pip install -r requirements-gemma.txt
```

> Two non-obvious traps:
> 1. The `any-to-any` pipeline is only in the transformers **dev** branch — a plain
>    `pip install transformers` will NOT work. `requirements-gemma.txt` pins the commit.
> 2. **torch must be >= 2.6** or you get
>    `ValueError: Using or_mask_function ... require torch>=2.6` during audio processing.

## 4. Hugging Face access (the model is GATED)

`google/gemma-4-E4B-it` requires accepting Google's license and logging in:

1. Create an account at https://huggingface.co
2. On the model page, click **"Agree and access repository"** (accept the Gemma license).
3. Create a read token: https://huggingface.co/settings/tokens
4. Log in on the deploy machine:
   ```powershell
   huggingface-cli login   # paste the token
   ```

First run downloads ~16 GB into `%USERPROFILE%\.cache\huggingface\hub`. Cached after.

## 5. Run it

```powershell
.\venv311\Scripts\Activate.ps1
python gemma.py
```

- Put your audio as **`input.wav`** in the project folder (resolved next to
  `gemma.py`, so cwd doesn't matter). Any sample rate / mono/stereo is fine.
- **Keep clips <= 30 seconds.**
- Output prints clearly in the terminal AND is saved to **`transcription.txt`**.

## 6. Verify the GPU is being used

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.6.0+cu124 True
# While running, `nvidia-smi` should show ~5-6 GB VRAM used and RAM staying low.
```

---

## Known notes / gotchas

- If `nvidia-smi` shows low VRAM but RAM/disk maxed out → the model is offloading
  (wrong model or quantization off). With E4B 4-bit it should fit fully in VRAM.
- The deprecation warnings about `generation_config` / `max_new_tokens` are harmless.
- Want a **smaller download** instead? Use a pre-quantized GGUF via Ollama
  (`ollama run gemma-4-e4b` style) — ~4-5 GB download, but that's a different API,
  not this `transformers` script.

## Exact versions known to work (this machine)

| Package      | Version             |
|--------------|---------------------|
| Python       | 3.11.9              |
| torch        | 2.6.0+cu124         |
| torchaudio   | 2.6.0+cu124         |
| torchvision  | 0.21.0+cu124        |
| transformers | git @ 1b7bc25 (dev) |
| bitsandbytes | >= 0.43             |
| accelerate   | 1.13.0              |
| soundfile    | 0.14.0              |
| librosa      | 0.11.0              |
| numpy        | 2.4.6               |
| Model        | google/gemma-4-E4B-it (4-bit) |
| NVIDIA driver| 591.86 (CUDA 12.4)  |
