# OmniGUI

This repo contains the local evaluation setup for paper OmniGUI: Benchmarking GUI Agents in
Omni-Modal Smartphone Environments.

## 1. Download the dataset

Download the OmniGUI dataset from Hugging Face:

`https://huggingface.co/datasets/XIAOCHENLIN00zz/OmniGUI`

One simple way is:

```bash
huggingface-cli download XIAOCHENLIN00zz/OmniGUI \
  --repo-type dataset \
  --local-dir data
```

After downloading, your local data directory should contain:

```text
data/merged_sorted.json
data/benchmark/
```

## 2. Create the environment

```bash
conda create -n OmniGUI python=3.10
conda activate OmniGUI
```

## 3. Install dependencies

Install the local `google-genai` package:

```bash
pip install google-genai
```

Install `lmms_eval`:

```bash
cd ../lmms_eval
pip install -e .
cd ..
```

## 4. Configure `.env`

The project reads model endpoints and keys from the root `.env` file.

Edit [`\.env`](C:\Users\xpeng\Desktop\OmniGUI_release\.env) and fill in the values you need:

```env
GEMINI3_PRO_API_KEY=
GEMINI3_FLASH_API_KEY=
GEMINI25_PRO_API_KEY=
GEMINI25_FLASH_API_KEY=
GEMINI_BASE_URL=

QWEN3_API_KEY=
QWEN3_BASE_URL=

MINICPM_O_API_URL=
BAICHUAN_OMNI_API_URL=

VITA_PRETRAINED=
```

Model mapping:

- `gemini3_pro` uses `GEMINI3_PRO_API_KEY` and `GEMINI_BASE_URL`
- `gemini3_flash` uses `GEMINI3_FLASH_API_KEY` and `GEMINI_BASE_URL`
- `gemini25_pro` uses `GEMINI25_PRO_API_KEY` and `GEMINI_BASE_URL`
- `gemini25_flash` uses `GEMINI25_FLASH_API_KEY` and `GEMINI_BASE_URL`
- `qwen3_omni` uses `QWEN3_API_KEY` and `QWEN3_BASE_URL`
- `minicpm_o` uses `MINICPM_O_API_URL`
- `baichuan_omni` uses `BAICHUAN_OMNI_API_URL`
- `vita` uses `VITA_PRETRAINED`

Compatibility note:

- If the per-model Gemini key is not set, the code still falls back to `GEMINI_API_KEY`, `GEMINI3_API_KEY`, and `GOOGLE_API_KEY`

## 5. Run evaluation scripts

All evaluation scripts are in [`scripts`](C:\Users\xpeng\Desktop\OmniGUI_release\scripts).

Each script writes results to:

```text
logs/<task_name>/
```

The scripts are organized into three experiment groups:

### 5.1 Main experiments

These are the 8 full-modality main experiments, corresponding to the main results table.

- `scripts/task_v2_gemini3_pro.sh`
- `scripts/task_v2_gemini3_flash.sh`
- `scripts/task_v2_gemini25_pro.sh`
- `scripts/task_v2_gemini25_flash.sh`
- `scripts/task_v2_qwen3_omni.sh`
- `scripts/task_v2_vita.sh`
- `scripts/task_v2_minicpm_o.sh`
- `scripts/task_v2_baichuan_omni.sh`

All of them use the `task_v2_*` configs, which keep the full multimodal input setting.

### 5.2 Audio & Video ablation experiments

These scripts are the ablation experiments for the three reported models:

- Gemini 3 Pro
- Gemini 2.5 Flash
- Qwen3-Omni

Each model has three ablation settings:

- `no_audio`: remove audio input
- `no_video`: remove video input
- `no_av`: remove both audio and video, keeping image-only input

Scripts:

- `scripts/task_abl1v2_no_audio_pro.sh`
- `scripts/task_abl2v2_no_video_pro.sh`
- `scripts/task_abl3v2_no_av_pro.sh`
- `scripts/task_abl1v2_no_audio_25flash.sh`
- `scripts/task_abl2v2_no_video_25flash.sh`
- `scripts/task_abl3v2_no_av_25flash.sh`
- `scripts/task_abl1v2_no_audio_qwen3.sh`
- `scripts/task_abl2v2_no_video_qwen3.sh`
- `scripts/task_abl3v2_no_av_qwen3.sh`

### 5.3 TTS audio replacement experiments

These scripts replace the original text instruction with TTS-generated audio instruction.

Scripts:

- `scripts/task_tts_gemini3_pro.sh`
- `scripts/task_tts_qwen3_omni.sh`

These correspond to the TTS ablation setting in the paper-style comparison table.

For example:

```bash
bash scripts/task_v2_qwen3_omni.sh
```

This script runs:

```bash
python -m lmms_eval \
  --model qwen3_omni \
  --tasks task_v2_qwen3_omni \
  --batch_size 1 \
  --log_samples \
  --output_path logs/task_v2_qwen3_omni
```

## Example script

Example script file:

[`scripts/task_v2_qwen3_omni.sh`](C:\Users\xpeng\Desktop\OmniGUI_release\scripts\task_v2_qwen3_omni.sh)

```bash
#!/bin/bash
set -euo pipefail

TASK_NAME="task_v2_qwen3_omni"
MODEL_NAME="qwen3_omni"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/${TASK_NAME}"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

echo "Launching ${TASK_NAME}..."
nohup python -m lmms_eval \
  --model "${MODEL_NAME}" \
  --tasks "${TASK_NAME}" \
  --batch_size 1 \
  --log_samples \
  --output_path "logs/${TASK_NAME}" \
  > "${LOG_DIR}/run.log" 2>&1 &
echo "  PID: $!"
echo "Logs: ${LOG_DIR}/run.log"
```
