# vla-finetune

A fork of [**OpenVLA**](https://github.com/openvla/openvla) with the [**MiniVLA**](https://huggingface.co/collections/Stanford-ILIAD/minivla-675a2a9aca369ff3a6c04e33) additions and an end-to-end **xArm real-robot data collection + finetuning** workflow. Built on top of [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms).

What's in this repo:

- **Train MiniVLA** (Qwen2.5 0.5B backbone + DinoSigLIP vision) on your own xArm demos, on a single 24 GB GPU.
- **Collect real-robot demos** via [ril-env](https://github.com/UCLA-Robot-Intelligence-Lab/ril-env) (xArm + Intel RealSense + 3Dconnexion SpaceMouse) and convert them to RLDS in two scripted stages.
- **Multi-image, wrist-camera, action-chunking** support from upstream MiniVLA.

For real-robot data collection, see **[`data_collection/README.md`](./data_collection/README.md)**.

---

## Table of contents

1. [What is MiniVLA?](#what-is-minivla)
2. [Repository layout](#repository-layout)
3. [Two environments, one repo](#two-environments-one-repo)
4. [End-to-end workflow](#end-to-end-workflow)
5. [Training paths](#training-paths)
6. [Adding a new task / dataset](#adding-a-new-task--dataset)
7. [Evaluating a trained policy](#evaluating-a-trained-policy)
8. [Pinned dependencies](#pinned-dependencies)
9. [References](#references)

---

## What is MiniVLA?

A smaller (≈0.5B-parameter) Vision-Language-Action model built on top of the same Prismatic VLM scaffolding as OpenVLA-7B. The key differences:

| | OpenVLA-7B | MiniVLA 0.5B |
|---|---|---|
| LLM backbone | Llama-2 7B | Qwen2.5 0.5B |
| Vision backbone | DinoSigLIP 224px | DinoSigLIP 224px |
| Action tokens | Last 256 vocab tokens | 256 **extra** `<extra_i>` tokens added during VLM pretrain |
| Multi-image / wrist / history | not supported (HF AutoClass path) | **supported** |
| Action chunking (Residual VQ) | not supported | **supported** |
| GPU memory for finetune | 80 GB | **24 GB fits** |
| Pretrained checkpoints | [openvla/openvla-7b](https://huggingface.co/openvla/openvla-7b) | [Stanford-ILIAD/minivla-*](https://huggingface.co/collections/Stanford-ILIAD/minivla-675a2a9aca369ff3a6c04e33) |

The **`extra_action_tokenizer`** is mandatory for Qwen2.5 backbones — selecting plain `action_tokenizer` is a silent footgun. All MiniVLA configs in `prismatic/conf/vla.py` use the extra tokens by default.

The relevant VLA configs in `prismatic/conf/vla.py`:

| `vla_id` | Cameras at inference | Use case |
|---|---|---|
| `prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90` | agentview only | Single-camera setup |
| `prism-qwen25-dinosiglip-224px-t2+0_5b+mx-libero-90` | agentview + 2-step history | Temporal context, no wrist |
| `prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-libero-90` | agentview + wrist | **Recommended for xArm** |
| `prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-xarm-pick-red-block` | agentview + wrist | Pre-configured for this fork's xArm pipeline |

---

## Repository layout

```
.
├── data_collection/        # ← NEW: xArm teleop, zarr→HDF5, RLDS builder
│   ├── collect_demos.py    # Stage 1: SpaceMouse teleop → zarr replay buffer + MP4s
│   ├── zarr_to_libero_hdf5.py  # Stage 2: zarr → LIBERO-format HDF5
│   ├── rlds_builder/       # Stage 3: TFDS DatasetBuilder for HDF5 → RLDS
│   ├── home.py             # Convenience: home the xArm
│   ├── pyproject.toml      # Lightweight collection venv (no torch)
│   └── README.md           # Detailed collection guide
│
├── ril-env/                # SUBMODULE: real-robot stack (xArm, RealSense, SpaceMouse)
│
├── prismatic/              # Core VLA + VLM library
│   ├── conf/
│   │   ├── vla.py          # VLAConfig + VLARegistry — register new train runs here
│   │   ├── models.py       # Base VLM configs (e.g. prism-qwen25-extra-dinosiglip-224px+0_5b)
│   │   └── datasets.py     # VLM (not VLA) pretrain dataset configs
│   ├── models/
│   │   ├── backbones/      # Vision (SigLIP/DINOv2/fused) + LLM (Llama/Vicuna/Qwen)
│   │   ├── vlms/           # PrismaticVLM
│   │   └── vlas/           # OpenVLA = PrismaticVLM + action head
│   ├── vla/
│   │   ├── action_tokenizer.py  # ActionTokenizer + VQActionTokenizer + ACTION_TOKENIZERS dict
│   │   └── datasets/rlds/oxe/   # Per-dataset configs.py, transforms.py, mixtures.py
│   └── training/strategies/{ddp,fsdp}.py
│
├── vla-scripts/
│   ├── train.py            # Full-finetune / pretraining (Prismatic FSDP path)
│   ├── finetune.py         # LoRA finetune (HF AutoClasses path)
│   ├── pretrain_vq.py      # Pretrain a Residual VQ action tokenizer
│   ├── deploy.py           # Lightweight REST server for inference
│   └── extern/convert_openvla_weights_to_hf.py
│
├── experiments/robot/      # Eval harnesses
│   ├── libero/             # LIBERO sim benchmark
│   ├── bridge/             # BridgeData V2 WidowX
│   ├── simpler/            # Simpler libero-style eval
│   └── xarm/               # xArm scaffold
│
├── vq/                     # Pretrained Residual VQ tokenizer checkpoints (tracked)
├── pyproject.toml          # Training environment (torch + tf + transformers, heavy)
└── runs/, checkpoints/, data/, wandb/  # gitignored output dirs
```

---

## Two environments, one repo

The repo has **two distinct uv-managed environments** for two different workflows. They don't share dependencies, and they can live on different machines if you want.

| | Data collection | Training |
|---|---|---|
| Location | `data_collection/` | repo root |
| Pin | `data_collection/pyproject.toml` | `pyproject.toml` |
| Heaviness | ~200 MB | ~5 GB |
| Includes torch / tf? | **No** | Yes (torch 2.2, tf 2.15, transformers 4.40, flash-attn 2.5.5) |
| Runs on | The robot host | A GPU box (RTX 4090 24 GB works for MiniVLA 0.5B) |
| Setup | `cd data_collection && uv sync` | `cd <repo root> && uv sync && pip install flash-attn==2.5.5 --no-build-isolation` |
| Activate | `source data_collection/.venv/bin/activate` | `source .venv/bin/activate` |

---

## End-to-end workflow

### 0. Clone (with submodule)

```bash
git clone --recurse-submodules <repo-url>
cd vla-finetune
# If you forgot --recurse-submodules:
git submodule update --init --recursive
```

### 1. Collect demos on the xArm

Follow **[`data_collection/README.md`](./data_collection/README.md)**. Short version:

```bash
cd data_collection
uv sync && source .venv/bin/activate
# Edit TASK_NAME and LANGUAGE at the top of collect_demos.py
python collect_demos.py        # use SpaceMouse + 'c'/'s'/'q'/backspace keys
python zarr_to_libero_hdf5.py  # converts to LIBERO HDF5 at recordings/_libero_hdf5/
```

Aim for **~50–100 demos per task** at varied block / target positions, control rate 30 Hz (collected) → 10 Hz (training).

### 2. Convert HDF5 → RLDS

```bash
cd data_collection/rlds_builder
uv sync && source .venv/bin/activate
tfds build XArmPickRedBlock --data_dir "$HOME/edward/vla-finetune/data/rlds"
```

Output: `data/rlds/x_arm_pick_red_block/1.0.0/`.

> If you're collecting a **new** task with a different `TASK_NAME`, see [Adding a new task / dataset](#adding-a-new-task--dataset) below — you'll need to copy and rename the builder, then register the dataset in four files.

### 3. Install the training environment + base checkpoint

```bash
cd <repo root>                # back to vla-finetune/
uv sync                        # ~5 GB; takes a few minutes
source .venv/bin/activate
pip install "flash-attn==2.5.5" --no-build-isolation
huggingface-cli download Stanford-ILIAD/minivla-libero90-prismatic \
    --local-dir checkpoints/minivla-libero90-prismatic
```

### 4. Train

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py \
  --vla.type "prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-xarm-pick-red-block" \
  --pretrained_checkpoint "checkpoints/minivla-libero90-prismatic/checkpoints/latest-checkpoint.pt" \
  --data_root_dir "$HOME/edward/vla-finetune/data/rlds" \
  --run_root_dir runs/ \
  --is_resume False \
  --image_aug True \
  --save_interval 1000 \
  --wandb_project "xarm-minivla"
```

Notes:
- `--data_root_dir` is the **parent** directory of `x_arm_pick_red_block/`, not the dataset itself (TFDS convention).
- `--is_resume False` because we're starting from a pretrained MiniVLA, not resuming a paused run.
- The `vla_id` `prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-xarm-pick-red-block` is already pre-tuned for a single 24 GB GPU (batch 8, FSDP, 50 epochs).
- If OOM: drop `per_device_batch_size` in the `Exp_*` class in `prismatic/conf/vla.py` from 8 to 4 or 2.

### 5. Eval

A barebones xArm eval scaffold lives at `experiments/robot/xarm/`. The LIBERO sim harness is more complete and is a useful reference; see `experiments/robot/libero/run_libero_eval.py`. For a deployed REST server, `vla-scripts/deploy.py` wraps a trained checkpoint behind HTTP.

---

## Training paths

This repo has **two parallel implementations** of OpenVLA training. They are not interchangeable; confusing them is the most common source of breakage.

### 1. Prismatic FSDP path — `vla-scripts/train.py`

- Loads via `prismatic.models.load*` (native PyTorch).
- Distributes with PyTorch FSDP (`prismatic/training/strategies/fsdp.py`).
- Consumes Prismatic-format `.pt` checkpoints (e.g. `openvla/openvla-7b-prismatic`).
- **All MiniVLA / multi-image / VQ training happens here.**
- This is what you use for the xArm pipeline above.

### 2. HF AutoClasses path — `vla-scripts/finetune.py`

- Loads via `AutoModelForVision2Seq.from_pretrained(...)` using wrappers in `prismatic/extern/hf/`.
- LoRA + DDP (one full replica per GPU; trains the LoRA adapters).
- Consumes HF-format checkpoints (e.g. `openvla/openvla-7b`).
- Does **not** support MiniVLA / multi-image / VQ models.

Conversion goes Prismatic → HF only (via `vla-scripts/extern/convert_openvla_weights_to_hf.py`), and **does not currently support** MiniVLA / VQ / multi-image variants per the conversion script's docstring.

---

## Adding a new task / dataset

Each new dataset is a **four-file** edit. Doing less than the full set will fail at import or dataloader time.

1. **`data_collection/rlds_builder/`** — copy `XArmPickRedBlock/` to `YourNewTask/`, edit:
   - the class name (`class YourNewTask(...)`)
   - `DATA_HDF5_GLOB` to point at your task's HDF5

2. **`prismatic/vla/datasets/rlds/oxe/configs.py`** — add an `OXE_DATASET_CONFIGS` entry mirroring `x_arm_pick_red_block`:
   ```python
   "your_new_task": {
       "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
       "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
       "state_obs_keys": ["EEF_state", None, "gripper_state"],
       "state_encoding": StateEncoding.POS_EULER,
       "action_encoding": ActionEncoding.EEF_POS,
   },
   ```

3. **`prismatic/vla/datasets/rlds/oxe/transforms.py`** — add to `OXE_STANDARDIZATION_TRANSFORMS`:
   ```python
   "your_new_task": libero_dataset_transform,
   ```

4. **`prismatic/vla/datasets/rlds/oxe/mixtures.py`** — add to `OXE_NAMED_MIXTURES`:
   ```python
   "your_new_task": [("your_new_task", 1.0)],
   ```

5. **`prismatic/conf/vla.py`** — clone the `Exp_Qwen25_DinoSigLIP_224px_wrist_0_5B_XArm_PickRedBlock` class, give it a new `vla_id` and `data_mix`, register in `VLARegistry`.

The `x_arm_pick_red_block` entries in all five files are working reference implementations — copy and modify.

---

## Evaluating a trained policy

After training, the run dir at `runs/<vla_id>+stage-finetune+x<seed>/` contains checkpoint `.pt` files. Two ways to use them:

**On the xArm (real robot):** roll your own using `vla-scripts/deploy.py` (REST server) + ril-env on the robot side. The scaffold under `experiments/robot/xarm/` is a starting point.

**On LIBERO sim (for sanity-check on pretrained MiniVLA + simulation):**
```bash
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint runs/<run_dir>/checkpoints/latest-checkpoint.pt \
    --task_suite_name libero_90 \
    --center_crop True
```

---

## Pinned dependencies

Don't bump casually. `pyproject.toml` pins:

- `torch==2.2.0`, `torchvision==0.17.0`
- `transformers==4.40.1`, `tokenizers==0.19.1`, `timm==0.9.10`
- `tensorflow==2.15.0`, `tensorflow_datasets==4.9.3`
- `peft==0.11.1`, `sentencepiece==0.1.99`, `draccus==0.8.0`
- `dlimp` from `git+https://github.com/moojink/dlimp_openvla`
- `flash-attn==2.5.5` (installed separately, see Quickstart)

Newer versions of `transformers` / `timm` / `tokenizers` have caused training regressions. The published LIBERO eval numbers are tied to this exact stack on A100s.

---

## References

- [OpenVLA paper](https://arxiv.org/abs/2406.09246) (Kim et al., 2024)
- [Original OpenVLA repo](https://github.com/openvla/openvla)
- [MiniVLA HuggingFace collection](https://huggingface.co/collections/Stanford-ILIAD/minivla-675a2a9aca369ff3a6c04e33)
- [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms)
- [ril-env](https://github.com/UCLA-Robot-Intelligence-Lab/ril-env) — submodule
- [rlds_dataset_builder](https://github.com/moojink/rlds_dataset_builder) — pattern for HDF5 → RLDS converters

---

## License

MIT, inherited from upstream OpenVLA. Note that pretrained model weights may inherit additional restrictions from their base LLMs (Llama-2 license for OpenVLA-7B; Qwen2.5 license for MiniVLA).
