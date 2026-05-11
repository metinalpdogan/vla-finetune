# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo flavor

This is a fork of OpenVLA carrying the **MiniVLA** additions (Qwen2.5 0.5B backbone, multi-image / wrist / history support, Residual VQ action chunking) and a custom **xArm** finetune setup. The README is the upstream OpenVLA README — when guidance there conflicts with what's actually in `prismatic/conf/vla.py` or `prismatic/vla/action_tokenizer.py`, trust the code.

## Common commands

```bash
# Lint / format (configured in pyproject.toml: black + ruff, line length 121)
make check         # check only
make autoformat    # apply fixes (also runs via pre-commit)
make clean         # remove __pycache__ / *.pyc

# VLA finetuning (LoRA, HF AutoClasses path)
torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
    --vla_path <hf_id_or_path> --data_root_dir <RLDS root> --dataset_name <name> \
    --run_root_dir <out> --adapter_tmp_dir <tmp> --lora_rank 32 --batch_size 16

# VLA full finetune / pretraining (Prismatic FSDP path)
torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/train.py \
    --vla.type <vla_id from VLARegistry> --data_root_dir <root> --run_root_dir <out>

# Pretrain a Residual VQ tokenizer (writes to vq/<exp_name>/)
python vla-scripts/pretrain_vq.py --data_dir <RLDS root> --data_mix <mix> \
    --action_dim 7 --future_action_horizon 7 --vqvae_n_embed 128

# Base Prismatic VLM pretrain (rarely used here; needed for new VLM backbones)
torchrun --standalone --nnodes 1 --nproc-per-node $K scripts/pretrain.py \
    --model.type prism-qwen25-extra-dinosiglip-224px+0_5b

# Convert a Prismatic-format checkpoint to HF AutoClasses
python vla-scripts/extern/convert_openvla_weights_to_hf.py \
    --openvla_model_path_or_id <run_dir> --output_hf_model_local_path <out>
# Note: that script expects checkpoints/latest-checkpoint.pt — symlink it before running.
# Note: this conversion is NOT supported for MiniVLA / VQ / multi-image models per README.

# Eval
python experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint <ckpt> \
    --task_suite_name libero_{spatial,object,goal,10} --center_crop True
python experiments/robot/simpler/run_simpler_eval.py    # simpler libero-style eval
python experiments/robot/bridge/run_bridgev2_eval.py    # WidowX (needs widowx_envs + edgeml + docker)
```

There is **no test suite**. `scripts/test_load_data.py` is a dataloader smoke test, not pytest. Don't try to add a test runner unless asked.

## Two training paths — these are not interchangeable

There are **two parallel implementations** of OpenVLA loading/training in this repo and confusing them is the most common source of breakage:

1. **HF AutoClasses path** (`vla-scripts/finetune.py`): loads via `AutoModelForVision2Seq.from_pretrained(...)` using the wrappers in `prismatic/extern/hf/` (`OpenVLAConfig`, `OpenVLAForActionPrediction`, `PrismaticImageProcessor`, `PrismaticProcessor`). Uses PEFT LoRA + DDP. Consumes HF-format checkpoints (e.g. `openvla/openvla-7b`).
2. **Prismatic FSDP path** (`vla-scripts/train.py`, `scripts/pretrain.py`): loads via `prismatic.models.load*`, distributes with FSDP through `prismatic/training/strategies/fsdp.py`. Consumes Prismatic-format `.pt` checkpoints (e.g. `openvla/openvla-7b-prismatic`). All MiniVLA / multi-image / VQ training happens here.

Conversion goes Prismatic → HF only, via the `convert_openvla_weights_to_hf.py` script, and per the README does **not** currently support MiniVLA, VQ, or multi-image variants.

## High-level architecture

```
prismatic/
├── conf/
│   ├── vla.py           # VLAConfig dataclass + VLARegistry — every training run is a registered subclass here.
│   ├── models.py        # Base Prismatic VLM configs (vision+LLM backbone combos, e.g. prism-qwen25-extra-dinosiglip-224px+0_5b)
│   └── datasets.py      # VLM (not VLA) pretraining dataset configs
├── models/
│   ├── backbones/{vision,llm}/   # SigLIP, DINOv2, fused DinoSigLIP; Llama-2, Vicuna, Qwen2.5, …
│   ├── vlms/prismatic.py         # Base PrismaticVLM (vision encoder → projector → LLM)
│   ├── vlas/openvla.py           # OpenVLA = PrismaticVLM + action head plumbing
│   ├── load.py / materialize.py / registry.py   # Factory wiring
├── vla/
│   ├── action_tokenizer.py       # ActionTokenizer (last-256 vocab bins) and VQActionTokenizer; ACTION_TOKENIZERS dict at bottom
│   ├── datasets/                 # RLDS pipeline (TFDS-backed) — datasets.py wraps it for PyTorch
│   │   └── rlds/oxe/             # Per-dataset configs.py, transforms.py, mixtures.py — these three files are
│   │                             #   what you edit to add a new robot dataset (see below).
│   ├── action_dataset_materialize.py   # Used by pretrain_vq.py to stream actions out of RLDS for VQ training
│   └── materialize.py
├── training/
│   ├── strategies/{ddp,fsdp}.py  # Two distributed strategies; FSDP is the default for VLAs
│   └── metrics.py
├── extern/hf/                    # HF AutoClasses wrappers (configuration_/processing_/modeling_prismatic.py)
└── overwatch/, preprocessing/, util/

vla-scripts/   # train.py (FSDP), finetune.py (LoRA/HF), pretrain_vq.py, deploy.py, extern/convert_openvla_weights_to_hf.py
scripts/       # Holdover from base prismatic-vlms repo: VLM (not VLA) training utilities
experiments/robot/   # Eval harnesses: bridge/, libero/, simpler/, xarm/ — each is its own embodiment-specific eval
vq/            # Pretrained Residual VQ tokenizer checkpoints; paths in ACTION_TOKENIZERS resolve relative to this dir
```

### Adding a new finetuning dataset (full-finetune / Prismatic path)

This is the standard four-file change — edits to anything less than the full set will fail at import or dataloader time:

1. `prismatic/vla/datasets/rlds/oxe/configs.py` — add observation/action space config to `OXE_DATASET_CONFIGS`.
2. `prismatic/vla/datasets/rlds/oxe/transforms.py` — add a standardization transform and register it in `OXE_STANDARDIZATION_TRANSFORMS`.
3. `prismatic/vla/datasets/rlds/oxe/mixtures.py` — add a mixture entry in `OXE_NAMED_MIXTURES`.
4. `prismatic/conf/vla.py` — add an `Exp_*` subclass of `VLAConfig` (or another `Exp_*`), give it a unique `vla_id`, then register it in `VLARegistry` at the bottom. The loop after the registry auto-registers it with draccus. Reference the new `vla_id` via `--vla.type`.

### Action tokenizer / Residual VQ

`prismatic/vla/action_tokenizer.py` provides two tokenizer classes plus an `ACTION_TOKENIZERS` dict keyed by string name; `--vla.action_tokenizer <name>` selects which one to use.

- `ActionTokenizer` (default for `siglip-224px` / Llama configs): bins continuous actions into the **last 256 vocab tokens**.
- `extra_action_tokenizer` (default for **all Qwen2.5 / MiniVLA configs**): uses the **256 extra `<extra_i>` tokens** that the `prism-qwen25-extra-*` VLM was trained with. Selecting plain `action_tokenizer` for a Qwen backbone is a silent footgun — it'll bin into unrelated vocab tokens.
- `VQActionTokenizer` (`*_vq_*` entries): wraps a pretrained Residual VQ at `vq/<run_name>/` and emits `T` discrete codeword bins per `(H × A)` action chunk. Pass `use_extra=True` for Qwen backbones. VQ paths in the dict resolve via `_OPENVLA_REPO_ROOT = Path(__file__).resolve().parents[2]`, so they work from any cwd as long as the `vq/` directory is at the repo root.

To register a new VQ tokenizer, add an entry to `ACTION_TOKENIZERS` pointing at your `vq/<run_name>/` checkpoint dir, then pass that name via `--vla.action_tokenizer`.

### Multi-image / wrist / history

The vision encoder runs once per raw image and the resulting token sequences are **concatenated with no separator tokens** before the LLM. Therefore:

- `vla.image_sequence_len` must equal the exact number of raw images at both train and inference time.
- For wrist-camera variants set `vla.use_wrist_image=True` (see `prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-libero-90`).
- At deploy/inference the caller is responsible for passing images **in the same order** as training — there is no positional tagging in the token stream.

## Pinned dependencies — don't bump casually

`pyproject.toml` pins `torch==2.2.0`, `torchvision==0.17.0`, `transformers==4.40.1`, `tokenizers==0.19.1`, `timm==0.9.10`, `tensorflow==2.15.0`, `tensorflow_datasets==4.9.3`, `peft==0.11.1`, `sentencepiece==0.1.99`, `draccus==0.8.0`, `dlimp` from `git+https://github.com/moojink/dlimp_openvla`. flash-attn 2.5.5 is installed separately (see README install section). Newer versions of `transformers`/`timm`/`tokenizers` have caused regressions; published LIBERO eval numbers are tied to this exact stack on A100s.

## Conventions worth knowing

- `.hf_token` at repo root (gitignored) holds the HuggingFace token; `train.py` reads it via the `hf_token` config field.
- `wandb_project` defaults to `"prismatic"` (or `"prismatic-vq-vla"` for VQ). Override per run.
- `runs/`, `checkpoints/` (at repo root), `wandb/`, `rollouts/`, and `experiments/logs/` are gitignored — drop large artifacts there. The leading `/` on `/runs/` and `/checkpoints/` in `.gitignore` is intentional so that nested `vq/<run>/checkpoints/model.pt` checkpoints (which *are* tracked) aren't excluded.
- BridgeData V2 must be on disk under the directory name `bridge_orig`, not `bridge_dataset` — the OXE configs reference this name and the upstream OXE bridge data is stale.
