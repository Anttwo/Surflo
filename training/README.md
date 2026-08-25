# Surflo training

Minimal, self-contained training code for Surflo (flow-matching 3D surface
reconstruction). It reproduces the released runs on preprocessed
(cached) DL3DV data and reuses the [`surflo`](../surflo) package for the model.

## Layout

```
training/
  train.py                 # Hydra entry point -> Trainer(**cfg).run()
  trainer.py               # DDP trainer (train loop, EMA, checkpoint, val loss + Chamfer)
  losses.py                # FlowLoss (L2 flow-matching objective)
  data/                    # cached DL3DV datasets + dynamic (multi-view) samplers
  utils/                   # checkpoint / optimizer / gradient-clip / logging / wandb helpers
```

Training and inference share a single config root at
[`../configs/`](../configs) (composed by `../configs/train.yaml`), so the model
architecture is **not** duplicated: `model: surflo` loads the exact
released-checkpoint architecture from
[`../configs/model/surflo.yaml`](../configs/model/surflo.yaml) — the same file
inference uses. Training only flips `model.compile: true`.

## Installation

Set up the environment first — see [`../install/README.md`](../install/README.md);
training uses the same environment as inference. Then, from the repository root:

```bash
pip install -e ".[train]"
```

That adds the training extras (`fvcore`, `iopath`, `wandb`, `ema-pytorch`) on top
of the core dependencies. Everything else, including the `torch-cluster` /
`torch-geometric` wheels used by the validation Chamfer distance, is already
installed by the environment file for your CUDA family.

## Download DL3DV-10K-Meshed

Start by downloading our [modified version of DL3DV-10K](https://huggingface.co/datasets/AntoineGuedon/DL3DV-10K-Meshed).
You can run our dedicated script as shown below; it requires access to the gated repo, so please authenticate first with `hf auth login`:

```bash
python scripts/download_dl3dv_10k_meshed.py --odir dl3dv-10k-meshed                 # everything
python scripts/download_dl3dv_10k_meshed.py --odir dl3dv-10k-meshed --subset 1K 2K  # selected subsets
python scripts/download_dl3dv_10k_meshed.py --odir dl3dv-10k-meshed --workers 4     # throttle if rate-limited
```

## Data preprocessing

Training consumes **preprocessed** scenes only (raw-image loading and the VGGT
forward pass are done offline). Each scene directory holds cached VGGT tokens,
COLMAP/VGGT cameras, the COLMAP↔VGGT alignment, and consolidated GT surface
points:

```
<DATA_DIR>/<scene_id>/
    sample_0000_views_016.pt    # (or sample_0000_views_004.pt for 4 views)
    ...
    surface_data.npz            # shuffled GT surface points/normals (fresh samples per epoch)
```

Generate these caches with [`../scripts/preprocess.py`](../scripts/preprocess.py), from the
repository root, e.g.:

```bash
python scripts/preprocess.py \
    --scene_list training/data/lists/all_scenes.txt \
    --data_dir /path/to/DL3DV-10K-Meshed \
    --output_dir /path/to/DL3DV-10K-preprocessed \
    --min_images 2 \
    --max_images 16 \
    --n_samples 1 \
    --save_vggt_world_points \
    --save_rgb_images
```

OOD validation points at its own preprocessed directory.

### View-count sampling

During training, each iteration draws one view count `N` for the whole batch 
in the interval `img_nums=[inf,sup]`, then loads that `N`'s cache. 
`image_num_weights` (in `configs/data/multiview.yaml`) gives each `N` an
unnormalized probability:

| value | effect |
|---|---|
| `{}` (default) | uniform over `img_nums` |
| `{16: 2}` | `N=16` twice as likely as each other count |
| `{2: 0, ..., 15: 0}` | never draw those counts; train on a single `N=16` |

Any `N` in range that the mapping omits gets `1.0`. A key outside `img_nums` is an
error rather than a silent no-op. The weight is per `N`, not per cache file: extra
samples at a view count add variety, not frequency.

**Coverage is checked at startup**, so a missing cache never surfaces mid-epoch:

- If **no** scene has a cache at a selectable `N`, the dataloader raises — no
  amount of pruning would make that `N` drawable.
- If **some** scenes have it and others do not, the scenes that do not are
  dropped, with a `WARNING` reporting how many and which counts they lacked.
  The sampler draws one `N` per batch but fills it from any scene, so a scene
  missing one selectable count would otherwise fail partway through training.
- Counts with weight `0` are skipped by both checks — zero-weighting an `N` also
  frees you from preprocessing it.

## Training on DL3DV-10K-Meshed

Training uses `torchrun` with 4 GPUs. Set `WANDB_MODE=offline` for logging
without a network connection, and put the repository root on `PYTHONPATH` so the
`surflo` package resolves (imports inside `training/` — `trainer`, `data.*`,
`utils.*` — resolve from the working directory).

Curriculum run on orbitshot scenes with variable view count `N ∈ [2, 16]`:

```bash
cd training
WANDB_MODE=offline PYTHONPATH=.. torchrun --nproc_per_node=4 train.py \
    override=orbit_2to16views \
    data_dir=/path/to/DL3DV-preprocessed \
    ood_data_dir=/path/to/FFM_test_preprocessed
```

Curriculum run on all scenes with fixed `N = 16`:

```bash
cd training
WANDB_MODE=offline PYTHONPATH=.. torchrun --nproc_per_node=4 train.py \
    override=full_16views \
    data_dir=/path/to/DL3DV-preprocessed \
    ood_data_dir=/path/to/FFM_test_preprocessed
```

Checkpoints (model + EMA + optimizer + AMP scaler) are written to
`logs/<exp_name>/ckpts/` after every epoch as `checkpoint.pt`, and every
`checkpoint.save_freq` epochs as `checkpoint_<epoch>.pt`. Saving is atomic with
a one-generation backup (`.bak`).

### Resuming

Resume is automatic: if `checkpoint.pt` (or its `.bak`) exists in
`checkpoint.save_dir`, it is loaded on startup. To resume from an explicit path:

```bash
... train.py override=orbit_2to16views checkpoint.resume_checkpoint_path=/path/to/checkpoint.pt ...
```

### Validation only

```bash
... torchrun --nproc_per_node=4 train.py override=orbit_2to16views mode=val \
    checkpoint.resume_checkpoint_path=/path/to/checkpoint.pt ...
```

## Configuration

Config groups under [`../configs/`](../configs) (composed by
`../configs/train.yaml`):

| Group | Options | Purpose |
|-------|---------|---------|
| `model` | `surflo` | Architecture (the shared inference config, reused verbatim). |
| `data` | `multiview` | Cached DL3DV pipeline. Serves a variable `N` (`img_nums: [2, 16]`) or a fixed one, per the override. |
| `optim` | `default` | AdamW + fvcore LR/WD schedulers + EMA + bf16 AMP + grad clip. |
| `loss` | `flow` | `FlowLoss`. |
| `logging` | `default` | Wandb logging + Chamfer-eval settings. |
| `distributed`, `cuda`, `checkpoint` | `default` | DDP / backend / checkpoint knobs. |
| `override` | `orbit_2to16views`, `full_16views` | The two canonical run presets (mandatory). |

Common overrides: `max_epochs`, `limit_train_batches`, `limit_val_batches`,
`num_workers`, `val_epoch_freq`, `ood_val_epoch_freq`, `seed_value`.

## Scope

This trainer is deliberately minimal: it reproduces the released runs and
nothing else.

**Supported:** DDP, bf16 AMP, gradient accumulation, gradient clipping on
`surface_net`, `where`-driven fvcore LR/WD schedules, EMA, preemption-safe
checkpoint save/resume, scalar and viz logging, and periodic validation loss +
Chamfer distance (in-distribution and OOD).

**Not supported**, should you need them:

- The PyTorch profiler.
- Objectives other than `FlowLoss`: mean-flow (iMF), elastic and
  self-distillation variants are not part of the released method, and
  `use_mean_flow=true` raises.
- Batch-repetition augmentation.
- The raw-image data path. Training reads preprocessed caches only, so image
  loading, augmentation and point-track generation have no place here — run
  [`../scripts/preprocess.py`](../scripts/preprocess.py) first.
- Per-parameter / per-module scheduler filtering (`param_names` /
  `module_cls_names`). Passing either raises rather than being ignored; the
  canonical runs use one default scheduler per option for all parameters.
