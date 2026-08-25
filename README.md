<div align="center">
<h1> 
  Surflo: Consistent 3D Surface Flow Model <br>with Global State
</h1>

<font>
<a href="https://anttwo.github.io/" style="font-size:100%;">Antoine Guédon*<sup>1</sup></a>&emsp;
<a href="https://scholar.google.com/citations?user=ma_HN-YAAAAJ&hl=ja" style="font-size:100%;">Shu Nakamura*<sup>2</sup></a>&emsp;
<a href="https://nicolas-dufour.github.io/" style="font-size:100%;">Nicolas Dufour*<sup>3</sup></a>&emsp;
<a href="https://jiahuilei.com/" style="font-size:100%;">Jiahui Lei<sup>4</sup></a>&emsp;<br>
<a href="https://vision.ist.i.kyoto-u.ac.jp/" style="font-size:100%;">Ko Nishino<sup>2</sup></a>&emsp;
<a href="https://people.eecs.berkeley.edu/~kanazawa/" style="font-size:100%;">Angjoo Kanazawa<sup>4</sup></a>&emsp;
</font>
<br>

<font>
<sup>1</sup>LIX, Ecole polytechnique&emsp;
<sup>2</sup>Kyoto University&emsp;
<sup>3</sup>Kyutai&emsp;
<sup>4</sup>UC Berkeley&emsp;
</font>
<br>

<font>
*Authors contributed equally to the paper.
</font>

| <a href="https://anttwo.github.io/surflo/">Webpage</a> | <a href="https://arxiv.org/abs/2606.13644">arXiv</a> | <a href="https://www.youtube.com/watch?v=uHdBCm2jhKA">Presentation video</a> | <a href="https://huggingface.co/AntoineGuedon/Surflo-v0">Weights</a> | <a href="https://huggingface.co/datasets/AntoineGuedon/DL3DV-10K-Meshed">Data</a> |


</div>

![Pipeline](media/surflo_pipeline_github.png)

<div align="justify">
<em>
<b>Surflo turns a handful of unposed RGB views into a detailed 3D surface.</b> From a variable number N of input views (N can be 2, 10, 30 or even 80), Surflo encodes the scene into a fixed-size latent and decodes it via flow matching into an arbitrary number of oriented surface points, yielding a clean mesh. Surflo exceeds pointmap-feed-forward models in quality, while running an order of magnitude faster than state-of-the-art optimization-based methods like Gaussian Wrapping, which require hundreds of views.
</em>
</div><br>

---

> **Image orientation.** The model is trained on landscape images only (width >
> height), so at inference **portrait images are automatically rotated 90° to
> landscape** (a warning is logged for each). This applies to every inference
> path (CLI, demo, Python API); training preprocessing is left untouched. One
> consequence: since the world frame is the first camera's frame, when the first
> view was portrait the reconstruction — and any saved point cloud / mesh — comes
> out in that rotated frame.

## Installation

<details>
<summary>Click to expand</summary>
<br>

Surflo is installed from a source checkout. It needs compiled
CUDA extensions built against your torch, so there is no `pip install surflo`.

Start by cloning the repo with `--recursive`:
```bash
git clone git@github.com:Anttwo/Surflo.git --recursive 
```

We provide ready-made installation files for CUDA **11.8**, **12.1** and **12.4** in
[`install/`](install/).
You can find instructions below; please see [`install/README.md`](install/README.md) 
for more details (extras, HPC / headless builds, troubleshooting).

We propose two options, either **conda** or **uv**. **We strongly recommend to use conda** as it allows for very easily building a 100% self-contained environment with any specified version of CUDA. Still, if your system CUDA is either 11.8, 12.1 or 12.4, you can also install with uv.

### **Option A — conda**, 100% self-contained: 
It brings its own `nvcc` and compiler (~55 MB), so nothing on your system 
has to match. Swap `cu118` for your family (`cu121` or `cu124`):

```bash
# Create and activate env
# Pick any CUDA version: cu118, cu121 or cu124
conda env create -f install/environment-cu118.yml
conda activate surflo-cu118

# Install dependencies with self-contained CUDA
pip install -e ".[demo,texture,train]"

# Write the CUDA hook for the environment
bash install/activate_cuda.sh && conda deactivate && conda activate surflo-cu118

# Compile CUDA extensions
# You might need to set TORCH_CUDA_ARCH_LIST first
bash install/build_extensions.sh --all

# Report what works
python install/verify_install.py --check-isolation
```

### **Option B — uv / pip**, lighter: 
You supply a CUDA toolkit matching your family (e.g. an HPC `module load`). 
Check with `nvcc --version` first.

```bash
# Create env
uv venv --python 3.10 && source .venv/bin/activate

# Install dependencies
# Pick the CUDA version matching your system: 11.8, 12.1 or 12.4
uv pip install --index-strategy unsafe-best-match -r install/requirements-cu118.txt
uv pip install -e ".[demo,texture,train]"

# Compile CUDA extensions
# You might need to set TORCH_CUDA_ARCH_LIST first
bash install/build_extensions.sh --all

# Report what works
python install/verify_install.py --check-isolation
```

(`--index-strategy unsafe-best-match` is required with uv; plain `pip -r` needs
no flag.)

Option B needs two OS packages for textured mesh export, EGL/GL development headers, which CUDA does not provide and pip cannot install:
```bash
sudo apt install libegl1-mesa-dev libgl1-mesa-dev      # Debian/Ubuntu
sudo dnf install mesa-libEGL-devel mesa-libGL-devel    # RHEL/Fedora
```

</details>

## Model weights

<details>
<summary>Click to expand</summary>
<br>

The Surflo checkpoint can be downloaded from 
[`AntoineGuedon/Surflo-v0`](https://huggingface.co/AntoineGuedon/Surflo-v0):

```bash
hf download AntoineGuedon/Surflo-v0 surflo_v0.pt --local-dir checkpoints/
```

`hf` ships with `huggingface_hub`, which is already a dependency. On older
versions of the package the command is `huggingface-cli download`.

Then point any entry point at the downloaded file, in this case `./checkpoints/surflo_v0.pt`.

You can also let Python fetch and cache it for you:

```python
from huggingface_hub import hf_hub_download
from surflo import Surflo

ckpt = hf_hub_download("AntoineGuedon/Surflo-v0", "surflo_v0.pt")
surflo = Surflo.from_checkpoint(ckpt, device="cuda")
```

</details>

## Demo

<details>
<summary>Click to expand</summary>
<br>

You can try Surflo with our interactive Gradio UI, which wraps the full pipeline (load images → encode → reconstruct points → convert to mesh):

```bash
python examples/gradio_demo.py --ckpt /path/to/surflo_v0.pt
# --host / --port to change the address, --share for a public link,
# --dev to load images from a server-side directory (useful over SSH)
```

Using the interface:

1. **Select images** — upload a set of views (or point to a server-side folder in `--dev` mode).
2. **Pick a mode and preset** — `guided` with a guidance preset (`minimal` / `short` / `default` / `default_highres` / `long` / …), or `plain` for deactivating guidance.
3. **Reconstruct** — once reconstruction is finished, click *"Show point cloud"* to make the point cloud appear in an interactive 3D plot; toggle **RGB** vs **Normals** coloring.
4. **Mesh** — optionally extract the mesh, switch its vertex colors (RGB / Normals / None), and download the `.ply`.

</details>

## CLI

<details>
<summary>Click to expand</summary>
<br>

The script `scripts/infer.py` runs the full Surflo pipeline to reconstruct a surface from a folder of images. It can be run with or without guidance.

### **Guided** (rendering-guided flow → point cloud + mesh):

The following commands run Surflo with various guidance presets, using 100,000 points and a variable number of input images sampled in a folder:

```bash
# Default preset, good balance between runtime and mesh details
python scripts/infer.py mode=guided guided=default \
    ckpt=/path/to/surflo_v0.pt \
    source.image_folder=/path/to/images \
    source.n_images=16 \
    num_query_points=100000 \
    output_dir=outputs/surflo_guided

# Minimal preset, fastest inference
python scripts/infer.py mode=guided guided=minimal \
    ckpt=/path/to/surflo_v0.pt \
    source.image_folder=/path/to/images \
    source.n_images=8 \
    num_query_points=100000 \
    output_dir=outputs/surflo_guided

# Default preset, but with more vertices
python scripts/infer.py mode=guided guided=default_highres \
    ckpt=/path/to/surflo_v0.pt \
    source.image_folder=/path/to/images \
    source.n_images=16 \
    num_query_points=100000 \
    output_dir=outputs/surflo_guided

# Long preset, powerful when using a larger number of images
python scripts/infer.py mode=guided guided=long \
    ckpt=/path/to/surflo_v0.pt \
    source.image_folder=/path/to/images \
    source.n_images=64 \
    num_query_points=100000 \
    output_dir=outputs/surflo_guided
```
The argument `source.n_images` determines the number of images in the folder to use as inputs for reconstruction. You can set it to the value of your choice, or remove it to use every image in the folder. By default, it takes evenly spaced frames spanning the whole folder, but you can add `source.sampling=random` to draw a random subset instead.

You can choose the guidance preset with `guided=<preset>`. 
Find below the approximate cost of each preset on a **single NVIDIA H100**, with `num_query_points=100000`:

| Preset | Runtime | Peak VRAM | Detail |
|---|---|---|---|
| `mode=plain` (no guidance) | ~8 s | ~8.5 GiB | Coarse |
| `guided=minimal` | ~15 s | ~14 GiB | Good |
| `guided=short` | ~25 s | ~14 GiB | Sharp |
| `guided=short_highres` | ~25 s | ~14 GiB | Sharp, with more vertices |
| `guided=default` | ~45 s | ~14 GiB | Sharper, with better detail |
| `guided=default_highres` | ~45 s | ~14 GiB | Sharper, with better detail and more vertices |
| `guided=long` | ~95 s | ~14 GiB | Even sharper with many images |

You can use the presets `guided=short_highres` and `guided=default_highres`, 
identical to short and default (similar runtime) but generating higher-resolution meshes via a densification strategy.

### **Plain** (point cloud only, no mesh):

The following command runs Surflo without guidance, using 100,000 points and 16 input images sampled in a folder:

```bash
python scripts/infer.py mode=plain \
    ckpt=/path/to/surflo_v0.pt \
    source.image_folder=/path/to/images \
    source.n_images=16 \
    num_query_points=100000 \
    output_dir=outputs/surflo_plain
```

<details>
<summary>Click to find a list of common arguments for <code>infer.py</code>:</summary>

| Argument | Default | What it does |
|---|---|---|
| `ckpt` | *required* | Path to the model checkpoint. EMA weights are used when the file contains them (`use_ema=true`). |
| `source.image_folder` | *required* | Folder of JPG/PNG images for one scene, read in filename order. |
| `source.n_images` | `null` | How many views to reconstruct from. `null` uses every image in the folder. |
| `source.sampling` | `uniform` | How those views are chosen when the folder holds more than `n_images`. `uniform` takes evenly spaced frames spanning the whole folder, so a video or orbit capture is covered end to end; `random` draws a random subset instead, reproducible from `seed`. Both keep filename order, and both are ignored when `n_images` is `null`. |
| `num_query_points` | `100000` | Points drawn from the source distribution and flowed through the ODE. Most `guided` modes include an optional densification and pruning mechanism, which might leave the final cloud at a different size. |
| `mode` | `guided` | Whether to use guidance (`guided`) or not (`plain`). `plain` writes oriented points only (`initial.ply` + `final.ply`); `guided` adds rendering guidance and writes point clouds + `mesh.ply`. |
| `guided` / `plain` | `default` | Which preset to load for the selected mode (see the table above). |
| `seed` | `42` | Seeds the source sample and `source.sampling=random`. |
| `output_dir` | `outputs/surflo` | Where the `.ply` outputs and the run summary are written. |
| `mesh.enabled` | `true` | Extract `mesh.ply` from the guided Gaussians. Set `false` to stop after the point cloud. |
| `texture.enabled` | `true` | Also write a coloured copy of the mesh. Guided only. |
| `texture.mode` | `vertex_colors` | `vertex_colors` bakes TSDF colours per vertex into `mesh_textured.ply` — no extra dependency. `uv_texture` bakes a real UV texture atlas into `mesh_textured.glb`, and needs nvdiffrast + `.[texture]` (see [`install/README.md`](install/README.md)). |
| `mesh.delaunay_method` | `geodel` | Delaunay backend for the pivot tetrahedralization, which dominates meshing time. `geodel` ([GeoDel](https://github.com/Anttwo/GeoDel)) is multi-threaded — ~20× faster than the single-threaded `scipy` at 200 k points on 32 cores, for an identical tetrahedralization — and falls back to `scipy` with a warning if not installed. |
</details>

</details>

## Python API

<details>
<summary>Click to expand</summary>
<br>

`surflo/api.py` exposes a small facade over the pipeline. See
`examples/quickstart.py` for a runnable script.

```python
import torch
from surflo import Surflo, load_preset, save_ply, save_mesh

device = "cuda"

# 1. Load the model (EMA-aware checkpoint loader).
surflo = Surflo.from_checkpoint("/path/to/surflo_v0.pt", device=device)

# 2. Encode images into a fixed-size global state.
#    Accepts a folder path, a list of image paths, or an image tensor.
scene = surflo.encode("/path/to/images", n_images=16)

# 3. The global state is available for downstream applications.
print(scene.global_state.shape)          # (1, K, D) latent tokens

# 4. Load an inference preset and run the reconstruction.
# For guided inference, presets are the following:
#   > "minimal": Good
#   > "short": Sharp
#   > "default": Sharper, with better details
#   > "long": Even sharper, especially when using many images
result = scene.reconstruct(
    mode="guided",
    config_block=load_preset("guided", "minimal"),
    expert_cfg=load_preset("expert"),    # monodepth expert
    num_query_points=100_000,
)
save_ply(result, "/path/to/point_cloud.ply")

# 5. Optionally color the points
points = result["points"]  # (N, 3)
colors = scene.color_points(points, result)  # (N, 3)

# 6. Extract a mesh (guided results only) and save it.
mesh = scene.extract_mesh(result, mesh_cfg=load_preset("mesh"))
save_mesh(mesh, "path/to/mesh.ply")

# 7. Optionally color the mesh
scene.color_mesh(mesh, result)
save_mesh(mesh, "path/to/textured_mesh.ply")
```

Use `load_preset("<group>", "<name>")` to read any preset under `configs/` 
into a dict. 

For **plain** reconstruction, call `scene.reconstruct(mode="plain",
num_query_points=100_000)` — no `expert_cfg` needed. 

The flow velocity is also accessible directly. Use `scene.velocity(x, t)` 
for a point tensor `x` and time `t` as follows:
```python
import numpy as np

# `SceneState` works in unbatched (P, D) throughout. `scene_mean` / `scene_std`
# are (1, 1, 3) as the model computes them, so squeeze before broadcasting.
mean = scene.scene_mean.squeeze(0)   # (1, 3)
std = scene.scene_std.squeeze(0)     # (1, 3)

# Sample random 3D points and normals
x = mean + std * torch.randn(100_000, 3, device=device)  # (P, 3)

n = torch.nn.functional.normalize(
    torch.randn(100_000, 3, device=device), 
    dim=-1
)  # (P, 3)

# Convert to 6D points
p = surflo.model.get_r6_points_from_points_normals(x, n)  # (P, 6)

# Move points to normalized Flow space
p = scene.lift_to_flow_space(p)  # (P, 6)

# Sample random time
t = np.random.rand()

# Get velocity
velocity = scene.velocity(p, t)  # (P, 6)
```

`scene.sample_source(100_000)` returns a source cloud already in flow space, if
you do not need to build the points yourself. `scene.unlift_from_flow_space(p)`
maps back to world coordinates.

</details>

## Data and Preprocessing

<details>
<summary>Click to expand</summary>
<br>

Both training and evaluation consume **preprocessed caches** rather than raw
images: `scripts/preprocess.py` runs the VGGT-1B backbone once per scene and
writes everything needed to rebuild a batch without re-running VGGT
(`<scene>/{sample_*.pt, surface_data.npz}`). Run it once per dataset.

### Input scene layout

Following the structure of our [meshed DL3DV-10K dataset](https://huggingface.co/datasets/AntoineGuedon/DL3DV-10K-Meshed), each scene should be a **COLMAP-style dataset** (posed RGB views) augmented with **ground-truth surface labels**. 
`--scene_list` is a text file with one scene path per line, 
relative to `--data_dir` (paths may be nested, e.g. `10K/<hash>`;
a scene may also be packed as `<scene>.tar` with the same internal layout):

```
<data_dir>/<scene>/
├── images/                          # RGB views (jpg/png; extension auto-detected)
│   └── <name>.<ext>
└── gw_output/
    ├── cameras.json                 # per-view cameras (3DGS / COLMAP format)
    ├── scene_extent.pth             # dict: {"scene_radius": float, "scene_center": (3,)}
    ├── depth/<name>.pth             # per-view depth tensor (H, W), stem matches image
    ├── surface_point_labels/*.pth   # GT surface points  — chunked (N, 3) tensors
    └── surface_normal_labels/*.pth  # GT surface normals — chunked (N, 3) tensors
```

Each `cameras.json` entry holds `img_name` (stem, no extension), `width`,
`height`, `position` (3,), `rotation` (3×3), `fx` and `fy`.

The `surface_point_labels` / `surface_normal_labels` are oriented points sampled
from the scene's **ground-truth mesh**. For our training data we obtained the
meshes by running **[Gaussian Wrapping](#acknowledgements)** on each COLMAP scene
(hence the `gw_output/` name), but the mesh can come from **any** method — only
the sampled point / normal labels, per-view depth and `cameras.json` are required
here.

### Download DL3DV-10K-Meshed

Start by downloading our [modified version of DL3DV-10K](https://huggingface.co/datasets/AntoineGuedon/DL3DV-10K-Meshed).
You can run our dedicated script as shown below; it requires access to the gated repo, so please authenticate first with `hf auth login`:

```bash
python scripts/download_dl3dv_10k_meshed.py --odir dl3dv-10k-meshed                 # everything
python scripts/download_dl3dv_10k_meshed.py --odir dl3dv-10k-meshed --subset 1K 2K  # selected subsets
python scripts/download_dl3dv_10k_meshed.py --odir dl3dv-10k-meshed --workers 4     # throttle if rate-limited
```

### Run preprocessing on DL3DV-10K-Meshed

Then, run the following preprocessing script:

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

- `--min_images … --max_images` — sweep the view count to cache per-view-count subsets (`sample_*_views_NNN.pt`).
- `--n_samples` — number of preprocessed samples to generate for each pair *(scene, view count)*.
- `--save_vggt_world_points` — saves the VGGT points. Required for the radius cull and the adjusted source sampling.
- `--save_rgb_images` — required for guidance.


### Output layout

`--output_dir` gets one subdirectory **per scene**, named by the scene's leaf
folder (a nested `10K/<hash>` input becomes `<hash>/`). Each holds the VGGT-token
caches plus the consolidated GT surface cloud:

```
<output_dir>/<scene>/
├── sample_0000_views_016.pt        # one cache per (sample, view-count)
├── sample_0001_views_016.pt        #   … extra samples with --n_samples > 1
├── sample_0000_views_008.pt        #   … extra view counts from a --min/--max sweep
└── surface_data.npz                # GT surface points + normals (shuffled, chunked)
```

Each `sample_*.pt` is a dict holding the used-layer VGGT aggregated tokens
(`aggregated_tokens_list`, fp16), the COLMAP + VGGT cameras, the COLMAP↔VGGT
alignment (`alignment_L`, `alignment_T`), the scene extent (`scene_radius`,
`scene_center`), and — when the matching flag is set — `vggt_world_points` /
`rgb_images`. `surface_data.npz` stores the GT cloud as `n_chunks` plus
`points_NNN` / `normals_NNN` arrays. These are exactly what the training and
evaluation dataloaders read back (`data_dir` points at this `<output_dir>`).

</details>

## Training

<details>
<summary>Click to expand</summary>
<br>

Training reuses the same `surflo` package and model config. Install the extras
(`pip install -e ".[train]"`), then launch with `torchrun` from `training/`
(`PYTHONPATH=..` so the package resolves):

```bash
cd training

# Curriculum run on orbitshot scenes with variable N ∈ [2, 16]:
WANDB_MODE=offline PYTHONPATH=.. torchrun --nproc_per_node=4 train.py \
    override=orbit_2to16views \
    data_dir=/path/to/DL3DV-preprocessed \
    ood_data_dir=/path/to/FFM_test_preprocessed

# Curriculum run on all scenes with fixed N = 16 views:
WANDB_MODE=offline PYTHONPATH=.. torchrun --nproc_per_node=4 train.py \
    override=full_16views \
    data_dir=/path/to/DL3DV-preprocessed \
    ood_data_dir=/path/to/FFM_test_preprocessed
```

Training consumes the [preprocessed caches](#data-and-preprocessing) from a single
`data_dir`, and needs that directory to hold every view count in
`img_nums`. Checkpoints are written atomically after every epoch and resume
automatically from `checkpoint.pt`. See [`training/README.md`](training/README.md) for more details.

**Logging (Weights & Biases).** Scalar metrics **and 3D point clouds** are
logged to W&B on rank 0 (project `surflo`, run name `exp_name`). The commands above set
`WANDB_MODE=offline`, which writes to a local `wandb/` directory (relocate with
`WANDB_DIR`) and needs no network — upload afterwards with `wandb sync <dir>`.
For live logging, run `wandb login` once and drop `WANDB_MODE=offline`; use
`WANDB_MODE=disabled` to turn it off entirely (training also proceeds if W&B is
unavailable). Set the destination with `logging.writer.project=…` /
`logging.writer.entity=…` and the run name with `exp_name=…`.

⚠️ **Point-cloud logging is on by default and is what makes a run directory
large.** `logging.viz` writes a 100k-point cloud per visualized scene every
`viz.val_epoch_freq` epochs, every OOD-val epoch, and every
`viz.train_iter_freq` training steps (the ground-truth cloud is written once per
phase). Pass `logging.viz.enabled=false` when you only need the
loss / Chamfer curves, or lower `logging.viz.num_query_points`.

</details>

## Evaluation

<details>
<summary>Click to expand</summary>
<br>

After [preprocessing](#data-and-preprocessing), `scripts/evaluate.py` scores a
checkpoint in a single pass: inference + alignment (Umeyama + robust ICP) +
symmetric Chamfer / F1.

```bash
# Plain flow, on T&T:
python scripts/evaluate.py benchmarks=tnt \
    mode=plain \
    ckpt=/path/to/surflo_v0.pt \
    data_dir=/path/to/TNT-preprocessed \
    output_json=eval_results/tnt_plain.json

# With rendering guidance, on DL3DV:
python scripts/evaluate.py benchmarks=dl3dv \
    mode=guided guided=no_densification \
    cull_opacity_threshold=0.1 num_query_points=200000 \
    ckpt=/path/to/surflo_v0.pt \
    data_dir=/path/to/DL3DV-val-preprocessed \
    output_json=eval_results/dl3dv_guided.json
```

Benchmark presets live under `configs/benchmarks/` (`dl3dv`, `tnt`, `mipnerf`,
`dtu`, `scrream`, `deepblending`, `blendedmvs`, `mlhypersim`); always pass the
concrete preprocessed `data_dir`.

The same script scores the two feed-forward depth baselines through the identical
alignment + metric core (no checkpoint needed). Add `use_tsdf=true` to fuse
per-view depth into a TSDF mesh and sample its surface instead of the raw point map:

```bash
# VGGT baseline:
python scripts/evaluate.py benchmarks=tnt predictor=vggt \
    data_dir=/path/to/TNT-preprocessed output_json=eval_results/tnt_vggt.json

# DepthAnything-3 baseline:
python scripts/evaluate.py benchmarks=tnt predictor=da3 \
    data_dir=/path/to/TNT-preprocessed output_json=eval_results/tnt_da3.json
```

</details>

## Citation

If you use Surflo in your research, please cite:

```bibtex
@article{guedon2026surflo,
  title   = {Surflo: Consistent 3D Surface Flow Model with Global State},
  author  = {Gu{\'e}don, Antoine and Nakamura, Shu and Dufour, Nicolas
             and Lei, Jiahui and Nishino, Ko and Kanazawa, Angjoo},
  journal = {arXiv preprint arXiv:2606.13644},
  year    = {2026}
}
```

## Acknowledgements

Surflo builds on a number of excellent open-source projects:

- **[VGGT](https://github.com/facebookresearch/vggt)** — the frozen backbone that encodes the input views.
- **[Depth-Anything-3](https://github.com/bytedance-seed/depth-anything-3)** — monocular depth priors for the optional guidance expert.
- **[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)** — the differentiable Gaussian representation at the origin of the rendering guidance.
- **[RaDe-GS](https://github.com/HKUST-SAIL/RaDe-GS)** — Gaussian rasterization with accurate depth / normals.
- **[Gaussian Wrapping](https://github.com/diego1401/GaussianWrapping)** — optimization-based surface reconstruction from Gaussians.
- **[GeoDel](https://github.com/Anttwo/GeoDel)** — fast Delaunay triangulation for 3D points.

We thank the authors of these works for releasing their code.