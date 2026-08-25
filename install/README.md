# Installing Surflo

Surflo is installed from a **source checkout** — it needs compiled CUDA
extensions that must be built against the torch in your environment, so there is
no `pip install surflo`.

That gives the install one golden rule:

> **Pick one CUDA family and use it for everything** — torch, the PyG wheels, and
> the `nvcc` that compiles the extensions. Once the extensions are built,
> nothing may re-install torch: a different torch is a different ABI, and the
> extensions stop loading.

Two ways to satisfy it:

| | Option A — conda | Option B — uv / pip |
|---|---|---|
| supplies `nvcc` + compiler | **yes**, from the env file | no — you supply a matching CUDA toolkit |
| extra disk | ~55 MB of CUDA build tools | none |
| best for | a workstation, or anywhere without a system CUDA | HPC with `module load cuda/...` |

**Both options install the same Python packages.** The conda env file supplies
only the build toolchain and then defers to the matching `requirements-*.txt`,
so torch comes from the same wheel either way. Option A is Option B plus a
compiler.

Pick your CUDA family:

| CUDA | conda env file | pip / uv requirements | torch |
|---|---|---|---|
| 11.8 | `install/environment-cu118.yml` | `install/requirements-cu118.txt` | 2.3.1+cu118 |
| 12.1 | `install/environment-cu121.yml` | `install/requirements-cu121.txt` | 2.3.1+cu121 |
| 12.4 | `install/environment-cu124.yml` | `install/requirements-cu124.txt` | 2.4.1+cu124 |

Run everything **from the repository root**. Substitute your family for
`cu118` throughout.

---

## Option A — conda (self-contained)

The env file brings its own `nvcc` and C++ compiler, so nothing outside conda
has to match. It also installs the whole Python stack, torch included, by
deferring to `requirements-cu118.txt` — so unlike Option B you never install
the dependencies yourself. The `pip install -e` below adds only the `surflo`
package itself and its extras.

> **Use the libmamba solver** if `conda config --show solver` says `classic` —
> the older solver is extremely slow on these channels (it did not converge in
> 15 minutes per file in our testing, where libmamba takes seconds):
>
> ```bash
> conda install -n base conda-libmamba-solver
> conda config --set solver libmamba
> ```
>
> conda ≥ 23.10 already defaults to libmamba, so this mainly affects older
> installations.

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

`activate_cuda.sh` writes a conda `activate.d` hook (and a matching
`deactivate.d` one) rather than exporting the variables once. That matters: a
one-off `export` is lost in the next shell, and a later rebuild would silently
pick up whatever `nvcc` happens to be on `PATH`, producing an extension that
does not match your torch.

**What conda installs.** Only four CUDA packages (~55 MB), not the multi-GB
`cuda-toolkit` metapackage: `cuda-nvcc`, `cuda-cudart-dev`, `cuda-driver-dev`
and `cuda-cccl`. Between them they cover every CUDA header the extensions
include (`cuda.h`, `cuda_runtime.h`, `cooperative_groups`, `cub/`) and the only
CUDA library they link against (`libcudart`). torch's own headers pull in no
CUDA math libraries. If a future extension does need one, replace those four
with `cuda-toolkit=11.8` in the env file.

Note the environment ends up with two copies of the CUDA runtime — conda's and
the one the torch wheel bundles. They are the same version by construction, so
it makes no difference which the loader reaches first; the conda copy is what
keeps the environment usable on a machine with no system CUDA at all.

## Option B — uv / pip (you supply CUDA)

Use this when you already have a CUDA toolkit whose version matches your target
family — an HPC `module load`, or a system install. Verify with `nvcc --version`
before starting.

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

⚠️ **`--index-strategy unsafe-best-match` is required with uv**, not optional.
By default uv only considers the first index that contains a package at all, so
it finds an outdated `tqdm` on the PyTorch index and never looks at PyPI —
resolution then fails outright. (Set `UV_INDEX_STRATEGY=unsafe-best-match` to
avoid repeating the flag.) The name is alarming but the trade-off is explicit:
it lets uv pick the best version across every configured index rather than
pinning to index precedence.

Plain `pip` needs no such flag — it already considers every index:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r install/requirements-cu124.txt
pip install -e .
```

The `pip install -e .` step is separate on purpose. The requirements files are
shared between both options, and a relative `-e` path inside them cannot be
right for both callers — pip resolves it against the *requirements file's* own
directory (`install/`), not the directory you run from.

---

## Compiled CUDA extensions

`build_extensions.sh` checks that your `nvcc` and torch agree **before** it
builds anything — a mismatch otherwise surfaces as pages of nvcc template errors
several minutes in. It then builds:

| | source | required for |
|---|---|---|
| `diff_gaussian_rasterization_surflo` | vendored in `submodules/` | guided inference, mesh extraction |
| `fused_ssim` | fetched from GitHub | guided inference |
| `geodel` | fetched from GitHub | fast Delaunay for mesh extraction — **optional in practice**: without it the code falls back to `scipy.spatial.Delaunay`, which is correct but single-threaded and dominates meshing time |
| `nvdiffrast` | `--with-nvdiffrast` | textured export — see [Optional features](#optional-features) |
| Depth-Anything-3 | `--with-da3` | monodepth expert — see [Optional features](#optional-features) |

```bash
bash install/build_extensions.sh                            # the two required ones
bash install/build_extensions.sh --with-nvdiffrast --with-da3
bash install/build_extensions.sh --all
```

The first two are **required for guided inference and mesh extraction**; the
other two are optional and covered below. **Plain inference (`mode=plain`) needs
none of them** — only the core package and torch, which
`verify_install.py --check-isolation` asserts stays true.

### GPU architectures

The script targets the GPU it can see. Override for a different target, or when
building on a login node with no GPU:

```bash
export TORCH_CUDA_ARCH_LIST="8.0;9.0"      # A100 + H100
```

With no GPU visible and nothing set, it falls back to `7.0;7.5;8.0;8.6;8.9;9.0`,
which builds slowly but runs anywhere in that range. None of the three CUDA
families here supports Blackwell (sm_100 / sm_120), which needs CUDA ≥ 12.8.

---

## Optional features

After a base install, `python install/verify_install.py` prints a feature list
with some rows marked `MISS`. Everything marked `MISS` is optional — plain
reconstruction, guided reconstruction and mesh extraction already work without
any of it. This table maps each row to what it gives you and how to get it.

| `verify_install.py` row | What it gives you | How to install |
|---|---|---|
| **monodepth expert** | Runs Depth-Anything-3 on every input view and feeds its depth estimates in as extra hints during guided reconstruction. 7 of the 8 guided presets use it when present. **The only optional item that changes reconstruction quality.** | `bash install/build_extensions.sh --with-da3` |
| **textured export** | Bakes a real texture *image* onto the mesh (a UV map) instead of one flat colour per vertex, so fine detail survives. Enabled with **`texture.mode=uv_texture`**, which writes `mesh_textured.glb`. Not the default: `texture.enabled` is already `true`, but `texture.mode=vertex_colors` writes per-vertex colours (`mesh_textured.ply`) and needs none of this. Needs extra system headers on the uv path — see below. | `bash install/build_extensions.sh --with-nvdiffrast` **and** `pip install -e ".[texture]"` |
| **gradio demo** | Browser interface: drop in photos, watch the reconstruction, download the result (`examples/gradio_demo.py`). | `pip install -e ".[demo]"` |
| **training** | Train a model. Not needed to *run* one. | `pip install -e ".[train]"` |

Everything at once:

```bash
bash install/build_extensions.sh --all  # nvdiffrast + Depth-Anything-3
pip install -e ".[demo,texture,train]"  # all Python extras in one go
```

Note the comma-separated form: repeating `-e ".[a]" -e ".[b]"` asks pip to
install the *same* project twice with different extras, which is ambiguous.
One bracket with commas is the unambiguous way to combine them.

### GeoDel — the Delaunay backend

Mesh extraction is dominated by the Delaunay tetrahedralization of the pivot
cloud. [GeoDel](https://github.com/Anttwo/GeoDel) wraps Geogram's
ParallelDelaunay3d and is **~15× faster** than `scipy.spatial.Delaunay`
at 200 k points, producing an identical tetrahedralization. That figure is from
a 32-core machine; since scipy's Qhull is single-threaded, the gap scales with
the core count and will be smaller on a laptop.

It is installed by `build_extensions.sh` and needs **no CUDA** — only a C++
compiler and OpenMP — because it vendors Geogram directly (no git submodule).

Select the backend without touching code:

```bash
python scripts/infer.py ... mesh.delaunay_method=scipy    # default: geodel
```

If geodel is not installed the code logs one warning and falls back to scipy,
so nothing breaks — you just get the slow path. `verify_install.py` reports it
as `fast delaunay` for that reason. Pin a different revision with
`GEODEL_REF=<sha> bash install/build_extensions.sh`.

### nvdiffrast builds late — and needs headers CUDA doesn't provide

nvdiffrast is the one dependency that does **not** compile when you install it.
It JIT-builds its plugins the *first time you texture a mesh*, so a missing
header stays invisible until then — `import nvdiffrast.torch` succeeds either
way. (`verify_install.py` therefore builds the plugins for real rather than
importing them.)

Its `framework.h` includes `<ATen/cuda/CUDAContext.h>` directly, which needs
`cusparse.h`, `cublas_v2.h`, `cublasLt.h` and `cusolverDn.h`; the GL backend
additionally needs `EGL/egl.h`. Neither set is required by the Surflo
rasterizer, so they are easy to miss.

- **conda path** — handled for you: the env files install `libcusparse-dev`,
  `libcublas-dev`, `libcusolver-dev`, `libegl-devel` and `libgl-devel`.
- **uv / pip path** — you supply these. A full CUDA toolkit already ships the
  four math headers, so usually only the GL ones are missing, and they come from
  the OS rather than from CUDA:

  ```bash
  sudo apt install libegl1-mesa-dev libgl1-mesa-dev     # Debian / Ubuntu
  sudo dnf install mesa-libEGL-devel mesa-libGL-devel   # RHEL / Fedora
  ```

  On an HPC module system, a `module load cuda/...` that exposes only nvcc and
  the runtime may also lack the math headers; `ls $CUDA_HOME/include/cusparse.h`
  tells you. Nothing here is needed unless you want textured export.

### Why some need a script and others just need pip

Three different kinds of dependency show up here, which is what the two columns
above really encode:

- **Compiled CUDA code** — C++/CUDA source that has to be turned into machine
  code against *your* torch, so it cannot be a ready-made download. This is what
  `build_extensions.sh` is for. `nvdiffrast` is in this group.
- **Ordinary Python packages** — already built, just fetched. `gradio`, `plotly`,
  `xatlas`, the training tools. These are the `pip install -e ".[…]"`
  extras, declared in `pyproject.toml`.
- **A source folder inside the repo** — not a package at all. Depth-Anything-3 is
  a directory that Python is *pointed at*; `surflo.inference.expert` adds it to
  `sys.path` on import. This is why it has its own flag rather than an extra.

**Textured export is the only feature needing two of these at once** — the
compiled renderer *and* the `xatlas` package — so one command is not enough.

The `pip install -e ".[…]"` commands are safe to run after building the
extensions: the torch requirement in `pyproject.toml` spans every supported CUDA
family, so pip leaves your torch alone rather than replacing it and breaking
what you just compiled.

### Reading a MISS row

The row names the *first* thing Python could not import, which is not always the
feature itself. For example:

```
[MISS]  monodepth expert    moviepy not importable
```

`moviepy`, not `depth_anything_3` — meaning the Depth-Anything-3 folder *was*
found and started loading, and only one of its own dependencies is missing. Had
the path wiring been wrong you would see `depth_anything_3 not importable`
instead. (`--with-da3` installs `moviepy<2` first for exactly this reason; see
Troubleshooting.)

---

## Weights

- **VGGT-1B** is fetched automatically from the Hugging Face Hub
  (`facebook/VGGT-1B`) on first model construction.
- The **Surflo checkpoint** is passed explicitly, e.g.
  `ckpt=/path/to/surflo_v0.pt`. The loader is EMA-aware and needs no
  extra package.

## Troubleshooting

**`nvcc` and torch disagree.** `build_extensions.sh` stops with the two versions
it found. Either point `CUDA_HOME` at a matching toolkit, or reinstall torch
from the `install/` file for the toolkit you have.

**`import torch_cluster` fails.** The wheel did not match your torch build.
Reinstall it from the `--find-links` index in your family's requirements file;
there is no matching wheel on PyPI, so a plain `pip install torch-cluster`
falls back to a slow source build.

**`depth_anything_3` not found.** It is not a pip package — it is a source tree
in `submodules/Depth-Anything-3/src` that `surflo.inference.expert` puts on
`sys.path` as an import side effect. A bare
`importlib.util.find_spec("depth_anything_3")` therefore reports missing on
every machine. Set `DA3_SRC` to override the location. DA3 also needs
`moviepy < 2` (2.x removed `moviepy.editor`); `--with-da3` handles that.

**The rasterizer build fails with `ModuleNotFoundError: torch`** (or fails only
on some pip versions). The extensions' `setup.py` imports
`torch.utils.cpp_extension` but they ship no `pyproject.toml` declaring torch as
a build dependency, so an isolated build cannot see it.
`build_extensions.sh` passes `--no-build-isolation` for exactly this reason; if
you build by hand, pass it yourself:

```bash
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization-surflo
```

**`fatal error: cusparse.h` / `EGL/egl.h` — usually while texturing a mesh.**
nvdiffrast JIT-builds its plugins on first use, so this surfaces long after
install. The conda env files ship the headers it needs; on the uv path you
supply them — see
[nvdiffrast builds late](#nvdiffrast-builds-late--and-needs-headers-cuda-doesnt-provide).
`python install/verify_install.py` reproduces it without running the demo.

**Some other CUDA header is missing.** The conda env files install a targeted
CUDA set rather than the full toolkit. If something needs a header none of them
provide, replace the whole `cuda-*` / `lib*-dev` block with a single
`cuda-toolkit=11.8` (matching your family) and recreate the environment.

**`identifier "uint32_t" is undefined` / `namespace "std" has no member
"uintptr_t"`** while compiling the rasterizer. GCC 13 stopped pulling `<cstdint>`
in transitively through other standard headers, so code that relied on that no
longer compiles. The vendored rasterizer in `submodules/` already carries the
fix — an explicit `#include <cstdint>` in the nine affected headers — so you
should only see this with a differently-sourced copy. **If you re-point that
submodule at another repository, make sure the fix is present there**, or the
CUDA 12.4 environment (which pins GCC 13) will stop building.

**The build is very slow.** Install `ninja` (it is in every install file); without
it the extension compiles single-threaded, roughly 20 minutes instead of 2.
