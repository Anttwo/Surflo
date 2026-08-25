# diff-gaussian-rasterization-surflo

A single CUDA extension that merges the two rasterizers used by Surflo:

- **`diff-gaussian-rasterization`** (RaDe-GS render path): RGB + depth/coord/normal
  geometry rendering, with SH evaluated on the CUDA side.
- **`diff-gaussian-rasterization_ours`** (occupancy path): the `integrate` used by
  the wrapping mesh extraction.

On top of merging them, the render pass is extended to render **3 or 6 channels**
in a single forward/backward: 3 for classic RGB, or 6 for **RGB + learned normals**
(so the two RaDe-GS passes previously used — one for RGB, one for normals — become
one pass).

## What it does / how it stays faithful

- **RGB (channels 0:2)** go through the RaDe-GS `computeColorFromSH` path (with the
  `+0.5` offset and clamp-to-`>=0`), exactly as before.
- **Learned normals (channels 3:5)** are signed vectors, so they bypass the SH
  offset/clamp and are copied **clamp-free** into the color buffer, then accumulated
  with the same per-Gaussian `alpha * T` weights as RGB. Because accumulation is
  per-channel with shared geometry/opacity weights, the merged 6-channel output is
  provably equal to the two separate `render_radegs` passes.
- **Geometry outputs** (`depth`, `coord`, `alpha`, geometry `normal`) are
  channel-independent and unchanged.
- **Occupancy** (`integrate`) is copied verbatim from
  `diff-gaussian-rasterization_ours` and isolated in `namespace dgr_occ` so its
  symbols (`CudaRasterizer`, `FORWARD`, `BACKWARD`, …) do not clash with the render
  pipeline at link time. It is bit-identical to the original.

## Layout

- `cuda_rasterizer/` — RaDe-GS render pipeline (extended to 3-or-6 channels).
- `cuda_rasterizer_occ/` — occupancy pipeline, copied verbatim and wrapped in
  `namespace dgr_occ`.
- `rasterize_points.{cu,h}`, `ext.cpp` — shared torch bindings.
- `diff_gaussian_rasterization_surflo/__init__.py` — Python API.
- `tests/test_parity.py` — parity checks vs the two original packages.

## Python API

```python
from diff_gaussian_rasterization_surflo import (
    GaussianRasterizationSettings, GaussianRasterizer,
)

rasterizer = GaussianRasterizer(raster_settings=settings)

# 3-channel RGB (normals=None) or 6-channel RGB+normals (normals=(N,3)).
color, radii, coord, mcoord, depth, mdepth, alpha, geom_normal = rasterizer(
    means3D=means3D, means2D=means2D, opacities=opacity,
    shs=shs,                 # or colors_precomp=...
    normals=normals,         # optional; when given, color has 6 channels
    scales=scales, rotations=rotations, cov3D_precomp=None,
)

# Occupancy integration for wrapping meshing (forward-only):
alpha_integrated, inside = rasterizer.integrate(
    points3D=points3D, means3D=means3D, opacities=opacity,
    scales=scales, rotations=rotations,
)  # returns (1 - transmittance, inside)
```

The higher-level wrappers live in `surflo/rendering/surflo.py`
(`render_surflo`, `integrate_surflo`), mirroring `render_radegs` / `integrate_wrapping`.

## Build

Requires a CUDA toolkit and a GPU.

```bash
pip install -e submodules/diff-gaussian-rasterization-surflo
```

`nvcc` flags mirror `diff-gaussian-rasterization_ours` (notably `--use_fast_math`);
see `setup.py` for the rationale and how to trade off bit-identity with the stock
`diff-gaussian-rasterization` render path.

## Parity test

`tests/test_parity.py` builds a small random scene on the GPU and compares forward
outputs and backward gradients against the two original packages (RGB with/without
SH, learned normals, the merged 6-channel pass, geometry, and occupancy). It requires
all three packages built in the same environment:

```bash
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/diff-gaussian-rasterization_ours
pip install -e submodules/diff-gaussian-rasterization-surflo
python submodules/diff-gaussian-rasterization-surflo/tests/test_parity.py
```

## Credits

Built on [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
(3D Gaussian Splatting) and RaDe-GS. Please cite the original works:

```bibtex
@Article{kerbl3Dgaussians,
  author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal      = {ACM Transactions on Graphics},
  number       = {4},
  volume       = {42},
  month        = {July},
  year         = {2023},
  url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
