# Third-party notices

Surflo builds on several third-party projects. This file records what is
redistributed in this repository, who owns it, and under which terms. It is
informational: each component remains governed by its own license, and nothing
here modifies those terms.

**Summary of what constrains you.** Two of the components below are restricted
to non-commercial research and evaluation use, and one of them (the
Gaussian-Splatting License, §4.2) requires derivative works to carry the same
use limitation. Surflo is therefore released under the Gaussian-Splatting
License (see [`LICENSE.md`](LICENSE.md)) and **may not be used commercially**
without prior written consent from the respective licensors. This applies to
Surflo as a whole, not only to the files listed as 3DGS-derived.

---

## Code redistributed in this repository

### Gaussian Splatting — Inria and Max Planck Institut für Informatik (MPII)

*Gaussian-Splatting License* — non-commercial, research and evaluation use only.
Full text in [`LICENSE.md`](LICENSE.md), and in
`submodules/diff-gaussian-rasterization-surflo/LICENSE.md`.

Derived from <https://github.com/graphdeco-inria/gaussian-splatting>, modified:

- `surflo/structures/cameras.py`
- `surflo/structures/struct_utils.py`
- `surflo/rendering/surflo.py`
- `submodules/diff-gaussian-rasterization-surflo/` (the whole vendored
  rasterizer, which retains its own copy of the license)

Contact for use outside these terms: `stip-sophia.transfert@inria.fr`.

### VGGT — Meta Platforms, Inc.

*VGGT License* — see <https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt>.
Redistribution (including of derivative works) is permitted only under the terms
of that agreement, a copy of which must be provided to any third party you
distribute to. Publications reporting results obtained with these materials must
acknowledge their use. The license also carries an Acceptable Use Policy.

Derived from <https://github.com/facebookresearch/vggt>, modified:

- `surflo/nn/vggt/` (backbone, heads, layers and utilities)

### PlenOctree — The PlenOctree Authors

*BSD 2-Clause License* (full text retained in the file header).

- `surflo/rendering/sh_utils.py`

### Kaolin — NVIDIA Corporation & Affiliates

*Apache License 2.0* (full text at <https://www.apache.org/licenses/LICENSE-2.0>).

Adapted from
<https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py>:

- `surflo/extraction/marchers/tetrahedra.py`
- `surflo/extraction/marchers/masked_tetrahedra.py`

### Bundled inside the vendored rasterizer

- **GLM** — G-Truc Creation, *MIT / The Happy Bunny License*
  (`submodules/diff-gaussian-rasterization-surflo/third_party/glm/`)
- **stb_image_write** — Sean Barrett, *MIT / public domain*
  (`submodules/diff-gaussian-rasterization-surflo/third_party/stbi_image_write.h`)

---

## Model weights downloaded at run time

### VGGT-1B — Meta Platforms, Inc.

*CC BY-NC 4.0* — **non-commercial**.
<https://huggingface.co/facebook/VGGT-1B>

Fetched automatically from the Hugging Face Hub on first model construction.
These weights are not redistributed in this repository, but every Surflo run
depends on them, so their non-commercial restriction applies in practice to any
use of Surflo.

---

## Optional components installed separately

These are not redistributed here. They are fetched by
`install/build_extensions.sh` or by the optional extras, and each is governed by
its own license.

| Component | License | Notes |
|---|---|---|
| [nvdiffrast](https://github.com/NVlabs/nvdiffrast) | NVIDIA Source Code License (1-Way Commercial) | Optional — textured-mesh export only |
| [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | Apache License 2.0 | Optional — monodepth expert guidance |
| [fused-ssim](https://github.com/rahul-goel/fused-ssim) | see upstream repository (no license declared in package metadata) | Required for guided inference |
| [GeoDel](https://github.com/Anttwo/GeoDel) | BSD 3-Clause | Fast Delaunay backend; vendors [Geogram](https://github.com/BrunoLevy/geogram) |

Python dependencies installed from PyPI (see `pyproject.toml`) carry their own
licenses. Note that `flow-matching` ships **contradictory metadata** — its
`License` field reads `CC-by-NC` while its trove classifier says `MIT License`.
Treat it as non-commercial until upstream clarifies, and verify before relying
on it for anything commercial.

---

*This file is a good-faith inventory prepared by the authors, not legal advice.
If you intend to use Surflo outside non-commercial research and evaluation, seek
your own legal review and contact the respective licensors.*
