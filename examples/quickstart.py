"""Surflo Python API quickstart.

Walks through the four things the ``surflo`` facade is built for, end-to-end:

  1. load a model + checkpoint,
  2. encode a scene's images (compressor + VGGT preprocessing),
  3. solve the ODE (plain, and optionally guided) into an oriented
     point cloud.
  4. extract and texture a surface mesh from the points.

Run it from the repository root:

    python examples/quickstart.py \
        --ckpt /path/to/surflo_v0.pt \
        --images media/sample \
        --out outputs/quickstart \
        --guided --preset default_highres

Add ``--guided`` to also run the rendering-guided variant (slower,
needs the guidance dependencies). Everything here is a thin wrapper over
``scripts/infer.py``'s building blocks; see ``surflo/api.py``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# Allow ``python examples/quickstart.py`` from a source checkout by putting the
# package root (the parent of examples/) on sys.path.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from surflo import (
    Surflo, 
    save_ply, 
    save_mesh, 
    load_preset, 
    set_global_seeds
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Surflo Python API quickstart.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to surflo_v0.pt")
    p.add_argument(
        "--images",
        type=str,
        required=True,
        help="Folder of JPG/PNG images (or pass a few paths yourself in code).",
    )
    p.add_argument("--out", type=str, default="outputs/quickstart", help="Output dir.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-images", type=int, default=None, help="Views to use (default: all).")
    p.add_argument("--cull-radius", type=float, default=10.0,
                   help="Spatial cull radius (CLI default 10.0; required for "
                        "guided mask losses). Pass a value <=0 to disable.")
    p.add_argument("--num-query-points", type=int, default=100_000)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--guided",
        action="store_true",
        help="Run guided inference with the DepthAnything-3 monodepth/normal prior "
        "(needs the guidance + DA3 deps installed).",
    )
    p.add_argument("--preset", type=str, default="default", help="Preset to use for guided inference.")
    return p.parse_args()


def main() -> None:
    # The library logs through the stdlib `logging` module. These entry points
    # are not Hydra apps (which would configure it for us), so set it up here or
    # nothing below WARNING would be shown.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seed the global RNGs (guided variants sample their source cloud / random
    # backgrounds from them) so runs are reproducible, exactly like the CLI.
    set_global_seeds(args.seed)

    # --- Point 1: load the model + checkpoint (bundled model config by default).
    surflo = Surflo.from_checkpoint(args.ckpt, device=args.device)

    # --- Point 2: encode the scene's images -> preprocessed data + global state.
    #     The global state can be used for downstream tasks.
    scene = surflo.encode(
        args.images,
        n_images=args.n_images,
        cull_radius=args.cull_radius,
    )
    print(
        f"[quickstart] encoded {scene.images.shape[0]} views "
        f"-> global state {tuple(scene.global_state.shape)}."
    )

    # --- Point 3a: plain (unguided) ODE solve -> oriented surface cloud.
    plain = scene.reconstruct(
        mode="plain",
        num_steps=args.num_steps,
        num_query_points=args.num_query_points,
        seed=args.seed,
        return_source=True,
    )
    n_plain = save_ply(plain, out_dir / "plain_points.ply")
    save_ply(plain, out_dir / "plain_source.ply", which="source")
    print(f"[quickstart] wrote {out_dir / 'plain_points.ply'} ({n_plain} points).")

    if args.guided:
        # --- Point 3b: optional guided inference.
        guided = scene.reconstruct(
            mode="guided",
            config_block=load_preset("guided", args.preset),
            expert_cfg=load_preset("expert"),    # monodepth expert
            num_query_points=args.num_query_points,
            seed=args.seed,
        )
        n_dens = save_ply(guided, out_dir / "guided_points.ply")
        print(f"[quickstart] wrote {out_dir / 'guided_points.ply'} ({n_dens} points).")
        
        # --- Point 4a: Extract a surface mesh from the guided points.
        mesh = scene.extract_mesh(guided, mesh_cfg=load_preset("mesh"))
        n_verts = mesh.verts.shape[0]
        n_faces = mesh.faces.shape[0]
        
        # --- Point 4b: Texture the mesh with vertex colors.
        scene.color_mesh(mesh, guided)
        save_mesh(mesh, out_dir / "guided_mesh.ply")
        print(f"[quickstart] wrote {out_dir / 'guided_mesh.ply'} ({n_verts} vertices, {n_faces} faces).")

    print("[quickstart] done.")


if __name__ == "__main__":
    with torch.no_grad():
        main()
