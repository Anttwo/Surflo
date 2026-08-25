#!/usr/bin/env python
"""Report which Surflo features the current environment can actually run.

Mostly import-only, and safe on a login or build node: the one exception is the
``textured export`` row, which JIT-compiles nvdiffrast's plugins (and therefore
touches the GPU) because importing them proves nothing -- they build lazily on
first use. That step is skipped automatically when no GPU is visible.

    python install/verify_install.py
    python install/verify_install.py --check-isolation

``--check-isolation`` additionally asserts that the *plain* inference path
still imports with the compiled CUDA extensions hidden. That property holds
only because ``surflo/inference/engine.py`` defers its ``.guided`` / ``.plain``
imports into the functions that need them; since ``surflo/__init__.py`` eagerly
imports ``api.py``, a single new top-level import of the guided stack would
make the CUDA extensions mandatory for every user. Worth re-running after any
change to the import graph.
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OK = "\033[32m OK \033[0m"
NO = "\033[31mMISS\033[0m"


# ---------------------------------------------------------------------------
# Feature tiers: (label, gating module(s), what provides them)
#
# The second field may name SEVERAL modules, and every one of them is imported.
# That matters when a feature needs more than one thing and the obvious entry
# point only reaches some of them: `textured export` pulls nvdiffrast in at the
# top of texture_map, but xatlas is imported lazily inside uv_atlas, so checking
# texture_map alone reports OK on a half-installed feature.
# ---------------------------------------------------------------------------
TIERS: List[Tuple[str, Tuple[str, ...], str]] = [
    ("plain inference",   ("surflo",),                              "core package"),
    ("guided inference",  ("surflo.inference.guided",),             "diff_gaussian_rasterization_surflo, fused_ssim"),
    ("mesh extraction",   ("surflo.extraction.occupancy.wrapping",), "diff_gaussian_rasterization_surflo"),
    ("textured export",   ("surflo.extraction.texture.texture_map",
                           "xatlas"),                               "nvdiffrast (build_extensions.sh --with-nvdiffrast) + xatlas (pip install -e '.[texture]')"),
    # Not a hard requirement: without geodel the code falls back to
    # scipy.spatial.Delaunay, which is correct but single-threaded and dominates
    # meshing time. Reported so the slow path is visible rather than silent.
    ("fast delaunay",     ("geodel",),                              "geodel  (build_extensions.sh) - optional; scipy fallback is slower"),
    ("monodepth expert",  ("depth_anything_3.api",),                "Depth-Anything-3  (build_extensions.sh --with-da3)"),
    ("gradio demo",       ("gradio",),                              "pip install -e '.[demo]'"),
    ("training",          ("fvcore.common.param_scheduler",),       "pip install -e '.[train]'"),
]


def _try_import(names) -> Optional[str]:
    """Import every module in ``names``; return None on success, else the reason.

    Accepts a single module name or an iterable of them, and reports the first
    that fails so a partially-installed feature cannot report OK.
    """
    if isinstance(names, str):
        names = (names,)
    for name in names:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            missing = getattr(exc, "name", None)
            return f"{missing} not importable" if missing else str(exc)
        except Exception as exc:  # noqa: BLE001 - a broken install can raise anything
            return f"{type(exc).__name__}: {exc}"
    return None


def _nvcc_version() -> str:
    cuda_home = os.environ.get("CUDA_HOME", "")
    nvcc = str(Path(cuda_home) / "bin" / "nvcc") if cuda_home else ""
    if not (nvcc and os.access(nvcc, os.X_OK)):
        nvcc = shutil.which("nvcc") or ""
    if not nvcc:
        return "not found (CUDA_HOME unset and no nvcc on PATH)"
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=30).stdout
        for line in out.splitlines():
            if "release" in line:
                return line.split("release")[1].split(",")[0].strip() + f"  ({nvcc})"
    except Exception:  # noqa: BLE001
        pass
    return f"unknown ({nvcc})"


def _nvdiffrast_jit_check() -> Optional[str]:
    """Actually JIT-compile nvdiffrast's plugins; return None on success.

    Importing ``nvdiffrast.torch`` proves nothing: it compiles its CUDA/GL
    plugins lazily on FIRST USE, so a missing header stays invisible until
    someone textures a mesh. Both plugins include
    ``<ATen/cuda/CUDAContext.h>`` (needing cusparse/cublas/cusolver headers)
    and the GL one additionally needs ``EGL/egl.h`` -- none of which the
    minimal CUDA set provides. This builds them for real.

    The compiled result is cached under ~/.cache/torch_extensions, so this is
    slow once and instant afterwards.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return "skipped (no GPU visible)"
        import nvdiffrast.torch as dr
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"

    for label, factory in (("CUDA", dr.RasterizeCudaContext), ("GL", dr.RasterizeGLContext)):
        try:
            factory()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            missing = re.findall(r"fatal error: ([^\n:]+)", msg)
            if missing:
                return f"{label} plugin build failed - missing header: {sorted(set(missing))[0]}"
            return f"{label} context failed: {msg.splitlines()[0][:100]}"
    return None


def _compiler_version() -> str:
    cxx = os.environ.get("CXX") or shutil.which("c++") or ""
    if not cxx:
        return "not found"
    try:
        out = subprocess.run([cxx, "--version"], capture_output=True, text=True, timeout=30).stdout
        return (out.splitlines() or ["unknown"])[0]
    except Exception:  # noqa: BLE001
        return f"unknown ({cxx})"


def print_environment() -> Optional[str]:
    """Print the resolved toolchain. Returns torch's CUDA version, if any."""
    print("\nEnvironment")
    print("-" * 72)
    print(f"  python        {sys.version.split()[0]}  ({sys.executable})")

    torch_cuda = None
    try:
        import torch

        torch_cuda = torch.version.cuda
        print(f"  torch         {torch.__version__}   (CUDA {torch_cuda or 'CPU-only'})")
        try:
            import torchvision

            print(f"  torchvision   {torchvision.__version__}")
        except ImportError:
            print(f"  torchvision   {NO}")
    except ImportError:
        print(f"  torch         {NO}  — nothing will work without it")

    try:
        import numpy

        print(f"  numpy         {numpy.__version__}")
    except ImportError:
        print(f"  numpy         {NO}")

    print(f"  nvcc          {_nvcc_version()}")
    print(f"  compiler      {_compiler_version()}")
    print(f"  CUDA_HOME     {os.environ.get('CUDA_HOME', '<unset>')}")
    return torch_cuda


def check_toolchain_consistency(torch_cuda: Optional[str]) -> List[str]:
    """The one invariant worth shouting about: nvcc must match torch's CUDA."""
    problems: List[str] = []
    if not torch_cuda:
        return problems
    nvcc = _nvcc_version()
    if "not found" in nvcc or "unknown" in nvcc:
        return problems
    nvcc_ver = nvcc.split()[0]
    if nvcc_ver.split(".")[0] != torch_cuda.split(".")[0]:
        problems.append(
            f"CUDA major mismatch: torch was built against {torch_cuda}, nvcc is {nvcc_ver}. "
            "Extensions built now would fail to load."
        )
    elif nvcc_ver != torch_cuda:
        problems.append(
            f"CUDA minor mismatch: torch {torch_cuda} vs nvcc {nvcc_ver}. Usually fine, but an exact match is safer."
        )
    return problems


def print_features() -> int:
    print("\nFeatures")
    print("-" * 72)

    # depth_anything_3 lives in an in-repo source tree that has to be put on
    # sys.path first, so a bare import would report it missing even when it is
    # perfectly usable. This is the same setup surflo.inference.expert performs.
    try:
        from surflo.utils.da3_path import ensure_da3_importable

        ensure_da3_importable()
    except ImportError:
        pass  # core package broken; the "plain inference" row will say so

    missing = 0
    for label, module, provider in TIERS:
        reason = _try_import(module)
        # Imports alone do not prove nvdiffrast works -- it builds its plugins
        # on first use, so verify that for real rather than reporting a
        # half-working feature as OK.
        if reason is None and label == "textured export":
            jit = _nvdiffrast_jit_check()
            if jit is not None and not jit.startswith("skipped"):
                reason = jit
            elif jit is not None:
                label = f"{label} ({jit})"
        if reason is None:
            print(f"  [{OK}]  {label:<18}")
        else:
            missing += 1
            print(f"  [{NO}]  {label:<18}  {reason}")
            print(f"          {'':<18}  needs: {provider}")
    return missing


def check_isolation() -> bool:
    """Assert plain inference still imports with the CUDA extensions hidden."""
    print("\nImport isolation (plain inference must not need the CUDA extensions)")
    print("-" * 72)

    blocked = [
        "diff_gaussian_rasterization_surflo",
        "fused_ssim",
        "nvdiffrast",
        "xatlas",
        "gsplat",
        "diff_surfel_rasterization",
        "depth_anything_3",
    ]

    # Run in a subprocess so a partially-imported surflo cannot leak into this
    # process (and so an already-imported surflo does not mask the result).
    script = f"""
import sys
blocked = {blocked!r}
class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in blocked:
            raise ImportError('blocked: ' + name)
        return None
sys.meta_path.insert(0, _Blocker())
sys.path.insert(0, {str(REPO_ROOT)!r})

import surflo                                     # must not pull in the extensions
from hydra import compose, initialize_config_dir
with initialize_config_dir(config_dir={str(REPO_ROOT / "configs")!r}, version_base=None):
    cfg = compose(config_name="infer", overrides=["mode=plain", "ckpt=/dev/null",
                                                  "source.image_folder=/dev/null"])
assert cfg.mode == "plain"
print("OK")
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    if proc.returncode == 0 and proc.stdout.strip().endswith("OK"):
        print(f"  [{OK}]  import surflo + compose(mode=plain) with extensions blocked")
        return True
    print(f"  [{NO}]  plain inference now requires a compiled CUDA extension")
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    for line in tail[-12:]:
        print(f"          {line}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-isolation", action="store_true",
                        help="also assert plain inference imports without the CUDA extensions")
    args = parser.parse_args()

    print("=" * 72)
    print("Surflo installation check")
    print("=" * 72)

    torch_cuda = print_environment()
    problems = check_toolchain_consistency(torch_cuda)
    missing = print_features()

    isolation_ok = True
    if args.check_isolation:
        isolation_ok = check_isolation()

    print("\n" + "=" * 72)
    for p in problems:
        print(f"  WARNING: {p}")
    if missing == 0 and not problems and isolation_ok:
        print("  Everything checked is available.")
    elif missing:
        print(f"  {missing} optional feature(s) unavailable — see install/README.md.")
    print("=" * 72 + "\n")

    # Only a hard failure (broken core, or a broken isolation guarantee) is a
    # non-zero exit; missing optional features are informational.
    core_broken = _try_import("surflo") is not None
    return 1 if (core_broken or not isolation_ok) else 0


if __name__ == "__main__":
    sys.exit(main())
