#!/usr/bin/env bash
# Build the compiled CUDA extensions Surflo needs, against the torch and the
# nvcc currently active.
#
#   REQUIRED for guided inference and mesh extraction:
#     * diff_gaussian_rasterization_surflo   the Surflo renderer + occupancy integration
#     * fused_ssim                           SSIM term of the rendering loss
#     * geodel                               fast Delaunay for mesh extraction; needs no
#                                            CUDA, and the code falls back to
#                                            scipy.spatial.Delaunay (slower) without it
#   OPTIONAL:
#     * nvdiffrast        --with-nvdiffrast   UV-textured mesh export (needs OpenGL/EGL)
#     * Depth-Anything-3  --with-da3          monodepth expert guidance
#
# Plain inference (`mode=plain`) needs none of these.
#
# Usage — run from the repository root, inside your environment:
#   bash install/build_extensions.sh                    # the three required ones
#   bash install/build_extensions.sh --with-nvdiffrast --with-da3
#   bash install/build_extensions.sh --all
#
# Override the target GPU architectures by setting TORCH_CUDA_ARCH_LIST before
# running (see "GPU architectures" below).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WITH_NVDIFFRAST=0
WITH_DA3=0
for arg in "$@"; do
    case "$arg" in
        --with-nvdiffrast) WITH_NVDIFFRAST=1 ;;
        --with-da3)        WITH_DA3=1 ;;
        --all)             WITH_NVDIFFRAST=1; WITH_DA3=1 ;;
        -h|--help)         sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$1" >&2; shift; for l in "$@"; do printf '       %s\n' "$l" >&2; done; exit 1; }
warn() { printf '\033[1;33mWARNING: %s\033[0m\n' "$*" >&2; }

# Some source trees keep nvdiffrast / Depth-Anything-3 at the repo root rather
# than under submodules/; accept either.
find_dir() {
    for candidate in "submodules/$1" "$1"; do
        if [[ -d "$candidate" ]]; then echo "$candidate"; return 0; fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# Preflight — every failure here would otherwise surface as a wall of nvcc
# template errors several minutes into a build.
# ---------------------------------------------------------------------------
say "Preflight"

command -v python >/dev/null 2>&1 || fail "no 'python' on PATH." "Activate your Surflo environment first."

TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null)" || fail \
    "could not import torch." \
    "Install the environment first (install/environment-cu*.yml or install/requirements-cu*.txt)."
[[ -n "$TORCH_CUDA" ]] || fail \
    "the installed torch is a CPU-only build (torch.version.cuda is None)." \
    "Reinstall from one of the install/ files for your CUDA family."

if [[ -z "${CUDA_HOME:-}" ]]; then
    fail "CUDA_HOME is not set." \
         "conda env: run 'bash install/activate_cuda.sh', then reactivate." \
         "otherwise:  export CUDA_HOME=/path/to/cuda-${TORCH_CUDA} (or 'module load cuda/${TORCH_CUDA}')."
fi

NVCC="$CUDA_HOME/bin/nvcc"
[[ -x "$NVCC" ]] || NVCC="$(command -v nvcc || true)"
[[ -n "$NVCC" && -x "$NVCC" ]] || fail \
    "nvcc not found (looked in \$CUDA_HOME/bin and on PATH)." \
    "CUDA_HOME is currently '$CUDA_HOME'."

NVCC_CUDA="$("$NVCC" --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')"
[[ -n "$NVCC_CUDA" ]] || fail "could not parse the version from '$NVCC --version'."

if [[ "${TORCH_CUDA%%.*}" != "${NVCC_CUDA%%.*}" ]]; then
    fail "CUDA major version mismatch — the extensions would not load." \
         "torch was built against CUDA $TORCH_CUDA, but nvcc is $NVCC_CUDA." \
         "Point CUDA_HOME at a CUDA ${TORCH_CUDA%%.*}.x toolkit, or reinstall torch" \
         "from the install/ file matching your toolkit."
elif [[ "$TORCH_CUDA" != "$NVCC_CUDA" ]]; then
    warn "torch was built against CUDA $TORCH_CUDA but nvcc is $NVCC_CUDA."
    warn "Same major version, so this usually works — but an exact match is safer."
fi

python -c 'import ninja' 2>/dev/null || warn "ninja is not installed — the build will be single-threaded (~20 min instead of ~2). 'pip install ninja' to fix."

# Needed because the builds below use --no-build-isolation (see the comment
# there), which means setup.py runs against this environment's setuptools.
python -c 'import setuptools' 2>/dev/null || fail \
    "setuptools is not installed in this environment." \
    "The extensions build with --no-build-isolation and need it: pip install setuptools wheel"

# --- GPU architectures -----------------------------------------------------
# Precedence: an explicit TORCH_CUDA_ARCH_LIST wins; otherwise, if a GPU is
# visible, torch auto-detects it (this is what the reference builds did);
# otherwise fall back to a broad list so headless build/login nodes work.
# Note none of the supported CUDA families covers Blackwell (sm_100/sm_120),
# which needs CUDA >= 12.8.
if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    ARCH_NOTE="TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST (preset)"
elif python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() else 1)' 2>/dev/null; then
    ARCH_NOTE="auto-detected from $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
else
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
    ARCH_NOTE="TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST (no GPU visible; broad fallback)"
    warn "No GPU visible — building for a broad architecture list. This is slower."
    warn "Set TORCH_CUDA_ARCH_LIST yourself to build only what you need."
fi

echo "  torch CUDA : $TORCH_CUDA"
echo "  nvcc       : $NVCC_CUDA  ($NVCC)"
echo "  compiler   : ${CXX:-<default c++>}"
echo "  arch       : $ARCH_NOTE"

# ---------------------------------------------------------------------------
# Required extensions
# ---------------------------------------------------------------------------
RASTERIZER="submodules/diff-gaussian-rasterization-surflo"
[[ -d "$RASTERIZER" ]] || fail "$RASTERIZER not found." "Run this from the repository root."

# --no-build-isolation is REQUIRED, not an optimisation, and it is applied to
# EVERY source-tree install below -- required and optional alike. The rasterizer,
# fused_ssim and (depending on the checkout) nvdiffrast all do
# `from torch.utils.cpp_extension import ...` at the top of setup.py without
# declaring torch as a build dependency, so pip's isolated build runs them in a
# fresh environment with no torch and they abort.
#
# Pin the torch already in this environment for every dependency resolution below.
# The extensions are compiled against this exact torch, so replacing
# it breaks them -- and it breaks them SILENTLY: the install succeeds, then
# importing an extension dies on an undefined symbol.
#
# This is not hypothetical. Depth-Anything-3 declares `torch>=2` and an
# UNPINNED `xformers`, while every xformers wheel hard-pins one exact torch
# (0.0.27 -> torch==2.3.1, 0.0.28.post1 -> torch==2.4.1). Nothing stops pip
# resolving an xformers that demands a different torch and upgrading yours.
# With this constraint pip fails loudly instead of quietly swapping torch.
#
# NOTE: numpy is deliberately NOT constrained -- DA3 genuinely requires
# `numpy<2`, and that downgrade is harmless here (the metrics use torch, and
# numpy's RNG streams are identical across 1.26/2.x for the calls this code
# makes). Constraining it would make the DA3 install unsatisfiable.
PIN_FILE="$(mktemp)"
trap 'rm -f "$PIN_FILE"' EXIT
python - > "$PIN_FILE" <<'EOF'
import torch, torchvision
print(f"torch=={torch.__version__}")
print(f"torchvision=={torchvision.__version__}")
EOF
echo "  pinning for dependency resolution: $(tr '\n' ' ' < "$PIN_FILE")"

BUILD_FLAGS=(--no-build-isolation)

say "1/3  diff_gaussian_rasterization_surflo  (Surflo renderer + occupancy)"
pip install "${BUILD_FLAGS[@]}" -c "$PIN_FILE" "./$RASTERIZER"

say "2/3  fused_ssim  (SSIM term of the rendering loss)"
if FUSED_SSIM_DIR="$(find_dir fused-ssim)"; then
    pip install "${BUILD_FLAGS[@]}" -c "$PIN_FILE" "./$FUSED_SSIM_DIR"
else
    pip install "${BUILD_FLAGS[@]}" -c "$PIN_FILE" git+https://github.com/rahul-goel/fused-ssim.git
fi

# GeoDel is the default Delaunay backend for mesh extraction, where the
# tetrahedralization dominates wall-clock time. Unlike the two above it needs
# no CUDA -- only a C++ compiler and OpenMP -- but it lives here because it is
# a compiled package from git, and because mesh extraction requires the CUDA
# rasterizer anyway. Without it the code still runs, falling back to
# scipy.spatial.Delaunay with a warning.
say "3/3  geodel  (fast Delaunay; falls back to scipy if absent)"
if GEODEL_DIR="$(find_dir GeoDel)"; then
    pip install "${BUILD_FLAGS[@]}" -c "$PIN_FILE" "./$GEODEL_DIR"
else
    pip install "${BUILD_FLAGS[@]}" -c "$PIN_FILE" "git+https://github.com/Anttwo/GeoDel@${GEODEL_REF:-164a66a}"
fi

# ---------------------------------------------------------------------------
# Optional extensions
# ---------------------------------------------------------------------------

if [[ "$WITH_NVDIFFRAST" == 1 ]]; then
    say "optional  nvdiffrast  (UV-textured mesh export)"
    if NVDIFFRAST_DIR="$(find_dir nvdiffrast)"; then
        pip install "${BUILD_FLAGS[@]}" -c "$PIN_FILE" "./$NVDIFFRAST_DIR"
    else
        fail "nvdiffrast/ not found (looked in submodules/ and the repo root)." \
             "Fetch the submodule, or drop --with-nvdiffrast."
    fi
fi

if [[ "$WITH_DA3" == 1 ]]; then
    say "optional  Depth-Anything-3  (monodepth expert guidance)"
    if DA3_DIR="$(find_dir Depth-Anything-3)"; then
        # DA3 leaves moviepy unpinned, but moviepy 2.x removed moviepy.editor.
        pip install -c "$PIN_FILE" "moviepy<2"
        # NO --no-build-isolation here: DA3 builds with hatchling (a PEP 517
        # backend), and isolation is exactly what provisions it. Passing the
        # flag makes pip fail with "Cannot import 'hatchling.build'". The flag
        # is only for packages whose setup.py imports torch itself.
        pip install -c "$PIN_FILE" -e "./$DA3_DIR"
    else
        fail "Depth-Anything-3/ not found (looked in submodules/ and the repo root)." \
             "Fetch the submodule, or drop --with-da3."
    fi
fi

say "Done"
echo "Verify what is now available with:"
echo "    python install/verify_install.py"
