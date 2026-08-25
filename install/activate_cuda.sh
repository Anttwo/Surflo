#!/usr/bin/env bash
# Make a conda environment's own CUDA toolchain the one that builds extensions.
#
# Installs an activate.d hook (and a matching deactivate.d hook) into the
# ACTIVE conda environment so that CUDA_HOME, the compilers and the library
# path point at $CONDA_PREFIX every time the environment is activated.
#
# This has to be a hook rather than a one-off `export`: the variables would
# otherwise be lost on the next shell, and a later rebuild would silently pick
# up whatever nvcc happens to be on PATH — producing an extension that does
# not match the installed torch.
#
# Only needed for the self-contained conda environments
# (install/environment-cu*.yml). If you supply your own CUDA toolkit (the
# install/requirements-cu*.txt path), set CUDA_HOME yourself instead.
#
# Usage:
#   conda activate surflo-cu118
#   bash install/activate_cuda.sh
#   conda deactivate && conda activate surflo-cu118    # reload
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: no conda environment is active." >&2
    echo "       Run 'conda activate surflo-cu118' (or your env) first." >&2
    exit 1
fi

if [[ ! -x "$CONDA_PREFIX/bin/nvcc" ]]; then
    echo "ERROR: $CONDA_PREFIX/bin/nvcc not found." >&2
    echo "       This environment does not carry its own CUDA toolkit, so this" >&2
    echo "       script would point CUDA_HOME at a toolkit that isn't there." >&2
    echo "       Either create the env from install/environment-cu*.yml, or use" >&2
    echo "       the install/requirements-cu*.txt path and set CUDA_HOME by hand." >&2
    exit 1
fi

CC_BIN="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
CXX_BIN="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
if [[ ! -x "$CC_BIN" || ! -x "$CXX_BIN" ]]; then
    echo "ERROR: conda compilers not found in $CONDA_PREFIX/bin." >&2
    echo "       Expected gxx_linux-64 / sysroot_linux-64 (see the env file)." >&2
    exit 1
fi

ACTIVATE_D="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_D="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_D" "$DEACTIVATE_D"

cat > "$ACTIVATE_D/surflo-cuda.sh" <<'EOF'
# Written by install/activate_cuda.sh — points the CUDA build toolchain at
# this conda environment. Remove this file (and the matching one in
# deactivate.d) to undo.
export SURFLO_CUDA_HOOK_OLD_CUDA_HOME="${CUDA_HOME:-}"
export SURFLO_CUDA_HOOK_OLD_CC="${CC:-}"
export SURFLO_CUDA_HOOK_OLD_CXX="${CXX:-}"
export SURFLO_CUDA_HOOK_OLD_NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export SURFLO_CUDA_HOOK_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"

# conda installs the CUDA libraries into $CONDA_PREFIX/lib, NOT lib64 (which
# many CUDA instructions assume and which often does not exist here), so both
# are added and only if present. This keeps the environment self-contained:
# libcudart resolves from the env even though the torch wheel also bundles a
# copy. Both are the same CUDA version by construction, so whichever the
# loader reaches first is equivalent.
for _surflo_libdir in "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"; do
    [ -d "$_surflo_libdir" ] && export LD_LIBRARY_PATH="$_surflo_libdir:${LD_LIBRARY_PATH:-}"
done
unset _surflo_libdir

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
EOF

# Restore on deactivate so CC/CXX/CUDA_HOME do not leak into other
# environments in the same shell.
cat > "$DEACTIVATE_D/surflo-cuda.sh" <<'EOF'
# Written by install/activate_cuda.sh — see activate.d/surflo-cuda.sh.
_surflo_restore() {
    local name="$1" saved="$2"
    if [[ -z "$saved" ]]; then unset "$name"; else export "$name=$saved"; fi
}
_surflo_restore CUDA_HOME "${SURFLO_CUDA_HOOK_OLD_CUDA_HOME:-}"
_surflo_restore CC "${SURFLO_CUDA_HOOK_OLD_CC:-}"
_surflo_restore CXX "${SURFLO_CUDA_HOOK_OLD_CXX:-}"
_surflo_restore NVCC_PREPEND_FLAGS "${SURFLO_CUDA_HOOK_OLD_NVCC_PREPEND_FLAGS:-}"
_surflo_restore LD_LIBRARY_PATH "${SURFLO_CUDA_HOOK_OLD_LD_LIBRARY_PATH:-}"
unset SURFLO_CUDA_HOOK_OLD_CUDA_HOME SURFLO_CUDA_HOOK_OLD_CC
unset SURFLO_CUDA_HOOK_OLD_CXX SURFLO_CUDA_HOOK_OLD_NVCC_PREPEND_FLAGS
unset SURFLO_CUDA_HOOK_OLD_LD_LIBRARY_PATH
unset -f _surflo_restore
EOF

echo "Wrote $ACTIVATE_D/surflo-cuda.sh"
echo "Wrote $DEACTIVATE_D/surflo-cuda.sh"
echo
echo "  nvcc : $("$CONDA_PREFIX/bin/nvcc" --version | tail -1 | sed 's/^ *//')"
echo "  CXX  : $("$CXX_BIN" --version | head -1)"
echo
echo "Reload the environment for these to take effect:"
echo "  conda deactivate && conda activate $(basename "$CONDA_PREFIX")"
