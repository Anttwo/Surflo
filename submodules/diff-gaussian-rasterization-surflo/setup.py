#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# NOTE(surflo): this single extension merges two upstream rasterizers:
#   * cuda_rasterizer/      -> the RaDe-GS render pipeline (diff-gaussian-rasterization),
#                              extended here to render 3 OR 6 channels (RGB + learned
#                              normals) in one forward/backward pass.
#   * cuda_rasterizer_occ/  -> the occupancy pipeline (diff-gaussian-rasterization_ours),
#                              copied verbatim and wrapped in `namespace dgr_occ` so its
#                              symbols do not clash with the render pipeline at link time.
# Both are compiled into a single `diff_gaussian_rasterization_surflo._C`.
#
# nvcc flags mirror diff-gaussian-rasterization_ours (notably --use_fast_math). If you
# require the render path to be bit-identical to the stock diff-gaussian-rasterization
# package (which builds WITHOUT --use_fast_math), drop --use_fast_math below; the
# occupancy path would then differ from diff-gaussian-rasterization_ours in the low bits.

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

glm_include = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")

setup(
    name="diff_gaussian_rasterization_surflo",
    packages=['diff_gaussian_rasterization_surflo'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_surflo._C",
            sources=[
                # RaDe-GS render pipeline (3-or-6 channel)
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                # Occupancy pipeline (namespaced dgr_occ, verbatim from _ours)
                "cuda_rasterizer_occ/rasterizer_impl.cu",
                "cuda_rasterizer_occ/render_forward.cu",
                "cuda_rasterizer_occ/render_backward.cu",
                "cuda_rasterizer_occ/sample_forward.cu",
                "cuda_rasterizer_occ/sample_backward.cu",
                # Shared torch bindings
                "rasterize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    # "--use_fast_math",
                    "-std=c++17",
                    "--extended-lambda",
                    "--expt-relaxed-constexpr",
                    "-I" + glm_include,
                ],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
