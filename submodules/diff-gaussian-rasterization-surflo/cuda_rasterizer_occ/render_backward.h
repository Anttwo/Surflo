/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_OCC_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_OCC_BACKWARD_H_INCLUDED

#include <cstdint>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace dgr_occ {
namespace BACKWARD {
void render(
    const dim3 grid, const dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H,
    const float* bg_color,
    const float2* means2D,
    const float4* conic_opacity,
    const float* colors,
    const float* depths,
    const float4* ray_planes,
    const float3* normals,
    const float* alphas,
    const float* normalmap,
    const float* mdepth,
    const float* normal_length,
    const uint32_t* n_contrib,
    const uint32_t* max_contributors,
    const float* dL_dpixels,
    const float* dL_dpixel_mdepth,
    const float* dL_dalphas,
    const float* dL_dpixel_normals,
    const float focal_x,
    const float focal_y,
    float3* dL_dmean2D,
    float4* dL_dconic2D,
    float* dL_dcolors,
    float4* dL_dray_planes,
    float3* dL_dnormals,
    bool require_depth);

void preprocess(
    int P, int SHD, int SHM, int SGD, int SGM,
    const float3* means3D,
    const float* opacities,
    const glm::vec3* scales,
    const float4* rotations,
    const float* cov3Ds,
    const float* shs,
    const float* sg_axis,
    const float* sg_sharpness,
    const float* sg_color,
    const float scale_modifier,
    const float* viewmatrix,
    const float* projmatrix,
    const float focal_x,
    const float focal_y,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const glm::vec3* campos,
    const int* radii,
    const bool* clamped,
    const float4* dL_dconic,
    const float4* dL_dray_plane,
    const float3* dL_dnormals,
    const float3* dL_dmean2D,
    const float* dL_dcolor,
    glm::vec3* dL_dmean3D,
    float* dL_dopacity,
    glm::vec3* dL_dscale,
    glm::vec4* dL_drot,
    float* dL_dcov3D,
    float* dL_dsh,
    float* dL_dsg_axis,
    float* dL_dsg_sharpness,
    float* dL_dsg_color);
} // namespace BACKWARD
} // namespace dgr_occ

#endif