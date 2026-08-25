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

#pragma once

#include <cstdint>
#include "rasterizer.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

namespace dgr_occ {
namespace CudaRasterizer {
template <typename T>
static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment) {
    std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
    ptr                = reinterpret_cast<T*>(offset);
    chunk              = reinterpret_cast<char*>(ptr + count);
}

struct GeometryState {
    size_t scan_size;
    float* depths;
    float4* ray_planes;
    float3* normals;
    char* scanning_space;
    bool* clamped;
    int* internal_radii;
    float2* means2D;
    float4* conic_opacity;
    float* rgb;
    uint32_t* point_offsets;
    uint32_t* tiles_touched;

    static GeometryState fromChunk(char*& chunk, size_t P);
};

struct GeometryBwdState {
    float4* ray_planes;
    float3* normals;
    float4* conic_opacity;
    static GeometryBwdState fromChunk(char*& chunk, size_t P);
};

struct PointState {
    size_t scan_size;
    float* depths;
    float2* points2D;
    uint32_t* n_contrib;
    float* median_depth;
    float* final_T;
    char* scanning_space;
    uint32_t* point_offsets;
    uint32_t* tiles_touched;

    static PointState fromChunk(char*& chunk, size_t P);
};

template <bool SAMPLE>
struct TileState;

template<>
struct TileState<true> {
    size_t scan_size;
    uint2* gaussian_ranges;
    uint2* point_ranges;
    uint32_t* tile_rounds;
    uint32_t* tile_offsets;
    char* scanning_space;

    static TileState<true> fromChunk(char*& chunk, size_t N);
};

template<>
struct TileState<false> {
    uint2* gaussian_ranges;
    uint32_t* max_contributor;

    static TileState<false> fromChunk(char*& chunk, size_t N);
};

struct DuplicatedTileState {
    uint32_t* tile_id;
    uint32_t* max_contributor;
    static DuplicatedTileState fromChunk(char*& chunk, size_t P);
};

struct ImageState {
    uint32_t* n_contrib;
    float* normal_length;

    static ImageState fromChunk(char*& chunk, size_t N);
};

struct BinningState {
    size_t sorting_size;
    uint64_t* point_list_keys_unsorted;
    uint64_t* point_list_keys;
    uint32_t* point_list_unsorted;
    uint32_t* point_list;
    char* list_sorting_space;

    static BinningState fromChunk(char*& chunk, size_t P);
};

template <typename T>
size_t required(size_t P) {
    char* size = nullptr;
    T::fromChunk(size, P);
    return ((size_t)size) + 128;
}
}; // namespace CudaRasterizer
} // namespace dgr_occ