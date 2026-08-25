import logging
from typing import List, Optional, Union

import torch

import nvdiffrast.torch as dr
from surflo.structures.cameras import Camera, transform_points_world_to_view
from surflo.structures.mesh import Meshes

_log = logging.getLogger(__name__)


def nvdiff_rasterization(
    camera,
    image_height: int,
    image_width: int,
    verts: torch.Tensor,
    faces: torch.Tensor,
    return_indices_only: bool = False,
    glctx=None,
    return_rast_out: bool = False,
    return_positions: bool = False,
):
    device = verts.device

    # Get full projection matrix
    camera_mtx = camera.full_proj_transform

    # Convert to homogeneous coordinates
    pos = torch.cat([verts, torch.ones([verts.shape[0], 1], device=device)], axis=1)

    # Transform points to NDC/clip space
    pos = torch.matmul(pos, camera_mtx)[None]

    # Rasterize with NVDiffRast
    # TODO: WARNING: pix_to_face is not in the correct range [-1, F-1] but in [0, F],
    # With 0 indicating that no triangle was hit.
    # So we need to subtract 1.
    rast_out, _ = dr.rasterize(glctx, pos=pos, tri=faces, resolution=[image_height, image_width])
    bary_coords, zbuf, pix_to_face = rast_out[..., :2], rast_out[..., 2], rast_out[..., 3].int() - 1

    if return_indices_only:
        return pix_to_face

    _output = (bary_coords, zbuf, pix_to_face)
    if return_rast_out:
        _output = _output + (rast_out,)
    if return_positions:
        _output = _output + (pos,)
    return _output


class Fragments:
    def __init__(self, bary_coords, zbuf, pix_to_face):
        self.bary_coords = bary_coords  # Shape (1, height, width, 1, 3)
        self.zbuf = zbuf  # Shape (1, height, width, 1)
        self.pix_to_face = pix_to_face  # Shape (1, height, width, 1)


class RasterizationSettings:
    """
    Rasterization settings for meshes.
    """

    def __init__(
        self,
        image_size=(1080, 1920),
        blur_radius=0.0,
        faces_per_pixel=1,
    ):
        self.image_size = image_size
        self.blur_radius = blur_radius
        self.faces_per_pixel = faces_per_pixel


class MeshRasterizer(torch.nn.Module):
    """
    Class for rasterizing meshes with NVDiffRast.
    """

    def __init__(
        self,
        cameras: Union[List[Camera], Camera] = None,
        raster_settings: RasterizationSettings = None,
        use_opengl=True,
    ):
        super().__init__()

        if cameras is None:
            if raster_settings is None:
                raster_settings = RasterizationSettings()
            self.raster_settings = raster_settings
            self.height, self.width = raster_settings.image_size
            self.cameras = None
        else:
            if isinstance(cameras, Camera):
                cameras = [cameras]
            # Get height and width if provided in cameras
            self.height = cameras[0].image_height
            self.width = cameras[0].image_width
            self.raster_settings = RasterizationSettings(
                image_size=(self.height, self.width),
            )
            self.cameras = cameras

        if use_opengl:
            self.gl_context = dr.RasterizeGLContext()
        else:
            self.gl_context = dr.RasterizeCudaContext()

    def forward(
        self,
        mesh: Meshes,
        cameras: List[Camera] = None,
        cam_idx: int = 0,
        return_only_pix_to_face: bool = False,
        return_rast_out: bool = False,
        return_positions: bool = False,
    ):
        if cameras is None:
            if self.cameras is None:
                raise ValueError("cameras must be provided either in the constructor or in the forward method")
            cameras = self.cameras

        if isinstance(cameras, Camera):
            render_camera = cameras
        else:
            render_camera = cameras[cam_idx]

        height, width = render_camera.image_height, render_camera.image_width
        nvdiff_rast_out = nvdiff_rasterization(
            camera=render_camera,
            image_height=height,
            image_width=width,
            verts=mesh.verts,
            faces=mesh.faces,
            return_indices_only=False,
            glctx=self.gl_context,
            return_rast_out=return_rast_out,
            return_positions=return_positions,
        )
        bary_coords, zbuf, pix_to_face = nvdiff_rast_out[:3]
        if return_rast_out:
            rast_out = nvdiff_rast_out[3]
        if return_positions:
            pos = nvdiff_rast_out[4]

        if return_only_pix_to_face:
            return pix_to_face.view(1, height, width, 1)
        bary_coords = torch.cat([bary_coords, 1.0 - bary_coords.sum(dim=-1, keepdim=True)], dim=-1)

        # TODO: Zbuf is still in NDC space, should convert to camera space
        fragments = Fragments(
            bary_coords.view(1, height, width, 1, 3),
            zbuf.view(1, height, width, 1),
            pix_to_face.view(1, height, width, 1),
        )
        _output = (fragments,)
        if return_rast_out:
            _output = _output + (rast_out,)
        if return_positions:
            _output = _output + (pos,)
        return _output


class MeshRenderer(torch.nn.Module):
    """
    Class for rendering meshes with NVDiffRast and a shader.
    """

    def __init__(self, rasterizer: MeshRasterizer):
        super().__init__()
        self.rasterizer = rasterizer
        # TODO: Add shader

    def forward(
        self,
        mesh: Meshes,
        cameras: Union[List[Camera], Camera] = None,
        cam_idx=0,
        return_depth=False,
        return_normals=False,
        use_antialiasing=True,
        return_pix_to_face=False,
        check_errors=True,
    ):
        fragments, rast_out, pos = self.rasterizer(mesh, cameras, cam_idx, return_rast_out=True, return_positions=True)
        if cameras is None:
            cameras = self.rasterizer.cameras
        if isinstance(cameras, Camera):
            cameras = [cameras]

        return_colors = mesh.verts_colors is not None

        output_pkg = {}

        # Compute per-vertex features to render
        features = torch.zeros(mesh.verts.shape[0], 0, device=mesh.verts.device)

        if return_depth:
            depth_idx = features.shape[-1]
            verts_depth = transform_points_world_to_view(mesh.verts, [cameras[cam_idx]])[..., 2].squeeze()  # Shape (N, )
            features = torch.cat([features, verts_depth.view(mesh.verts.shape[0], 1)], dim=-1)

        if return_colors:
            color_idx = features.shape[-1]
            features = torch.cat([features, mesh.verts_colors], dim=-1)  # Shape (N, n_features)

        # Compute image
        feature_img, _ = dr.interpolate(features[None], rast_out, mesh.faces)  # Shape (1, H, W, n_features)

        # Antialiasing for propagating gradients
        if use_antialiasing:
            feature_img = dr.antialias(feature_img, rast_out, pos, mesh.faces)  # Shape (1, H, W, n_features)

        if return_depth:
            output_pkg["depth"] = feature_img[..., depth_idx : depth_idx + 1]  # Shape (1, H, W)
        if return_colors:
            output_pkg["rgb"] = feature_img[..., color_idx : color_idx + 3]  # Shape (1, H, W, 3)

        # Compute per-face normals
        if return_normals:
            valid_mask = fragments.pix_to_face >= 0  # Shape (1, H, W, 1)
            if check_errors:
                error_mask = fragments.pix_to_face >= mesh.faces.shape[0]
                error_encountered = torch.sum(error_mask)
                if error_encountered > 0:
                    _log.warning(f"Rasterized {error_encountered} pixels with invalid triangle index.")
                    fragments.pix_to_face = torch.clamp(fragments.pix_to_face, min=0, max=mesh.faces.shape[0] - 1)
                    valid_mask = valid_mask & ~error_mask
            output_pkg["normals"] = mesh.face_normals[fragments.pix_to_face].squeeze()[None] * valid_mask  # Shape (1, H, W, 3)
            # if use_antialiasing:
            #     output_pkg["normals"] = dr.antialias(output_pkg["normals"], rast_out, pos, mesh.faces)  # Shape (1, H, W, 3)

        if return_pix_to_face:
            output_pkg["pix_to_face"] = fragments.pix_to_face

        return output_pkg


def fuse_fragments(fragments1: Fragments, fragments2: Fragments):
    raster_mask1 = fragments1.pix_to_face > -1
    raster_mask2 = fragments2.pix_to_face > -1

    # raster_mask1 = fragments1.zbuf > 0.
    # raster_mask2 = fragments2.zbuf > 0.

    no_raster_mask = (~raster_mask1) & (~raster_mask2)

    zbuf1 = torch.where(raster_mask1, fragments1.zbuf, 1000.0)
    zbuf2 = torch.where(raster_mask2, fragments2.zbuf, 1000.0)

    all_zbufs = torch.cat([zbuf1[..., None], zbuf2[..., None]], dim=-1)  # Shape (1, H, W, 1, 2)
    zbuf, argzbuf = torch.min(all_zbufs, dim=-1)  # argzbuf is of shape (1, H, W, 1)
    zbuf[no_raster_mask] = 0.0

    all_pix_to_face = torch.cat([fragments1.pix_to_face[..., None], fragments2.pix_to_face[..., None]], dim=-1)  # Shape (1, H, W, 1, 2)
    pix_to_face = torch.gather(all_pix_to_face, dim=-1, index=argzbuf[..., None])[..., 0]  # Shape (1, H, W, 1)
    pix_to_face[no_raster_mask] = -1

    all_bary_coords = torch.cat([fragments1.bary_coords[..., None], fragments2.bary_coords[..., None]], dim=-1)
    bary_coords = torch.gather(all_bary_coords, dim=-1, index=argzbuf[..., None, None].expand(-1, -1, -1, -1, 3, -1))[
        ..., 0
    ]  # Shape (1, H, W, 1, 3)

    return Fragments(
        bary_coords,
        zbuf,
        pix_to_face,
    )


class ScalableMeshRenderer(torch.nn.Module):
    """
    Class for rendering big meshes with NVDiffRast.
    """

    def __init__(self, rasterizer: MeshRasterizer):
        super().__init__()
        self.rasterizer = rasterizer
        # TODO: Add shader

    def forward(
        self,
        mesh: Meshes,
        cameras: Union[List[Camera], Camera] = None,
        cam_idx: int = 0,
        return_depth: bool = False,
        return_normals: bool = False,
        use_antialiasing: bool = True,
        return_pix_to_face: bool = False,
        check_errors: bool = True,
        max_triangles_in_batch: int = 2**24,  # Corresponds to the nb of triangles above which Nvdiffrast breaks
    ):
        n_passes = (mesh.faces.shape[0] + max_triangles_in_batch - 1) // max_triangles_in_batch

        fragments = None
        idx_shift = 0
        for i_pass in range(n_passes):
            start_idx = i_pass * max_triangles_in_batch
            end_idx = min(start_idx + max_triangles_in_batch, mesh.faces.shape[0])

            # Compute submesh
            sub_faces = mesh.faces[start_idx:end_idx]
            sub_mesh = Meshes(verts=mesh.verts, faces=sub_faces)

            # Rasterize submesh
            _fragments, _, pos = self.rasterizer(sub_mesh, cameras, cam_idx, return_rast_out=True, return_positions=True)

            # Combine fragments
            if fragments is None:
                fragments = _fragments
            else:
                # _fragments.pix_to_face = _fragments.pix_to_face + idx_shift
                _fragments.pix_to_face = torch.where(
                    _fragments.pix_to_face > -1, _fragments.pix_to_face + idx_shift, _fragments.pix_to_face
                )
                fragments = fuse_fragments(fragments, _fragments)
                # fragments = _fragments

            # Update idx shift
            idx_shift = idx_shift + len(sub_faces)

        # Filter mesh and fragments to keep only rasterized faces. This will decrease the number of faces to at most H * W.
        # Reducing the number of faces is necessary to avoid errors when running dr.interpolate and dr.antialias
        if True:
            filtered_face_idx, filtered_pix_to_face = fragments.pix_to_face.unique(return_inverse=True)
            filtered_face_idx = filtered_face_idx[1:]
            filtered_faces = mesh.faces[filtered_face_idx]
            filtered_pix_to_face = filtered_pix_to_face - 1
            mesh = Meshes(verts=mesh.verts, faces=filtered_faces, verts_colors=mesh.verts_colors)
            fragments.pix_to_face = filtered_pix_to_face

        # Rebuild rast_out
        rast_out = torch.zeros(*fragments.zbuf.shape[:-1], 4, device=fragments.zbuf.device)
        rast_out[..., :2] = fragments.bary_coords[..., 0, :2]
        rast_out[..., 2:3] = fragments.zbuf
        rast_out[..., 3:4] = fragments.pix_to_face.float() + 1

        if cameras is None:
            cameras = self.rasterizer.cameras
        if isinstance(cameras, Camera):
            cameras = [cameras]

        return_colors = mesh.verts_colors is not None

        output_pkg = {}

        # Compute per-vertex features to render
        features = torch.zeros(mesh.verts.shape[0], 0, device=mesh.verts.device)

        if return_depth:
            depth_idx = features.shape[-1]
            verts_depth = transform_points_world_to_view(mesh.verts, [cameras[cam_idx]])[..., 2].squeeze()  # Shape (N, )
            features = torch.cat([features, verts_depth.view(mesh.verts.shape[0], 1)], dim=-1)

        if return_colors:
            color_idx = features.shape[-1]
            features = torch.cat([features, mesh.verts_colors], dim=-1)  # Shape (N, n_features)

        # Compute image
        if True:
            feature_img, _ = dr.interpolate(features[None], rast_out, mesh.faces)  # Shape (1, H, W, n_features)
        else:
            pix_to_verts = mesh.faces[fragments.pix_to_face]  # Shape (1, H, W, 1, 3)
            pix_to_features = features[pix_to_verts]  # Shape (1, H, W, 1, 3, n_features)
            feature_img = (pix_to_features * fragments.bary_coords[..., None]).sum(dim=-2)  # Shape (1, H, W, 1, n_features)
            feature_img = feature_img.squeeze(-2)  # Shape (1, H, W, n_features)

        # Antialiasing for propagating gradients
        if use_antialiasing:
            feature_img = dr.antialias(feature_img, rast_out, pos, mesh.faces)  # Shape (1, H, W, n_features)

        if return_depth:
            output_pkg["depth"] = feature_img[..., depth_idx : depth_idx + 1]  # Shape (1, H, W)

        if return_colors:
            output_pkg["rgb"] = feature_img[..., color_idx : color_idx + 3]  # Shape (1, H, W, 3)

        # Compute per-face normals
        if return_normals:
            valid_mask = fragments.pix_to_face >= 0  # Shape (1, H, W, 1)
            if check_errors:
                error_mask = fragments.pix_to_face >= mesh.faces.shape[0]
                error_encountered = torch.sum(error_mask)
                if error_encountered > 0:
                    _log.warning(f"Rasterized {error_encountered} pixels with invalid triangle index.")
                    fragments.pix_to_face = torch.clamp(fragments.pix_to_face, min=0, max=mesh.faces.shape[0] - 1)
                    valid_mask = valid_mask & ~error_mask
            output_pkg["normals"] = mesh.face_normals[fragments.pix_to_face].squeeze()[None] * valid_mask  # Shape (1, H, W, 3)
            # if use_antialiasing:
            #     output_pkg["normals"] = dr.antialias(output_pkg["normals"], rast_out, pos, mesh.faces)  # Shape (1, H, W, 3)

        if return_pix_to_face:
            output_pkg["pix_to_face"] = fragments.pix_to_face

        #### TO REMOVE
        output_pkg["fragments"] = fragments
        output_pkg["rast_out"] = rast_out
        #### TO REMOVE

        return output_pkg
