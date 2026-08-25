import logging
import sys
import os
from typing import List, Dict, Any, Tuple, Optional, Union
import json
import pathlib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF
from surflo.structures.cameras import Camera
from surflo.structures.cameras import transform_points_world_to_view as transform_points_world_to_view_cameralist
from surflo.structures.cameras import transform_points_view_to_world as transform_points_view_to_world_cameralist
from surflo.structures.multi_cameras import MultiCameras
from surflo.structures.multi_cameras import transform_points_world_to_view as transform_points_world_to_view_multicameras
from surflo.structures.multi_cameras import transform_points_view_to_world as transform_points_view_to_world_multicameras

_log = logging.getLogger(__name__)


# TODO: Change how the image is resized.
# It should also be cropped to match multiples of 14 rather than being simply resized.
def load_and_preprocess_images(image_path_list, mode="no_stretch", target_ratio=None, target_size=518, rotate_portrait=False):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.
        target_ratio (float, optional): Target aspect ratio (width / height). Defaults to None.
        rotate_portrait (bool, optional): If True, portrait images (height > width)
                             are rotated 90 degrees to landscape before any resizing,
                             and a warning is logged for each one. The model is trained
                             on landscape images only, so this reduces the orientation
                             bias at inference time. Defaults to False. It should stay
                             False for training-data preprocessing, whose portrait views
                             come with their own (un-rotated) COLMAP cameras. Note that
                             enabling it puts the whole reconstruction in a rotated world
                             frame (the world frame is the first camera's frame).

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad", "no_stretch"]:
        raise ValueError("Mode must be either 'crop', 'pad' or 'no_stretch'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()

    # First process all images and collect their shapes
    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        # The model is trained on landscape images (width > height) and is biased
        # toward that orientation. When enabled (inference only), rotate portrait
        # inputs to landscape BEFORE the dimensions are read, so all the resizing
        # / cropping below operates on the landscape image with no other change.
        if rotate_portrait and img.height > img.width:
            _log.warning(
                f"[load_and_preprocess_images] rotate_portrait=True: rotating "
                f"portrait image {img.width}x{img.height} -> landscape (the model "
                f"is trained on landscape images): {image_path}"
            )
            img = img.transpose(Image.Transpose.ROTATE_90)

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        elif mode == "crop":
            # Fix the width to target_size, then derive the height.
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14
        elif mode == "no_stretch":
            max_side = max(width, height)
            size_ratio = target_size / max_side
            new_width = round(width * size_ratio)
            new_height = round(height * size_ratio)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
                
        if mode == "no_stretch":
            cx, cy = new_width // 2, new_height // 2
            if target_ratio is None:
                if new_height % 14 != 0:
                    new_height = int(height * size_ratio / 14) * 14
                if new_width % 14 != 0:
                    new_width = int(width * size_ratio / 14) * 14
            else:
                if new_width % 14 != 0:
                    new_width = int(width * size_ratio / 14) * 14
                ratio_height = int(width / target_ratio * size_ratio / 14) * 14
                if ratio_height > new_height:
                    raise ValueError(f"Target ratio {target_ratio} is too small for image {image_path}!")
                else:
                    new_height = ratio_height
            img = img[:, cy - new_height // 2 : cy + new_height // 2, cx - new_width // 2 : cx + new_width // 2]

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        _log.info(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


def load_guidance_images(
    image_path_list: List[str],
    target_aspect_ratio: float,
    max_width: int = 1600,
) -> torch.Tensor:
    """Load high-resolution images and crop them so their shape matches a target aspect ratio.

    The returned images are intended as supervision targets for the rendering loss
    in :func:`surflo.inference.engine.guided_inference` while the lower-resolution
    518-px VGGT inputs are kept for the encoder. They are essentially higher-res
    versions of the VGGT inputs: same field of view, same center, just more pixels.
    They are NOT fed to VGGT, so there is no patch-grid divisibility constraint
    on their pixel dimensions.

    Each image is processed as follows:
        1. Open and convert to RGB (alpha is composited over a white background).
        2. If ``width > max_width``, the image is downscaled to ``width=max_width``
           with bicubic resampling, preserving the original aspect ratio. Otherwise
           the image is used as-is.
        3. Center-cropped to the largest ``(W, H)`` inside the source such that
           ``W / H == target_aspect_ratio`` (rounded to the nearest pixel).

    Args:
        image_path_list: list of image paths, one per camera. They are processed in order.
        target_aspect_ratio: target ``W / H`` aspect ratio. Should match the VGGT input
            aspect ratio so that the high-res images are spatial supersets of the
            VGGT preprocessed images (modulo a center crop).
        max_width: cap on the final width. Images wider than this are downscaled to this
            width before cropping. Defaults to 1600.

    Returns:
        torch.Tensor of shape ``(N, 3, H, W)`` with ``W / H ≈ target_aspect_ratio``.
        All ``N`` images share the same ``(H, W)``.

    Raises:
        ValueError: if the list is empty, if ``target_aspect_ratio`` is non-positive,
            or if cropping would produce a degenerate (<1 px) shape.
    """
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    if target_aspect_ratio <= 0:
        raise ValueError(f"target_aspect_ratio must be positive, got {target_aspect_ratio}")
    if max_width <= 0:
        raise ValueError(f"max_width must be positive, got {max_width}")

    to_tensor = TF.ToTensor()
    images: List[torch.Tensor] = []
    shapes = set()

    for image_path in image_path_list:
        img = Image.open(image_path)

        # Composite alpha over white, then drop the alpha channel.
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")

        width, height = img.size  # PIL convention: (W, H)

        # Step 1 / 2: downscale to ``max_width`` if the source is wider; keep as-is otherwise.
        if width > max_width:
            scale = max_width / width
            new_w = max_width
            new_h = max(1, round(height * scale))
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            width, height = new_w, new_h

        # Step 3: pick the largest (W, H) inside the current image such that
        # ``W / H == target_aspect_ratio``. This is just the standard center-crop
        # for matching an aspect ratio: if the image is wider than the target ratio
        # we trim the width, otherwise we trim the height.
        cur_ratio = width / height
        if cur_ratio >= target_aspect_ratio:
            target_h = height
            target_w = int(round(target_h * target_aspect_ratio))
        else:
            target_w = width
            target_h = int(round(target_w / target_aspect_ratio))

        # Defensive clamp: rounding can push target_w / target_h one pixel above
        # the source on edge cases.
        target_w = min(target_w, width)
        target_h = min(target_h, height)
        if target_w < 1 or target_h < 1:
            raise ValueError(
                f"Image '{image_path}' is too small for cropping: "
                f"got target=({target_w}, {target_h})."
            )

        img_t = to_tensor(img)  # (3, H, W) float in [0, 1]
        h, w = img_t.shape[1], img_t.shape[2]

        # Symmetric center crop. Mirrors the integer-division convention used by
        # ``load_and_preprocess_images`` so the high-res crop stays centered on
        # the same pixel as the VGGT preprocessing.
        cy, cx = h // 2, w // 2
        top = cy - target_h // 2
        left = cx - target_w // 2
        img_t = img_t[:, top:top + target_h, left:left + target_w]

        shapes.add((img_t.shape[1], img_t.shape[2]))
        images.append(img_t)

    # All images should already share the same target shape; pad to a common shape
    # only as a defensive fallback (matches ``load_and_preprocess_images``).
    if len(shapes) > 1:
        _log.warning(f"load_guidance_images: heterogeneous shapes after crop: {shapes}")
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)
        padded: List[torch.Tensor] = []
        for img_t in images:
            h, w = img_t.shape[1], img_t.shape[2]
            pad_h = max_h - h
            pad_w = max_w - w
            if pad_h > 0 or pad_w > 0:
                img_t = torch.nn.functional.pad(
                    img_t,
                    (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                    mode="constant", value=1.0,
                )
            padded.append(img_t)
        images = padded

    return torch.stack(images, dim=0)  # (N, 3, H, W)


def load_colmap_cameras(
    camera_file:str, 
    target_size:int,
    image_names:list[str],
    device:torch.device="cpu",
    size_divider:Optional[int]=14,
    target_ratio:Optional[float]=None,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Load Colmap cameras from a file.

    Args:
        camera_file (str): Path to the Colmap cameras file.
        target_resolution (int): Target resolution of the images.
        image_names (list[str]): List of image names.
        device (torch.device, optional): Device to load the cameras to. Defaults to "cpu".
        size_divider (int, optional): Size multiple to use for the images. Defaults to None.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]: Tuple of intrinsics, extrinsics, and the new height and width. 
            Intrinsics of shape (N, 3, 3) and extrinsics of shape (N, 3, 4). 
            The new height and width are the dimensions of the resized images.
    """
    if size_divider is not None:
        assert target_size % size_divider == 0, f"Target size must be divisible by size multiple {size_divider}!"
    
    with open(camera_file, "r") as f:
        cameras = json.load(f)
    
    img_names_stems = [pathlib.Path(name).stem for name in image_names]
    
    # Create a dictionary for quick lookup
    camera_dict = {camera["img_name"]: camera for camera in cameras if camera["img_name"] in img_names_stems}
    
    positions = torch.tensor(
        [camera_dict[img_name]["position"] for img_name in img_names_stems],
        dtype=torch.float32
    ).to(device)  # (N, 3)
    orientations = torch.tensor(
        [camera_dict[img_name]["rotation"] for img_name in img_names_stems],
        dtype=torch.float32
    ).to(device)  # (N, 3, 3)

    R = orientations.transpose(-1, -2)  # (N, 3, 3)
    t = -R @ positions.unsqueeze(-1)  # (N, 3, 1)
    extrinsics = torch.cat((R, t), dim=-1)  # (N, 3, 4)

    fx = torch.tensor(
        [camera_dict[img_name]["fx"] for img_name in img_names_stems],
        dtype=torch.float32
    ).to(device)  # (N,)
    fy = torch.tensor(
        [camera_dict[img_name]["fy"] for img_name in img_names_stems],
        dtype=torch.float32
    ).to(device)  # (N,)
    
    width = cameras[0]["width"]
    height = cameras[0]["height"]
    max_side = max(width, height)
    size_ratio = target_size / max_side

    new_width = round(width * size_ratio)
    new_height = round(height * size_ratio)
    
    # In practice, only one of the following two conditions can be true.
    if target_ratio is None:
        if size_divider is not None:
            if new_height % size_divider != 0:
                new_height = int(height * size_ratio / size_divider) * size_divider
            if new_width % size_divider != 0:
                new_width = int(width * size_ratio / size_divider) * size_divider
    else:
        if size_divider is not None:
            if new_width % size_divider != 0:
                new_width = int(width * size_ratio / size_divider) * size_divider
            ratio_height = int(width / target_ratio * size_ratio / size_divider) * size_divider
        else:
            ratio_height = int(width / target_ratio * size_ratio)
        if ratio_height > new_height:
            raise ValueError(f"Target ratio {target_ratio} is too small for image {image_names[0]}!")
        else:
            new_height = ratio_height
    
    fx = fx * size_ratio
    fy = fy * size_ratio
    
    cx, cy = new_width / 2, new_height / 2
    
    intrinsics = torch.zeros((len(img_names_stems), 3, 3), dtype=torch.float32, device=device)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = cx
    intrinsics[..., 1, 2] = cy
    intrinsics[..., 2, 2] = 1.0
    
    return intrinsics, extrinsics, (new_height, new_width)


def load_sdf_data(
    sdf_dir:str, 
    device:str|torch.device ="cpu",
    chunk_index:Optional[int]=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load SDF data from a directory.

    Args:
        sdf_dir (str): Path to the directory containing the SDF data.
        device (torch.device, optional): Device to load the SDF data to. Defaults to "cpu".

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of points and SDF values. Points of shape (N, 3) and SDF values of shape (N,).
    """
    sdf_files = os.listdir(sdf_dir)
    if chunk_index is None:
        sdf_file = np.random.choice(sdf_files)
    else:
        sdf_file = sorted(sdf_files)[chunk_index]

    sdf_data = torch.load(os.path.join(sdf_dir, sdf_file), map_location=device)
    
    return sdf_data["points"], sdf_data["sdfs"]

def load_surface_point_data(
    point_dir:str, 
    device:str|torch.device ="cpu",
    chunk_index:Optional[int]=None,
    normal_dir:Optional[str]=None,
) -> torch.Tensor:
    """
    Load surface point data from a directory.

    Args:
        point_dir (str): Path to the directory containing the surface point data.
        device (torch.device, optional): Device to load the surface point data to. Defaults to "cpu".
        chunk_index (int, optional): Index of the chunk to load. Defaults to None.
        normal_dir (str, optional): Path to the directory containing the surface normal data. Defaults to None.

    Returns:
        torch.Tensor: Tensor of shape (N, 3) containing the surface points.
        
    Raises:
        RuntimeError: If surface point data cannot be loaded.
    """
    load_normals = normal_dir is not None
    
    point_files = os.listdir(point_dir)
    if load_normals:
        normal_files = os.listdir(normal_dir)
        assert len(point_files) == len(normal_files), f"Number of point and normal files must be the same! {len(point_files)=}, {len(normal_files)=}"
    
    if chunk_index is None:
        chunk_index = np.random.randint(0, len(point_files))
    
    point_file = sorted(point_files)[chunk_index]
    if load_normals:
        normal_file = sorted(normal_files)[chunk_index]

    point_data = torch.load(os.path.join(point_dir, point_file), map_location=device)
    if load_normals:
        normal_data = torch.load(os.path.join(normal_dir, normal_file), map_location=device)
        return point_data, normal_data
    else:
        return point_data


def load_depth_data(
    depth_dir:str, 
    image_names:list[str], 
    target_size:int,
    size_divider:int=14,
    device:torch.device="cpu",
    target_ratio:Optional[float]=None,
) -> torch.Tensor:
    """
    Load depth data from a directory.

    Args:
        depth_dir (str): Path to the directory containing the depth data.
        image_names (list[str]): List of image names.
        device (torch.device, optional): Device to load the depth data to. Defaults to "cpu".

    Returns:
        torch.Tensor: Tensor of shape (N, 1, H, W) containing the depth data.
    """
    if size_divider is not None:
        assert target_size % size_divider == 0, f"Target size must be divisible by size multiple {size_divider}!"
    
    depth_files = [
        os.path.join(depth_dir, pathlib.Path(name).stem + ".pth") 
        for name in image_names
    ]
    depth_data = [torch.load(file, map_location=device)[None] for file in depth_files]
    depth_data = torch.cat(depth_data, dim=0)  # (N, 1, H, W)
    
    height, width = depth_data.squeeze().shape[-2:]

    # Resize depth data to match target size
    max_side = max(width, height)
    size_ratio = target_size / max_side
    new_width = round(width * size_ratio)
    new_height = round(height * size_ratio)
    
    cx, cy = new_width // 2, new_height // 2
    
    # Interpolate depth data to new size
    depth_data = torch.nn.functional.interpolate(
        depth_data,
        size=(new_height, new_width),
        mode="nearest",
    )  # (N, 1, new_height, new_width)
    
    # Center crop depth data to match multiples of size_divider.
    # In practice, only one of the following two conditions can be true.
    if target_ratio is None:
        if size_divider is not None:
            if new_height % size_divider != 0:
                new_height = int(height * size_ratio /size_divider) * size_divider
            if new_width % size_divider != 0:
                new_width = int(width * size_ratio / size_divider) * size_divider
            
            depth_data = depth_data[
                :, :, 
                cy - new_height // 2 : cy + new_height // 2, 
                cx - new_width // 2 : cx + new_width // 2
            ]  # (N, 1, new_height, new_width)
    else:
        if size_divider is not None:
            if new_width % size_divider != 0:
                new_width = int(width * size_ratio / size_divider) * size_divider
            ratio_height = int(width / target_ratio * size_ratio / size_divider) * size_divider
        else:
            ratio_height = int(width / target_ratio * size_ratio)
        if ratio_height > new_height:
            raise ValueError(f"Target ratio {target_ratio} is too small for image {image_names[0]}!")
        else:
            new_height = ratio_height
        depth_data = depth_data[
            :, :, 
            cy - new_height // 2 : cy + new_height // 2, 
            cx - new_width // 2 : cx + new_width // 2
        ]  # (N, 1, new_height, new_width)
    
    return depth_data  # (N, 1, new_height, new_width)


def align_points_with_respect_to_main_camera(
    points: torch.Tensor,
    colmap_cameras: Union[List[List[Camera]], MultiCameras],
    vggt_cameras: Union[List[List[Camera]], MultiCameras],
    main_cam_idx: int=0,
    return_scale_ratio: bool=True,
) -> torch.Tensor:
    """
    Align points from COLMAP world space to VGGT world space, by relying on the main camera of each scene.

    Args:
        points (torch.Tensor): Points in COLMAP world space. Should have shape (B, P, 3).
        colmap_cameras (List[List[Camera]]): List of lists of COLMAP cameras. Should contain B elements, 
            where each element is a list of cameras for a given scene. Can be a MultiCameras object.
        vggt_cameras (List[List[Camera]]): List of lists of VGGT cameras. Should contain B elements, 
            where each element is a list of cameras for a given scene. Can be a MultiCameras object.
        main_cam_idx (int, optional): Index of the main camera. Defaults to 0.

    Returns:
        torch.Tensor: Points in VGGT world space. Has shape (B, P, 3).
    """
    B, P, _ = points.shape
    if isinstance(colmap_cameras, MultiCameras):
        # Get MultiCameras object for the main cameras, R must be (B, N, 3, 3)
        assert colmap_cameras.R.ndim == 4
        assert colmap_cameras.batch_size == B
        main_colmap_cameras = colmap_cameras.get_sub_cameras(dim=1, index=main_cam_idx)
        
        # Move points from COLMAP world to COLMAP camera space
        points_view = transform_points_world_to_view_multicameras(
            points=points,
            cameras=main_colmap_cameras,
        )  # (B, P, 3)
        
        # Get camera centers
        colmap_camera_centers = colmap_cameras.camera_center  # (B, N, 3)
        vggt_camera_centers = vggt_cameras.camera_center  # (B, N, 3)
    else:
        assert len(colmap_cameras) == B
        main_colmap_cameras = [colmap_cameras[i][main_cam_idx] for i in range(B)]  # List[Camera] with B elements
    
        # Move points from COLMAP world to COLMAP camera space
        points_view = transform_points_world_to_view_cameralist(
            points=points,
            cameras=main_colmap_cameras,
        )  # (B, P, 3)
    
        # Get camera centers
        colmap_camera_centers = torch.cat(
            [
                torch.cat(
                    [camera.camera_center.reshape(1, 1, 3) for camera in colmap_cameras[i]], dim=1,
                ) for i in range(B)
            ],
            dim=0,
        )  # (B, N, 3)
        vggt_camera_centers = torch.cat(
            [
                torch.cat(
                    [camera.camera_center.reshape(1, 1, 3) for camera in vggt_cameras[i]], dim=1,
                ) for i in range(B)
            ],
            dim=0,
        )  # (B, N, 3)
    
    # Scale points to match VGGT scale
    # colmap_scale = (colmap_camera_centers - colmap_camera_centers.mean(dim=1, keepdim=True)).norm(dim=-1).std(dim=-1)  # (B,)
    # vggt_scale = (vggt_camera_centers - vggt_camera_centers.mean(dim=1, keepdim=True)).norm(dim=-1).std(dim=-1)  # (B,)
    colmap_scale = (colmap_camera_centers - colmap_camera_centers.mean(dim=1, keepdim=True)).norm(dim=-1).mean(dim=-1)  # (B,)
    vggt_scale = (vggt_camera_centers - vggt_camera_centers.mean(dim=1, keepdim=True)).norm(dim=-1).mean(dim=-1)  # (B,)
    
    scale_is_zero = (colmap_scale == 0.0) | (vggt_scale == 0.0)
    if scale_is_zero.any():
        raise ValueError(f"[ERROR] Scale is zero for some cameras! {colmap_scale=}, {vggt_scale=}")
    
    scale_ratio = vggt_scale / colmap_scale  # (B,)
    points_view = points_view * scale_ratio.view(B, 1, 1)  # (B, P, 3)
    
    # Move points from VGGT camera to VGGT world space
    if isinstance(vggt_cameras, MultiCameras):
        # Get MultiCameras object for the main cameras, R must be (B, N, 3, 3)
        assert vggt_cameras.R.ndim == 4
        assert vggt_cameras.batch_size == B
        main_vggt_cameras = vggt_cameras.get_sub_cameras(dim=1, index=main_cam_idx)
        
        points_world = transform_points_view_to_world_multicameras(
            points=points_view,
            cameras=main_vggt_cameras,
        )  # (B, P, 3)
    else:
        assert len(vggt_cameras) == B
        main_vggt_cameras = [vggt_cameras[i][main_cam_idx] for i in range(B)]  # List[Camera] with B elements
        
        points_world = transform_points_view_to_world_cameralist(
            points=points_view,
            cameras=main_vggt_cameras,
        )  # (B, P, 3)
    
    return points_world, scale_ratio if return_scale_ratio else points_world


def find_affine_transform(
    X:torch.Tensor, 
    Y:torch.Tensor,
    weights:Optional[torch.Tensor]=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finds affine transform (L, T) such that Y = X @ L + T, with optional weights.

    Args:
        X (torch.Tensor): Has shape (..., N, d)
        Y (torch.Tensor): Has shape (..., N, d)
        weights (torch.Tensor): Has shape (..., N)

    Returns:
        M (torch.Tensor): Has shape (..., d+1, d+1)
        L (torch.Tensor): Has shape (..., d, d)
        T (torch.Tensor): Has shape (..., d)
    """
    d = X.shape[-1]
    
    _X = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    _Y = torch.cat([Y, torch.ones_like(Y[..., :1])], dim=-1)
    
    _X = _X if weights is None else _X * weights[..., None]
    _Y = _Y if weights is None else _Y * weights[..., None]

    M = torch.linalg.lstsq(_X, _Y).solution
    L = M[..., :d, :d]
    T = M[..., d, :d]
    
    return M, L, T


def transform_points(
    X:torch.Tensor,
    L:torch.Tensor=None,
    T:torch.Tensor=None,
    M:torch.Tensor=None
) -> torch.Tensor:
    """Transforms points X by the affine transform (L, T).
    A matrix M with shape (..., d+1, d+1) can also be provided, 
    in which case L and T are extracted from M.
    
    Args:
        X (torch.Tensor): Has shape (..., N, d)
        L (torch.Tensor): Has shape (..., d, d)
        T (torch.Tensor): Has shape (..., d)
        M (torch.Tensor): Has shape (..., d+1, d+1)

    Returns:
        Y (torch.Tensor): Has shape (..., N, d)
    """
    assert ((L is not None) and (T is not None)) or (M is not None)
    if M is not None:
        d = X.shape[-1]
        L = M[..., :d, :d]
        T = M[..., d, :d]

    return X @ L + T[..., None, :]


def find_robust_alignment_transform(
    colmap_world_points:torch.Tensor,
    vggt_world_points:torch.Tensor,
    colmap_cameras: Union[List[List[Camera]], MultiCameras],
    vggt_cameras: Union[List[List[Camera]], MultiCameras],
    main_cam_idx: int=0,
    spatial_std: Union[torch.Tensor, float]=10.0,
    weight_quantile_threshold: float=0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get a robust alignment transform between COLMAP and VGGT points.
    First, a coarse but robust alignment transform is computed using only the COLMAP and VGGT cameras, and applied to the COLMAP world points.
    Then, outliers are identified based on the resulting alignment error.
    Finally, a final, refined alignment transform is computed using only the filtered COLMAP and VGGT points.

    Args:
        colmap_world_points (torch.Tensor): Has shape (B, P, 3)
        vggt_world_points (torch.Tensor): Has shape (B, P, 3)
        colmap_cameras (Union[List[List[Camera]], MultiCameras]): List of lists of COLMAP cameras. Should contain B elements, 
            where each element is a list of cameras for a given scene. Can be a MultiCameras object for parallel processing.
        vggt_cameras (Union[List[List[Camera]], MultiCameras]): List of lists of VGGT cameras. Should contain B elements, 
            where each element is a list of cameras for a given scene. Can be a MultiCameras object for parallel processing.
        main_cam_idx (int, optional): Index of the main camera. Defaults to 0.
        spatial_std (Union[torch.Tensor, float], optional): Characteristic spatial scale (per-channel std or scalar). Used to normalize alignment error for weight computation. Defaults to 10.0.
        weight_quantile_threshold (float, optional): Quantile threshold for filtering outliers based on the weights. Defaults to 0.5.

    Returns: Tuple of (M, L, T), where:
        M (torch.Tensor): Has shape (..., 4, 4)
        L (torch.Tensor): Has shape (..., 3, 3)
        T (torch.Tensor): Has shape (..., 3)
    """
    B = colmap_world_points.shape[0]
    
    # Compute coarse alignment transform between COLMAP and VGGT cameras, and apply it to the colmap world points
    coarse_aligned_colmap_world_points, _ = align_points_with_respect_to_main_camera(
        points=colmap_world_points, 
        colmap_cameras=colmap_cameras, 
        vggt_cameras=vggt_cameras,
        main_cam_idx=main_cam_idx,
    )  # (B, P, 3)
    
    # Compute weights based on the coarse alignment transform
    if isinstance(spatial_std, torch.Tensor):
        characteristic_scale = spatial_std.mean().item()
    else:
        characteristic_scale = float(spatial_std)
    weights = (coarse_aligned_colmap_world_points - vggt_world_points).norm(dim=-1) / characteristic_scale  # (B, P)
    weights = torch.nn.Softmin(dim=-1)(weights)  # (B, P)
    
    # Filter out outliers based on the weights
    weights = torch.where(
        weights < torch.quantile(weights, q=weight_quantile_threshold, dim=-1, keepdim=True),  # (B, P)
        torch.zeros_like(weights),  # (B, P)
        weights,  # (B, P)
    )  
    
    # Compute the final alignment transform between the filtered COLMAP and VGGT points
    M, L, T = find_affine_transform(
        X=colmap_world_points,  # (B, P, 3)
        Y=vggt_world_points,  # (B, P, 3)
        weights=weights,  # (B, P)
    )  # (B, 4, 4), (B, 3, 3), (B, 3)
    
    return M, L, T