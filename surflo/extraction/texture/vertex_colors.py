import logging
from typing import List
import torch
from surflo.structures.cameras import Camera
from surflo.structures.mesh import Meshes
from surflo.rendering.mesh import MeshRasterizer, MeshRenderer, ScalableMeshRenderer
from surflo.data.surface_sampling import frustum_cull_mesh
from fused_ssim import fused_ssim
from tqdm import tqdm
from random import randint
import gc

_log = logging.getLogger(__name__)


def _l1_loss(pred:torch.Tensor, gt:torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - gt).mean()


def optimize_vertex_colors(
    mesh:Meshes,
    cameras:List[Camera],
    images:torch.Tensor,
    num_iterations:int=1000,
    learning_rate:float=0.0025,
    lambda_dssim:float=0.2,
    use_scalable_renderer:bool=True,
) -> Meshes:
    """
    Optimize the vertex colors of the mesh using the images and cameras.

    Args:
        mesh (Meshes): The mesh to optimize the vertex colors of.
        cameras (List[Camera]): The cameras to render the mesh from.
        images (torch.Tensor): The images to optimize the vertex colors for.
        num_iterations (int, optional): The number of iterations to run the optimization for. Defaults to 1000.
        learning_rate (float, optional): The learning rate to use for the optimization. Defaults to 0.0025.
        lambda_dssim (float, optional): The lambda value for the DSSIM loss. Defaults to 0.2.
        use_scalable_renderer (bool, optional): Whether to use the scalable renderer. Defaults to True.

    Returns:
        Meshes: The optimized mesh.
    """
    # Get device
    device = torch.device(torch.cuda.current_device())
    
    # Get mesh args
    _verts_colors = mesh.verts_colors.clone()
    _verts_colors = torch.nn.Parameter(_verts_colors, requires_grad=True).to(device=device)
    # Instantiates parameters and optimizer
    l = [{'params': [_verts_colors], 'lr': learning_rate, "name": "verts_colors"}]
    optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    
    # Define mesh renderer
    if use_scalable_renderer:
        renderer = ScalableMeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras)
        )
    else:
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras)
        )
        
    # Texture refinement
    viewpoint_idx_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(num_iterations), desc="Texture refinement progress")
    
    _log.info(f"Starting texture refinement with {num_iterations} iterations")
    for i_iter in range(num_iterations):
        # Get updated mesh
        updated_mesh = Meshes(
            verts=mesh.verts,
            faces=mesh.faces,
            verts_colors=0. + _verts_colors,
        )
        
        # Get random viewpoint
        if not viewpoint_idx_stack:
            viewpoint_idx_stack = list(range(len(cameras)))
        _random_view_idx = randint(0, len(viewpoint_idx_stack)-1)
        viewpoint_idx = viewpoint_idx_stack.pop(_random_view_idx)
        
        # Render frustum-culled mesh
        rendered_image = renderer(
            mesh=frustum_cull_mesh(updated_mesh, cameras[viewpoint_idx]),  # FIXME: Add znear
            cam_idx=viewpoint_idx,
            return_depth=True,
            return_normals=True,
            use_antialiasing=True,  
        )
        mesh_rgb = rendered_image['rgb'].squeeze(0).permute(2, 0, 1)
        
        gt_image = images[viewpoint_idx].to(device)
        Ll1 = _l1_loss(mesh_rgb, gt_image)
        ssim_value = fused_ssim(mesh_rgb.unsqueeze(0), gt_image.unsqueeze(0))
        loss = ((1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_value))
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            
            if i_iter % 5 == 0:
                postfix_dict = {}
                postfix_dict["Loss"] = f"{ema_loss_for_log:.{7}f}"
                progress_bar.set_postfix(postfix_dict)
                progress_bar.update(5)
        
        if i_iter % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
    _log.info(f"Texture refinement completed")
    
    # Create final mesh
    with torch.no_grad():
        # Get updated mesh
        updated_mesh = Meshes(
            verts=mesh.verts.detach(),
            faces=mesh.faces.detach(),
            verts_colors=_verts_colors.detach(),
        )
    
    return updated_mesh
